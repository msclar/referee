import os
import gc
import argparse
import torch

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    AutoModelForCausalLM, AutoModelForSequenceClassification

cache_dir = '/gscratch/xlab/msclar/.cache'
generated_datasets_dir = 'filtered-datasets'

if not torch.cuda.is_available():
    cache_dir = './.cache'


from pynvml import *


DELIMITER = ' TL;DR: '
END_TOKEN_GENERATIONS = 'Δ'  # tokenizer.eos_token


def print_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU_{i} memory occupied: {info.used//1024**2} MB.")


def format_realnews_dataset(dataset_input, tokenizer):
    dataset = dataset_input.map(
        lambda batch: tokenizer(batch["text"], max_length=200, truncation=True, padding="max_length"),
        batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset


def load_summary_dataset_as_pairs(summary_path, chunk_list=None):
    import re
    original_sentence_path = 'outputs/realnews_100k'
    summary_file_regex = re.compile('^summaries_chunk_[0-9]+.txt$')

    visited_chunks = set()
    dataset_lines = []
    for summary_filename in os.listdir(summary_path):
        if summary_file_regex.match(summary_filename):
            with open(os.path.join(summary_path, summary_filename), 'r') as f_summary:
                summaries = [line.strip() for line in f_summary.readlines()]

            chunk_id = int(summary_filename[len('summaries_chunk_'):-len('.txt')])
            original_sentence_filename = f'realnews_s1_chunk_{chunk_id}.txt'
            if chunk_list and chunk_id not in chunk_list:
                continue
            visited_chunks.add(chunk_id)
            with open(os.path.join(original_sentence_path, original_sentence_filename), 'r') as f_summary:
                original_sentences = [line.strip() for line in f_summary.readlines()]
                original_sentences = original_sentences[:len(summaries)]  # in case of bottleEx crashing

            for s1, s1_summary in zip(original_sentences, summaries):
                if len(s1) < 10 or len(s1_summary) < 1:
                    continue
                dataset_lines.append((s1, s1_summary))

    if chunk_list and len(chunk_list) != len(visited_chunks):
        print(f"Some chunks were not collected: visited {list(visited_chunks)}; wanted {chunk_list}")
        0 / 0

    return dataset_lines


def load_summary_dataset(summary_path, delim, end_token, chunk_list):
    dataset_lines = load_summary_dataset_as_pairs(summary_path, chunk_list=chunk_list)
    result = []
    for s1, s1_summary in dataset_lines:
        if delim in s1 or delim in s1_summary or end_token in s1 or end_token in s1_summary:
            print(s1, s1_summary)
            continue
        result.append(s1 + f' {delim} ' + s1_summary + end_token)

    return Dataset.from_dict({'text': result})


def main(args):
    if args.min_chunk_num == -1 and not args.chunk_list:
        chunk_list = None
    else:
        chunk_list = [int(t) for t in args.chunk_list.split(',')] if args.chunk_list else list(
            range(args.min_chunk_num, args.max_chunk_num + 1))

    print('chunk_list', chunk_list)

    dataset_lines = load_summary_dataset_as_pairs(args.summaries_dir, chunk_list=chunk_list)
    dataset = Dataset.from_dict(
        {'s1': [s for s, s_summ in dataset_lines], 's1_summary': [s_summ for s, s_summ in dataset_lines]}
    )

    # apply filters before running NLI model, since it is expensive
    dataset = dataset.filter(lambda x: DELIMITER not in x['s1'] and DELIMITER not in x['s1_summary'])
    dataset = dataset.filter(
        lambda x: END_TOKEN_GENERATIONS not in x['s1'] and END_TOKEN_GENERATIONS not in x['s1_summary'])
    print('prefilter', dataset)
    dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))
    print('postfilter', dataset)

    if args.filter_dataset_based_on_nli:
        model_wanli = AutoModelForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli', cache_dir=cache_dir)
        tokenizer_wanli = AutoTokenizer.from_pretrained('alisawuffles/roberta-large-wanli', cache_dir=cache_dir)

        dataset = dataset.map(
            lambda x: tokenizer_wanli(x["s1"], x["s1_summary"], max_length=200, truncation=True, padding="max_length"),
            batched=True)

        trainer = Trainer(model=model_wanli, eval_dataset=dataset)
        print(dataset)
        predictions_tmp = trainer.predict(dataset).predictions
        print(predictions_tmp)
        predicted_label_ids = predictions_tmp.argmax(axis=1).tolist()
        predictions = [model_wanli.config.id2label[p] for p in predicted_label_ids]
        dataset = dataset.add_column("wanli_prediction", predictions)

        model_name_collaged = args.summaries_dir.replace("/", "__")
        if len(model_name_collaged) > 110:
            model_name_collaged = model_name_collaged[:110] + '_etal'
        shortened_filename = f'nli_filtered_{model_name_collaged}_chunks_{args.min_chunk_num}_{args.max_chunk_num}.csv'
        dataset.to_csv(os.path.join(generated_datasets_dir, shortened_filename))

        dataset = dataset.filter(lambda x: x['wanli_prediction'] == 'entailment')
    elif args.train_from_wanli_gpt3_dataset:
        dataset = load_dataset("csv", data_files="nli_gpt3_chunks_1_3.csv", cache_dir=cache_dir)['train']
        dataset = dataset.filter(lambda x: x['wanli_prediction'] == 'entailment')
        dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))
        dataset = dataset.map(lambda x: {"text": x['s1'] + f' {DELIMITER} ' + x['s1_summary'] + END_TOKEN_GENERATIONS})
    else:
        pass
        #dataset = load_summary_dataset(
        #    args.summaries_dir, delim=DELIMITER, end_token=END_TOKEN_GENERATIONS, chunk_list=chunk_list)

    dataset = dataset.map(lambda x: {"text": x['s1'] + f' {DELIMITER} ' + x['s1_summary'] + END_TOKEN_GENERATIONS})
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir=cache_dir)
    if args.model_type.startswith('gpt2') or args.model_type.startswith('EleutherAI'):
        tokenizer.pad_token = tokenizer.eos_token  # required for GPT2 but not RoBERTa

    #dataset.to_csv(os.path.join(generated_datasets_dir, 'delimitered_1.csv'))
    dataset = format_realnews_dataset(dataset, tokenizer).shuffle(seed=42)
    # dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))
    print('postfilter', dataset)
    #dataset.to_csv(os.path.join(generated_datasets_dir, 'delimitered_2.csv'))
    dataset = dataset.train_test_split(test_size=0.15)
    print(dataset)

    train_size = len(dataset['train'])

    model_type = args.finetuned_model_path.replace('/', '__') if args.finetuned_model_path else args.model_type.split('/')[-1]
    if len(model_type) > 110:
        model_type = model_type[-110:] + '_etal'

    if args.custom_model_name:
        filename = args.custom_model_name
    else:
        filename = f"{args.custom_token}realnews_100k{'_filtered' if args.filter_dataset_based_on_nli else ''}_size_{train_size}_{model_type}_nepochs_{args.n_epochs}_lr_{args.learning_rate}_compression_{args.compression_rate}"
    print(filename)

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        output_dir=os.path.join('finetuned_bottleself', filename),
        overwrite_output_dir=True,
        save_strategy="epoch",
        warmup_ratio=0.2,  # 0.002
        weight_decay=0.01,
        evaluation_strategy="epoch",
        #save_total_limit=2,
        report_to='none',
        dataloader_drop_last=True  # trying this to avoid division by zero error?
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.finetuned_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.finetuned_model_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_type, cache_dir=cache_dir)
    print_gpu_utilization()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        # prediction_loss_only=True,  # not in Transformers Trainer, but what does it mean?
    )

    torch.cuda.empty_cache()
    gc.collect()

    print_gpu_utilization()
    trainer.train()
    model.save_pretrained(os.path.join('finetuned_bottleself', filename))


if __name__ == "__main__":
    # finetune code from here: https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb

    parser = argparse.ArgumentParser(
        description='Finetune model to summarization dataset (realnews sentences passed through BottleEx).')
    parser.add_argument('--model_type', type=str, default='gpt2')
    parser.add_argument('--finetuned_model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--min_chunk_num', type=int, default=-1)
    parser.add_argument('--max_chunk_num', type=int, default=-1)
    parser.add_argument('--filter_dataset_based_on_nli', action='store_true')
    parser.add_argument('--summaries_dir', type=str, default='summaries_realnews_100k')  # default = BottleEx data
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--train_from_wanli_gpt3_dataset', action='store_true')
    parser.add_argument('--compression_rate', type=float, default=0)  # iff train_from_wanli_gpt3_dataset=True
    parser.add_argument('--chunk_list', type=str, default='')
    parser.add_argument('--custom_token', type=str, default='')
    parser.add_argument('--custom_model_name', type=str, default=None)

    # parser.add_argument('--lm_coef', type=float, default=0.9)

    args = parser.parse_args()
    main(args)
