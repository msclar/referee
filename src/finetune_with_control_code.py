# coding=utf-8
import os
import gc
import argparse
import torch
import numpy as np
from collections import Counter

from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    AutoModelForCausalLM, AutoModelForSequenceClassification
from pynvml import *

cache_dir = '/gscratch/xlab/msclar/.cache'
filtered_datasets_dir = 'filtered-datasets'

if not torch.cuda.is_available():
    cache_dir = './.cache'

CONTROL_CODE_TOKEN = 'Ξ'
DELIMITER = ' TL;DR: '
END_TOKEN_GENERATIONS = 'Δ'  # tokenizer.eos_token
FULL_SENTENCE_FORMAT = "{original_sentence} {CONTROL_CODE_TOKEN} {compression_rate_bucket} {DELIMITER} {summary} {END_TOKEN_GENERATIONS}"
PROMPT_FORMAT = "{original_sentence} {CONTROL_CODE_TOKEN} {compression_rate_bucket} {DELIMITER}"  # to generate summary
max_sentence_length = 200
max_two_sentences_length = max_sentence_length * 2

# original_sentence Ξ 1 1 1 1 1 1 1 1 TL;DR: ...

BUCKET_STRUCTURES = {
    'original_fourway_bucket': [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)],
    'fiveway_bucket': [(i / 5, (i + 1) / 5) for i in range(5)],
    'tenway_bucket': [(i / 10, (i + 1) / 10) for i in range(10)]
}


def get_average_logprobs(model, encoder, target):
    '''

    Get log probabilities over the target (by token)
    given the source.

    NOTE: target should generally begin with a space (for tokenization)
        e.g. source = "I went to the"
             target = " store to buy food.<|endoftext|>"

          Exception: if source is just <|endoftext|> then do not include
                     a preceding space

    And target should at least end with <|endoftext|>
    '''

    torch.cuda.empty_cache()
    device = model.transformer.wte.weight.device

    prompt = '<|endoftext|>'
    s = encoder.encode(prompt)
    t = encoder.encode(target + prompt)

    input_ids = torch.tensor(s + t).long().view(1, -1)
    labels = torch.tensor([-100] * len(s) + t).long().view(1, -1)

    # get logits from the model
    with torch.no_grad():
        input_ids = input_ids.to(device)
        logits = model(input_ids).logits.cpu()[:, :-1].contiguous()

    # get cross-entropies given the logits
    logit_shape = logits.shape
    logits = logits.view(-1, logit_shape[-1])
    nll_list = torch.nn.functional.cross_entropy(logits, labels[:, 1:].contiguous().view(-1), reduction='none')
    nll_list = nll_list.view(1, -1).squeeze().tolist()[-len(t):]
    return np.mean(nll_list)


def get_bucket(summary_ratio, bucket_structure_id, repeat_bucket_id_to_fixate_idea):
    bucket_id = -1
    for i, (lower, upper) in enumerate(BUCKET_STRUCTURES[bucket_structure_id]):
        if lower <= summary_ratio < upper:
            bucket_id = i
            break

    # bucket_id = -1 means no bucket found. FIX MAY 25 2022
    if repeat_bucket_id_to_fixate_idea and bucket_id != -1:
        bucket_id = " ".join([str(bucket_id) for _ in range(10)])
    return bucket_id


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


#def format_realnews_dataset(dataset_input, tokenizer):
#    dataset = dataset_input.map(
#        lambda batch: tokenizer(batch["text"], max_length=max_sentence_length, truncation=True, padding="max_length"),
#        batched=True)
#    dataset.set_format(type="torch", columns=["input_ids"])
#    return dataset


def load_dataset_as_pair_from_path(original_sentences_path_filename, summary_path_filename):
    dataset_lines = []
    with open(summary_path_filename, 'r') as f_summary:
        summaries = [line.strip() for line in f_summary.readlines()]
    with open(original_sentences_path_filename, 'r') as f_summary:
        original_sentences = [line.strip() for line in f_summary.readlines()]
        original_sentences = original_sentences[:len(summaries)]  # in case of bottleEx crashing

    for s1, s1_summary in zip(original_sentences, summaries):
        if len(s1) < 10 or len(s1_summary) < 1:
            continue
        dataset_lines.append((s1, s1_summary, summary_path_filename))
    return dataset_lines


def load_summary_dataset_as_pairs(summary_path, dataset_paths, min_chunk_num=-1, max_chunk_num=-1):
    # MSCLAR this should be rewritten so that load_dataset_as_pair_from_path is the core function for all
    if dataset_paths:
        files = dataset_paths.split(',')
        dataset_lines = []
        for i in range(0, len(files), 2):
            dataset_lines.extend(load_dataset_as_pair_from_path(files[i], files[i+1]))
        return dataset_lines

    import re
    original_sentence_path = 'outputs/realnews_100k'
    summary_file_regex = re.compile('^summaries_chunk_[0-9]+.txt$')

    dataset_lines = []
    for summary_filename in os.listdir(summary_path):
        if summary_file_regex.match(summary_filename):
            with open(os.path.join(summary_path, summary_filename), 'r') as f_summary:
                summaries = [line.strip() for line in f_summary.readlines()]

            chunk_id = int(summary_filename[len('summaries_chunk_'):-len('.txt')])
            original_sentence_filename = f'realnews_s1_chunk_{chunk_id}.txt'
            if chunk_id > max_chunk_num >= 0 or (min_chunk_num >= 0 and chunk_id < min_chunk_num):
                continue
            with open(os.path.join(original_sentence_path, original_sentence_filename), 'r') as f_summary:
                original_sentences = [line.strip() for line in f_summary.readlines()]
                original_sentences = original_sentences[:len(summaries)]  # in case of bottleEx crashing

            for s1, s1_summary in zip(original_sentences, summaries):
                if len(s1) < 10 or len(s1_summary) <= 3:  # changed min summary length from < 1 to <= 3
                    continue
                dataset_lines.append((s1, s1_summary, str(os.path.join(summary_path, summary_filename))))

    return dataset_lines


def rebalance_dataset(dataset, min_samples):
    most_common_compression_rates = Counter(dataset['compression_rate_bucket']).most_common()
    if most_common_compression_rates[0][1] >= min_samples:
        max_per_class = min([v for k, v in most_common_compression_rates if v >= min_samples])
    else:
        max_per_class = most_common_compression_rates[0][1]

    all_datasets = []
    compression_rate_buckets = list(set(dataset['compression_rate_bucket']))
    for cr in compression_rate_buckets:
        tmp = dataset.filter(lambda x: x['compression_rate_bucket'] == cr).shuffle(seed=42)
        tmp = Dataset.from_dict(tmp[:max_per_class])
        all_datasets.append(tmp)

    return concatenate_datasets(all_datasets)


def rebalance_dataset_respecting_order(dataset, min_samples):
    # Assumption is that a lower step model is preferrable, since each training step adds the possibility of drifting

    most_common_compression_rates = Counter(dataset['compression_rate_bucket']).most_common()
    if most_common_compression_rates[0][1] >= min_samples:
        max_per_class = min([v for k, v in most_common_compression_rates if v >= min_samples])
    else:
        max_per_class = most_common_compression_rates[0][1]

    all_datasets = []
    compression_rate_buckets = list(set(dataset['compression_rate_bucket']))
    for cr in compression_rate_buckets:
        tmp = dataset.filter(lambda x: x['compression_rate_bucket'] == cr)
        tmp = Dataset.from_dict(tmp[:max_per_class])
        all_datasets.append(tmp)

    return concatenate_datasets(all_datasets)


def main(args):
    # assumption: args.dataset_paths will be given in descending order of perceived quality to the user
    dataset_lines = load_summary_dataset_as_pairs(
        args, args.dataset_paths, min_chunk_num=args.min_chunk_num, max_chunk_num=args.max_chunk_num)
    dataset_lines = list(set(dataset_lines))  # remove duplicate entries!
    dataset = Dataset.from_dict(
        {'s1': [s for s, s_summ, _ in dataset_lines],
         's1_summary': [s_summ for s, s_summ, _ in dataset_lines],
         'file': [f for s, s_summ, f in dataset_lines]}
    )

    # apply filters before running NLI model, since it is expensive
    dataset = dataset.filter(lambda x: DELIMITER not in x['s1'] and DELIMITER not in x['s1_summary'])
    dataset = dataset.filter(lambda x: CONTROL_CODE_TOKEN not in x['s1'] and CONTROL_CODE_TOKEN not in x['s1_summary'])
    dataset = dataset.filter(
        lambda x: END_TOKEN_GENERATIONS not in x['s1'] and END_TOKEN_GENERATIONS not in x['s1_summary'])
    dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))
    dataset = dataset.map(lambda x: {
        "compression_rate_bucket": get_bucket(len(x['s1_summary']) / len(x['s1']),
                                              args.bucket_structure_id, args.repeat_bucket_id_to_fixate_idea)})
    dataset = dataset.filter(lambda x: x['compression_rate_bucket'] != -1)
    print(dataset)
    dataset = rebalance_dataset_respecting_order(dataset, min_samples=4 * args.min_samples_per_class_in_rebalanced_dataset)
    print('pre wanli filter')
    for k, v in Counter(dataset['compression_rate_bucket']).most_common():
        print(k, v)
    print()

    for k, v in Counter(dataset['compression_rate_bucket']).most_common():
        for a, v2 in Counter(dataset.filter(lambda x: x['compression_rate_bucket'] == k)['file']).most_common():
            print(k, v, a, v2)
        print()
    print()

    if args.filter_dataset_based_on_nli:
        model_wanli = AutoModelForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli', cache_dir=cache_dir)
        tokenizer_wanli = AutoTokenizer.from_pretrained('alisawuffles/roberta-large-wanli', cache_dir=cache_dir)

        dataset = dataset.map(
            lambda x: tokenizer_wanli(x["s1"], x["s1_summary"], max_length=500, truncation=True, padding="max_length"),
            batched=True)

        trainer = Trainer(model=model_wanli, eval_dataset=dataset)
        predictions_tmp = trainer.predict(dataset).predictions
        if args.min_wanli_prob:
            predictions_probas = torch.tensor(predictions_tmp).softmax(dim=1).squeeze(0)
            predicted_probas_entailment = predictions_probas[:, 1].tolist()
            predictions = ['entailment' if p > args.min_wanli_prob else '' for p in predicted_probas_entailment]
        else:
            predicted_label_ids = predictions_tmp.argmax(axis=1).tolist()
            predictions = [model_wanli.config.id2label[p] for p in predicted_label_ids]

        dataset = dataset.add_column("wanli_prediction", predictions)
        dataset = dataset.filter(lambda x: x['wanli_prediction'] == 'entailment')

        del model_wanli
        del tokenizer_wanli
    else:
        pass  # 0 / 0

    dataset = dataset.map(lambda x: {
        "compression_rate_bucket": get_bucket(
            len(x['s1_summary']) / len(x['s1']), args.bucket_structure_id, args.repeat_bucket_id_to_fixate_idea)})

    dataset = dataset.map(lambda x: {"text": FULL_SENTENCE_FORMAT.format(
        original_sentence=x['s1'], CONTROL_CODE_TOKEN=CONTROL_CODE_TOKEN,
        compression_rate_bucket=x['compression_rate_bucket'],
        DELIMITER=DELIMITER, summary=x['s1_summary'], END_TOKEN_GENERATIONS=END_TOKEN_GENERATIONS)})
    dataset = dataset.filter(lambda x: x['compression_rate_bucket'] != -1)

    if args.filter_based_on_fluency:
        model_gpt2 = AutoModelForCausalLM.from_pretrained('gpt2-large', cache_dir=cache_dir)
        model_gpt2.eval()
        model_gpt2.to('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2-large', cache_dir=cache_dir)
        tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

        print('pre fluency filter')
        print(dataset)
        dataset = dataset.filter(
            lambda x: get_average_logprobs(model_gpt2, tokenizer_gpt2, x['s1_summary']) / get_average_logprobs(model_gpt2, tokenizer_gpt2, x['s1']) < args.fluency_ratio_boundary,
            batched=False)
        print('post fluency filter')
        print(dataset)
        del model_gpt2
        del tokenizer_gpt2
    dataset = rebalance_dataset_respecting_order(dataset, min_samples=args.min_samples_per_class_in_rebalanced_dataset)

    print('final stats')
    for k, v in Counter(dataset['compression_rate_bucket']).most_common():
        print(k, v)
    print()
    for k, v in Counter(dataset['compression_rate_bucket']).most_common():
        for a, v2 in Counter(dataset.filter(lambda x: x['compression_rate_bucket'] == k)['file']).most_common():
            print(k, v, a, v2)
        print()
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir=cache_dir)
    if args.model_type.startswith('gpt2') or args.model_type.startswith('EleutherAI'):
        tokenizer.pad_token = tokenizer.eos_token  # required for GPT2 but not RoBERTa

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], max_length=min(max_two_sentences_length, 512), truncation=True, padding="max_length"),
        batched=True)

    #     dataset = format_realnews_dataset(dataset, tokenizer).shuffle(seed=42)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.train_test_split(test_size=0.15)
    print(dataset)

    train_size = len(dataset['train'])

    model_type = args.finetuned_model_path.replace('/', '__') if args.finetuned_model_path else args.model_type.split('/')[-1]
    if len(model_type) > 110:
        model_type = model_type[-110:] + '_etal'

    if args.custom_model_name:
        filename = args.custom_model_name
    else:
        filename = f"realnews_100k{'_filtered' if args.filter_dataset_based_on_nli else ''}_size_{train_size}_{model_type}_nepochs_{args.n_epochs}_lr_{args.learning_rate}_compression_{args.compression_rate}"
    if args.repeat_bucket_id_to_fixate_idea:
        filename += '_repeatbucketid'
    if args.filter_based_on_fluency:
        filename += f'_fluencyfilter_{args.fluency_ratio_boundary}'
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
    parser.add_argument('--dataset_paths', type=str, default=None, help="Comma separated original + summary dataset paths")
    parser.add_argument('--filter_dataset_based_on_nli', action='store_true')
    parser.add_argument('--summaries_dir', type=str, default='summaries_realnews_100k')  # default = BottleEx data
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--compression_rate', type=float, default=0)
    parser.add_argument('--bucket_structure_id', type=str, default="original_fourway_bucket")
    parser.add_argument('--custom_model_name', type=str, default=None)
    parser.add_argument('--repeat_bucket_id_to_fixate_idea', action='store_true')
    parser.add_argument('--min_samples_per_class_in_rebalanced_dataset', type=int, default=3000)
    parser.add_argument('--min_wanli_prob', type=float, default=None)
    parser.add_argument('--filter_based_on_fluency', action='store_true')
    parser.add_argument('--fluency_ratio_boundary', type=float, default=None)

    args = parser.parse_args()
    assert args.bucket_structure_id in BUCKET_STRUCTURES

    # wandb.init(project="finetune_with_control_code")
    main(args)
