import argparse
import gc
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    AutoModelForCausalLM, AutoModelForSequenceClassification

cache_dir = '/gscratch/xlab/msclar/.cache' if torch.cuda.is_available() else './.cache'
original_sentence_path = 'data/source-sentences/realnews_100k'
finetuned_models_path = 'finetuned-models/referee-distill'

DELIMITER = ' TL;DR: '
END_TOKEN_GENERATIONS = 'Δ'  # tokenizer.eos_token
FULL_SENTENCE_FORMAT = "{original_sentence} {DELIMITER} {summary}{END_TOKEN_GENERATIONS}"
PROMPT_FORMAT = "{original_sentence} {DELIMITER} "


def load_summary_dataset_as_pairs(summary_path, chunk_list=None):
    import re
    summary_file_regex = re.compile('^summaries_chunk_[0-9]+.txt$')

    visited_chunks = set()
    dataset_lines = []
    for summary_filename in os.listdir(summary_path):
        if summary_file_regex.match(summary_filename):
            with open(os.path.join(summary_path, summary_filename), 'r') as f_summary:
                summaries = [line.strip() for line in f_summary.readlines()]

            chunk_id = int(summary_filename[len('summaries_chunk_'):-len('.txt')])
            if chunk_list and chunk_id not in chunk_list:
                continue
            visited_chunks.add(chunk_id)
            original_sentence_filename = f'realnews_s1_chunk_{chunk_id}.txt'
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


def get_chunk_list(args):
    if args.chunk_list:
        chunk_list = [int(t) for t in args.chunk_list.split(',')]
    elif args.min_chunk_num > -1:
        chunk_list = list(range(args.min_chunk_num, args.max_chunk_num + 1))
    else:
        chunk_list = None
    return chunk_list


def filter_dataset_based_on_nli(dataset, max_length=200):
    model_wanli = AutoModelForSequenceClassification.from_pretrained(
        'alisawuffles/roberta-large-wanli', cache_dir=cache_dir)
    tokenizer_wanli = AutoTokenizer.from_pretrained('alisawuffles/roberta-large-wanli', cache_dir=cache_dir)
    dataset = dataset.map(
        lambda x:
        tokenizer_wanli(x["s1"], x["s1_summary"], max_length=max_length, truncation=True, padding="max_length"),
        batched=True
    )

    trainer = Trainer(model=model_wanli, eval_dataset=dataset)
    predictions_tmp = trainer.predict(dataset).predictions
    predicted_label_ids = predictions_tmp.argmax(axis=1).tolist()
    predictions = [model_wanli.config.id2label[p] for p in predicted_label_ids]
    dataset = dataset.add_column("wanli_prediction", predictions)
    dataset = dataset.filter(lambda x: x['wanli_prediction'] == 'entailment')

    del model_wanli
    del tokenizer_wanli
    torch.cuda.empty_cache()
    gc.collect()

    return dataset


def main(args):
    chunk_list = get_chunk_list(args)

    dataset_lines = load_summary_dataset_as_pairs(args.summaries_dir, chunk_list=chunk_list)
    dataset = Dataset.from_dict(
        {'s1': [s for s, s_summ in dataset_lines], 's1_summary': [s_summ for s, s_summ in dataset_lines]}
    )
    dataset = dataset.filter(lambda x: DELIMITER not in x['s1'] and DELIMITER not in x['s1_summary'])
    dataset = dataset.filter(
        lambda x: END_TOKEN_GENERATIONS not in x['s1'] and END_TOKEN_GENERATIONS not in x['s1_summary'])
    dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))

    if args.filter_dataset_based_on_nli:
        dataset = filter_dataset_based_on_nli(dataset, max_length=200)

    dataset = dataset.map(
        lambda x: {"text":
            FULL_SENTENCE_FORMAT.format(
                original_sentence=x['s1'],
                DELIMITER=DELIMITER,
                summary=x['s1_summary'],
                END_TOKEN_GENERATIONS=END_TOKEN_GENERATIONS
            )
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir=cache_dir)
    if args.model_type.startswith('gpt2') or args.model_type.startswith('EleutherAI'):
        tokenizer.pad_token = tokenizer.eos_token  # required for GPT2 but not RoBERTa

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], max_length=200, truncation=True, padding="max_length"),
        batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.15)

    if args.custom_model_name:
        filename = args.custom_model_name
    else:
        filtered_str = '_nli' if args.filter_dataset_based_on_nli else ''
        filename = f"{args.custom_token}{filtered_str}_nepochs_{args.n_epochs}_compression_{args.compression_rate}"
    print(filename)

    os.makedirs(finetuned_models_path, exist_ok=True)
    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        output_dir=os.path.join(finetuned_models_path, filename),
        overwrite_output_dir=True,
        save_strategy="epoch",
        warmup_ratio=0.2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=3,
        report_to='none',
        dataloader_drop_last=True  # trying this to avoid division by zero error?
    )

    if args.finetuned_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.finetuned_model_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_type, cache_dir=cache_dir)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )

    torch.cuda.empty_cache()
    gc.collect()
    trainer.train()
    model.save_pretrained(os.path.join(finetuned_models_path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Finetune model to summarization dataset (realnews sentences passed through BottleEx).')
    parser.add_argument('--model_type', type=str, default='gpt2')
    parser.add_argument('--finetuned_model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=5)

    parser.add_argument('--chunk_list', type=str, default='')
    parser.add_argument('--min_chunk_num', type=int, default=-1)
    parser.add_argument('--max_chunk_num', type=int, default=-1)

    parser.add_argument('--filter_dataset_based_on_nli', action='store_true')
    parser.add_argument('--summaries_dir', type=str, default='summaries_gpt3_curie_realnews_100k')
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--compression_rate', type=float, default=0)
    parser.add_argument('--custom_token', type=str, default='')
    parser.add_argument('--custom_model_name', type=str, default=None)

    args = parser.parse_args()
    main(args)
