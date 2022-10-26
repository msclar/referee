import argparse
import gc
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    AutoModelForCausalLM

from finetune_referee_distill import get_chunk_list, filter_dataset_based_on_nli

cache_dir = '/gscratch/xlab/msclar/.cache' if torch.cuda.is_available() else './.cache'
original_sentence_path = 'data/source-sentences/realnews_100k'
finetuned_models_path = 'finetuned-models/referee-distill-with-context-filter'

DELIMITER = ' TL;DR: '
END_TOKEN_GENERATIONS = 'Δ'  # tokenizer.eos_token
DELIMITER_BETWEEN_SENTENCES = 'Ξ'
FULL_SENTENCE_FORMAT = "{original_sentence} {DELIMITER_BETWEEN_SENTENCES} {next_sentence} {DELIMITER} {summary} {END_TOKEN_GENERATIONS}"
PROMPT_FORMAT = "{original_sentence} {DELIMITER_BETWEEN_SENTENCES} {next_sentence} {DELIMITER}"  # to generate summary

CE_nored = torch.nn.CrossEntropyLoss(reduction='none')
max_sentence_length = 200
max_two_sentences_length = max_sentence_length * 2


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

            next_sentence_filename = f'realnews_s2_chunk_{chunk_id}.txt'
            with open(os.path.join(original_sentence_path, next_sentence_filename), 'r') as f_summary:
                next_sentences = [line.strip() for line in f_summary.readlines()]
                next_sentences = next_sentences[:len(summaries)]  # in case of bottleEx crashing

            for s1, s2, s1_summary in zip(original_sentences, next_sentences, summaries):
                if len(s1) < 10 or len(s1_summary) < 1:
                    continue
                dataset_lines.append((s1, s2, s1_summary))

    if chunk_list and len(chunk_list) != len(visited_chunks):
        print(f"Some chunks were not collected: visited {list(visited_chunks)}; wanted {chunk_list}")
        0 / 0

    return dataset_lines


def computing_log_probability_one_sample(outputs, input_ids, attention_mask):
    shift_logits = outputs.logits[:-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    assert shift_logits.shape[0] == shift_labels.shape[0]
    if shift_labels.shape[0] == 0:
        return torch.zeros([shift_labels.shape[0]])
    neglog_prob = CE_nored(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    neglog_prob = neglog_prob.view(shift_logits.shape[:-1])
    neglog_prob = (neglog_prob * attention_mask[1:]).sum(axis=0)
    log_p_snext = - neglog_prob
    return log_p_snext


def add_column_log_probability(model_gpt2, tokenizer_gpt2, dataset, device, max_length,
                               column_to_tokenize='concat_s1_summary_s2', new_column_name='log_p_summ_s2',
                               ):
    dataset = dataset.map(
        lambda x: tokenizer_gpt2(x[column_to_tokenize], max_length=max_length, truncation=True, padding="longest"),
        batched=True
    )
    logprob = []
    for row in dataset:
        input_ids = torch.tensor(row['input_ids'], device=device)
        attention_mask = torch.tensor(row['attention_mask'], device=device)
        if len(input_ids) == 0:
            logprob.append(1)
        else:
            outputs = model_gpt2(input_ids=input_ids, attention_mask=attention_mask)
            logprob.append(computing_log_probability_one_sample(outputs, input_ids, attention_mask).item())

    dataset = dataset.add_column(new_column_name, logprob)
    return dataset


def precompute_for_next_sentence_filtering(dataset, model_gpt2, tokenizer_gpt2, device):
    # different " and ' may affect probabilities, so we're standardizing
    for col in ['s1', 's2', 's1_summary']:
        dataset = dataset.map(lambda x: {col: x[col].replace('“', '"').replace('”', '"').replace('’', '\'')})

    # computing log(p(summ, snext))
    dataset = dataset.map(lambda x: {"concat_summ_s2": x['s1_summary'] + " " + x['s2']})
    dataset = add_column_log_probability(
        model_gpt2, tokenizer_gpt2, dataset, device,
        column_to_tokenize='concat_summ_s2', new_column_name='log_p_summ_s2',
        max_length=max_two_sentences_length)

    # computing log(p(s1, snext))
    dataset = dataset.map(lambda x: {"concat_s1_s2": x['s1'] + " " + x['s2']})
    dataset = add_column_log_probability(
        model_gpt2, tokenizer_gpt2, dataset, device,
        column_to_tokenize='concat_s1_s2', new_column_name='log_p_s1_s2',
        max_length=max_two_sentences_length)

    # computing log(p(summ))
    dataset = add_column_log_probability(
        model_gpt2, tokenizer_gpt2, dataset, device,
        column_to_tokenize='s1_summary', new_column_name='log_p_summ',
        max_length=max_sentence_length)

    # computing log(p(s1))
    dataset = add_column_log_probability(
        model_gpt2, tokenizer_gpt2, dataset, device,
        column_to_tokenize='s1', new_column_name='log_p_s1',
        max_length=max_sentence_length)

    # log p(snext|summ) = log p(summ, snext) - log p(summ)
    dataset = dataset.map(
        lambda x: {'next_sentence_prediction_summ_score': x['log_p_summ_s2'] - x['log_p_summ']})
    dataset = dataset.map(lambda x: {'next_sentence_prediction_score': x['log_p_s1_s2'] - x['log_p_s1']})

    # diff_nsp_score := log p(snext|s) - log p(snext|summ) = log {p(snext|s) / p(snext|summ)} <= 6
    # - log {p(snext|summ)/ p(snext|s)} <= 6
    # log {p(snext|summ)/ p(snext|s)} >= -6
    # p(snext|summ)/ p(snext|s) >= e^-6
    dataset = dataset.map(lambda x: {
        'diff_nsp_score': x['next_sentence_prediction_score'] - x['next_sentence_prediction_summ_score']})
    return dataset


def filter_by_next_sentence_probability(dataset):
    # log p(s2|summ) = log p(summ, s2) - log p(summ)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_gpt2 = AutoModelForCausalLM.from_pretrained('gpt2-large', cache_dir=cache_dir)
    model_gpt2.eval()
    model_gpt2.to(device)

    tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2-large', cache_dir=cache_dir)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

    with torch.no_grad():
        dataset = precompute_for_next_sentence_filtering(dataset, model_gpt2, tokenizer_gpt2, device)
        dataset = dataset.filter(lambda x: x['diff_nsp_score'] <= args.max_diff_nsp_score)

    del model_gpt2
    del tokenizer_gpt2
    torch.cuda.empty_cache()
    gc.collect()

    return dataset


def main(args):
    chunk_list = get_chunk_list(args)

    dataset_lines = load_summary_dataset_as_pairs(args.summaries_dir, chunk_list=chunk_list)
    dataset = Dataset.from_dict(
        {'s1': [s for s, snext, s_summ in dataset_lines],
         's2': [snext for s, snext, s_summ in dataset_lines],
         's1_summary': [s_summ for s, snext, s_summ in dataset_lines]}
    )
    dataset = dataset.filter(lambda x: DELIMITER not in x['s1'] and DELIMITER not in x['s1_summary'])
    dataset = dataset.filter(
        lambda x: DELIMITER_BETWEEN_SENTENCES not in x['s1'] and DELIMITER_BETWEEN_SENTENCES not in x['s1_summary'])
    dataset = dataset.filter(
        lambda x: END_TOKEN_GENERATIONS not in x['s1'] and END_TOKEN_GENERATIONS not in x['s1_summary'])
    dataset = dataset.filter(lambda x: (1 - args.compression_rate) * len(x['s1']) > len(x['s1_summary']))

    if args.filter_dataset_based_on_nli:
        dataset = filter_dataset_based_on_nli(dataset, max_length=500)

    if args.filter_by_next_sentence_probability:
        dataset = filter_by_next_sentence_probability(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: {"text":
            FULL_SENTENCE_FORMAT.format(
                original_sentence=x['s1'], next_sentence=x['s2'],
                DELIMITER_BETWEEN_SENTENCES=DELIMITER_BETWEEN_SENTENCES,
                DELIMITER=DELIMITER, summary=x['s1_summary'],
                END_TOKEN_GENERATIONS=END_TOKEN_GENERATIONS
            )
        }
    )
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], max_length=max_two_sentences_length, truncation=True, padding="longest"),
        batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.15)

    filtered_str = '_nli' if args.filter_dataset_based_on_nli else ''
    filename = f"maxdiffnsp_{args.max_diff_nsp_score}{args.custom_token}{filtered_str}_nepochs_{args.n_epochs}_compression_{args.compression_rate}"
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
    # finetune code from here: https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb

    parser = argparse.ArgumentParser(
        description='Finetune model to summarization dataset (realnews sentences passed through BottleEx).')
    parser.add_argument('--model_type', type=str, default='gpt2')
    parser.add_argument('--finetuned_model_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)

    parser.add_argument('--min_chunk_num', type=int, default=-1)
    parser.add_argument('--max_chunk_num', type=int, default=-1)
    parser.add_argument('--chunk_list', type=str, default='')

    parser.add_argument('--filter_dataset_based_on_nli', action='store_true')
    parser.add_argument('--summaries_dir', type=str, default='summaries_gpt3_curie_realnews_100k')
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--compression_rate', type=float, default=0)
    parser.add_argument('--custom_token', type=str, default='')

    parser.add_argument('--filter_by_next_sentence_probability', action='store_true')
    parser.add_argument('--max_diff_nsp_score', type=float, default=6)

    args = parser.parse_args()
    assert args.filter_by_next_sentence_probability
    if not args.filter_dataset_based_on_nli:
        print("WARNING: are you sure you do not want WANLI?!")
    main(args)
