# MSCLAR note: code extracted from https://huggingface.co/blog/how-to-generate

import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from finetune_referee_distill_with_context_filter import PROMPT_FORMAT, DELIMITER_BETWEEN_SENTENCES, DELIMITER, \
    END_TOKEN_GENERATIONS, max_sentence_length, original_sentence_path

generated_datasets_path = 'generated-datasets/referee-distill-with-context-filter'
#finetuned_models_path = 'finetuned-models/referee-distill'
INVALID_SYMBOL_IN_ORIGINAL_SENTENCE = 'INVALID_SYMBOL_IN_ORIGINAL_SENTENCE_HENCE_SKIPPING_LINE'


def load_unseen_realnews_dataset(original_sentence_path, delim, end_token, chunk_id_to_search):
    assert DELIMITER == delim

    import re
    file_regex = re.compile('^realnews_s1_chunk_[0-9]+.txt$')

    dataset_lines = []
    valid_ids_dict = {}
    invalid_ids_dict = {}
    for filename in os.listdir(original_sentence_path):
        if file_regex.match(filename):
            chunk_id = int(filename[len('realnews_s1_chunk_'):-len('.txt')])
            if chunk_id != chunk_id_to_search:
                continue

            with open(os.path.join(original_sentence_path, filename), 'r') as f, \
                    open(os.path.join(original_sentence_path, f'realnews_s1_chunk_{chunk_id}.txt'), 'r') as g:
                tmp_s1 = [line.strip() for line in f.readlines()]
                tmp_s2 = [line.strip() for line in g.readlines()]

                valid_ids = [i for i, d in enumerate(tmp_s1) if not (delim in d or end_token in d or len(d) < 10)]
                dataset_lines.extend([
                    PROMPT_FORMAT.format(original_sentence=tmp_s1[i],
                                         DELIMITER_BETWEEN_SENTENCES=DELIMITER_BETWEEN_SENTENCES,
                                         next_sentence=tmp_s2[i],
                                         DELIMITER=DELIMITER
                                         ) for i in valid_ids])

                invalid_ids = [i for i, d in enumerate(tmp_s1) if (delim in d or end_token in d or len(d) < 10)]
                invalid_ids_dict[chunk_id] = invalid_ids
                valid_ids_dict[chunk_id] = valid_ids

    return Dataset.from_dict({'text': dataset_lines}), valid_ids_dict, invalid_ids_dict


def extract_summaries_from_generations(model, tokenizer, dataset, params, device, args):
    """
    Note: this should work identically as the non-context filter one
    (I should've unified the code early on, I know!) but there is one minor difference.
    In this case, FULL_SENTENCE_FORMAT has a space before the end token, making
    tokenizer.encode(END_TOKEN_GENERATIONS) two characters, which required some adjustment.

    This artisanal code is to dynamically adapt the number of samples to send in one batch without OOM.
    """

    delimiter_token_ids = tokenizer.encode(DELIMITER)
    end_token_token_id = 37455  # tokenizer.encode(END_TOKEN_GENERATIONS)  # beware, because encoding 'lambda' will turn into two tokens

    generated_texts = []

    # batched text generation: https://github.com/huggingface/transformers/pull/7552#issue-497255933
    lowerbound = 0
    upperbound = 0
    while upperbound < len(dataset):
        torch.cuda.empty_cache()

        # adaptative batch size
        lowerbound = upperbound  # new lowerbound is the first element we didn't include before
        max_char_length = 0
        while upperbound < len(dataset):
            max_char_length = max(max_char_length, len(dataset["text"][upperbound]))
            if lowerbound < upperbound and max_char_length * (upperbound + 1 - lowerbound) >= args.max_constant:
                break
            upperbound += 1

        sentences = dataset["text"][lowerbound:upperbound]
        inputs = tokenizer(
            sentences, max_length=max_sentence_length, truncation=True, padding="longest", return_tensors='pt')

        new_params = params.copy()
        new_params['input_ids'] = inputs['input_ids'].to(device)
        new_params['attention_mask'] = inputs['attention_mask'].to(device)
        new_params['num_return_sequences'] = 1 if not args.select_from_top_n_beams else args.n_beams
        # does not work for lambda because it is two characters :(
        new_params['eos_token_id'] = end_token_token_id  # to stop generating when all batch reached this
        new_params['max_length'] = new_params['input_ids'].shape[1] + max_sentence_length + 2

        with torch.no_grad():
            sample_output = model.generate(**new_params)

        tmp = [{} for _ in range(len(sample_output))]
        for i in range(len(sample_output)):
            tokens = sample_output[i].tolist()

            found_bucket_token_id = False
            for j, t in enumerate(tokens):
                if all(tokens[j + k] == delimiter_token_ids[k] for k in range(len(delimiter_token_ids))):
                    tokens = tokens[j + len(delimiter_token_ids):]
                    found_bucket_token_id = True
                    break

            if not found_bucket_token_id:
                # sentence is too long and delimiter was erased by the tokenizer
                tmp[i]['article'] = ""
                continue

            for j, t in enumerate(tokens):
                if tokens[j] == end_token_token_id:
                    tokens = tokens[:j]
                    break

            full_text_decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            tmp[i]['article'] = full_text_decoded.strip()

        generated_texts.extend([e['article'] for e in tmp])

    return generated_texts


def main(args):
    extended_decoding = f'p=None-temp=None-k=None-reppen={args.repetition_penalty}-nbeams-{args.n_beams}'

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(args.finetuned_model_path)
    model = model.to(device)

    params = {
        'do_sample': True,
        'max_length': max_sentence_length * 2,
        'repetition_penalty': args.repetition_penalty,
        'num_beams': args.n_beams,
        'early_stopping': True
    }

    model_dir = args.finetuned_model_path.split('/')[-1]
    os.makedirs(generated_datasets_path, exist_ok=True)
    os.makedirs(os.path.join(generated_datasets_path, model_dir), exist_ok=True)
    os.makedirs(os.path.join(generated_datasets_path, model_dir, extended_decoding), exist_ok=True)

    for chunk_id in range(args.min_chunk_id, args.max_chunk_id + 1):
        dataset, valid_ids_by_chunk_id, invalid_ids_by_chunk_id = load_unseen_realnews_dataset(
            original_sentence_path, delim=DELIMITER, end_token=END_TOKEN_GENERATIONS,
            chunk_id_to_search=chunk_id)
        dataset = dataset.map(
            lambda batch: tokenizer(batch["text"], max_length=max_sentence_length, truncation=True, padding="longest"),
            batched=True)

        generated_texts = extract_summaries_from_generations(model, tokenizer, dataset, params, device, args)

        valid_ids = valid_ids_by_chunk_id[chunk_id]
        invalid_ids = invalid_ids_by_chunk_id[chunk_id]

        max_id = max(max(invalid_ids), max(valid_ids)) if invalid_ids else max(valid_ids)
        final_generated_texts = [INVALID_SYMBOL_IN_ORIGINAL_SENTENCE for _ in range(max_id + 1)]

        assert len(valid_ids) == len(generated_texts)
        for i, t in zip(valid_ids, generated_texts):
            final_generated_texts[i] = t

        with open(os.path.join(
                generated_datasets_path, model_dir, extended_decoding, f'summaries_chunk_{chunk_id}.txt'), 'w') as g:
            for entry in final_generated_texts:
                g.write(entry)
                g.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create summaries by getting Realnews100k sentences '
                    'and a finetuned GPT2 model, and completing with a summary.')
    parser.add_argument('--finetuned_model_path', type=str)
    parser.add_argument('--model_type', type=str, default='gpt2')

    parser.add_argument('--n_beams', type=int, required=True)
    parser.add_argument('--select_from_top_n_beams', action='store_true')

    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--min_chunk_id', type=int, default=0)
    parser.add_argument('--max_chunk_id', type=int, default=99)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--max_constant', type=int, default=1700)

    args = parser.parse_args()
    main(args)
