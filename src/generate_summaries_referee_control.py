# MSCLAR note: code extracted from https://huggingface.co/blog/how-to-generate

import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from finetune_referee_control import PROMPT_FORMAT, CONTROL_CODE_TOKEN, BUCKET_STRUCTURE, \
    END_TOKEN_GENERATIONS, DELIMITER, max_sentence_length, repeat_bucket_id, original_sentence_path

from generate_summaries_referee_distill_with_context_filter import extract_summaries_from_generations

generated_datasets_path = 'generated-datasets/referee-control'
cache_dir = os.path.join('/gscratch/xlab/msclar/' if torch.cuda.is_available() else './', '.cache')
INVALID_SYMBOL_IN_ORIGINAL_SENTENCE = 'INVALID_SYMBOL_IN_ORIGINAL_SENTENCE_HENCE_SKIPPING_LINE'


# same as generate_summaries_referee_distill but with other prompt format requirements (fix copy paste!)
def load_unseen_realnews_dataset(original_sentence_path, delim, end_token, chunk_id_to_search, compression_rate_bucket):
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

            with open(os.path.join(original_sentence_path, filename), 'r') as f:
                tmp = [line.strip() for line in f.readlines()]
                valid_ids = [i for i, d in enumerate(tmp) if not (delim in d or end_token in d or len(d) < 10)]
                dataset_lines.extend([
                    PROMPT_FORMAT.format(
                        original_sentence=tmp[i], CONTROL_CODE_TOKEN=CONTROL_CODE_TOKEN,
                        compression_rate_bucket=compression_rate_bucket, DELIMITER=delim) for i in valid_ids
                ])

                invalid_ids = [i for i, d in enumerate(tmp) if (delim in d or end_token in d or len(d) < 10)]
                invalid_ids_dict[chunk_id] = invalid_ids
                valid_ids_dict[chunk_id] = valid_ids

    return Dataset.from_dict({'text': dataset_lines}), valid_ids_dict, invalid_ids_dict


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

    for j, compression_rate_bucket in enumerate(range(len(BUCKET_STRUCTURE))):
        if args.enumerated_bucket_ids and str(j) not in args.enumerated_bucket_ids.split(','):
            continue
        for chunk_id in range(args.min_chunk_id, args.max_chunk_id + 1):
            print('chunk_id', chunk_id)
            compression_rate_bucket_repetitions = repeat_bucket_id(compression_rate_bucket)

            dataset, valid_ids_by_chunk_id, invalid_ids_by_chunk_id = load_unseen_realnews_dataset(
                original_sentence_path, delim=DELIMITER, end_token=END_TOKEN_GENERATIONS,
                chunk_id_to_search=chunk_id, compression_rate_bucket=compression_rate_bucket_repetitions)

            generated_texts = extract_summaries_from_generations(
                model, tokenizer, dataset, params, device, args.max_constant)

            valid_ids = valid_ids_by_chunk_id[chunk_id]
            invalid_ids = invalid_ids_by_chunk_id[chunk_id]
            max_id = max(max(invalid_ids), max(valid_ids)) if invalid_ids else max(valid_ids)
            final_generated_texts = [INVALID_SYMBOL_IN_ORIGINAL_SENTENCE for _ in range(max_id + 1)]

            assert len(valid_ids) == len(generated_texts)
            for i, t in zip(valid_ids, generated_texts):
                final_generated_texts[i] = t

            tmp = BUCKET_STRUCTURE[compression_rate_bucket]
            bucket_filename = f'summaries_chunk_{chunk_id}_compression_range_{tmp[0]}-{tmp[1]}.txt'
            with open(os.path.join(original_sentence_path, model_dir, extended_decoding, bucket_filename), 'w') as g:
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
    parser.add_argument('--max_constant', type=int, default=1700)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--min_chunk_id', type=int, default=0)
    parser.add_argument('--max_chunk_id', type=int, default=99)
    parser.add_argument('--enumerated_bucket_ids', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--select_from_top_n_beams', action='store_true')

    args = parser.parse_args()

    main(args)
