# MSCLAR note: code extracted from https://huggingface.co/blog/how-to-generate

import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from finetune_with_control_code import PROMPT_FORMAT, CONTROL_CODE_TOKEN, BUCKET_STRUCTURES, \
    END_TOKEN_GENERATIONS, DELIMITER, max_sentence_length

output_dir = 'generated-datasets'
cache_dir = os.path.join('/gscratch/xlab/msclar/' if torch.cuda.is_available() else './', '.cache')
# max_two_sentence_sizes = 200
INVALID_SYMBOL_IN_ORIGINAL_SENTENCE = 'INVALID_SYMBOL_IN_ORIGINAL_SENTENCE_HENCE_SKIPPING_LINE'


def load_unseen_realnews_dataset(original_sentence_path, delim, end_token, chunk_id_to_search, compression_rate_bucket,
                                 max_sentences=-1):
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
                dataset_lines.extend([PROMPT_FORMAT.format(
                    original_sentence=tmp[i], CONTROL_CODE_TOKEN=CONTROL_CODE_TOKEN,
                    compression_rate_bucket=compression_rate_bucket, DELIMITER=delim) for i in valid_ids])

                invalid_ids = [i for i, d in enumerate(tmp) if (delim in d or end_token in d or len(d) < 10)]
                invalid_ids_dict[chunk_id] = invalid_ids
                valid_ids_dict[chunk_id] = valid_ids

            if len(dataset_lines) > max_sentences > 0:
                break

    return Dataset.from_dict({'text': dataset_lines[:max_sentences] if max_sentences > 0 else dataset_lines}), \
           valid_ids_dict, \
           invalid_ids_dict


def load_any_dataset(original_sentence_filename, delim, end_token, chunk_id_to_search, compression_rate_bucket,
                     max_sentences=-1):
    dataset_lines = []
    valid_ids_dict = {}
    invalid_ids_dict = {}
    with open(os.path.join(original_sentence_filename), 'r') as f:
        tmp = [line.strip() for line in f.readlines()]
        valid_ids = [i for i, d in enumerate(tmp) if not (delim in d or end_token in d or len(d) < 10)]
        dataset_lines.extend([PROMPT_FORMAT.format(
            original_sentence=tmp[i], CONTROL_CODE_TOKEN=CONTROL_CODE_TOKEN,
            compression_rate_bucket=compression_rate_bucket, DELIMITER=delim) for i in valid_ids])

        invalid_ids = [i for i, d in enumerate(tmp) if (delim in d or end_token in d or len(d) < 10)]
        invalid_ids_dict[chunk_id_to_search] = invalid_ids
        valid_ids_dict[chunk_id_to_search] = valid_ids

    return Dataset.from_dict({'text': dataset_lines[:max_sentences] if max_sentences > 0 else dataset_lines}), \
           valid_ids_dict, \
           invalid_ids_dict


def main(args):
    assert args.top_p or args.top_k or args.n_beams

    if args.top_p:
        top_p_string = f'{args.top_p:.2f}' if args.top_p else 'None'
        extended_decoding = f'p={top_p_string}-temp={args.temperature}-k=None-reppen={args.repetition_penalty}-nbeams-{args.n_beams}'
    else:
        extended_decoding = f'p=None-temp={args.temperature}-k={args.top_k}-reppen={args.repetition_penalty}-nbeams-{args.n_beams}'

    if args.filepath_to_summarize:
        extended_decoding = args.filepath_to_summarize
    print(extended_decoding)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if args.model_type.startswith('gpt2') or args.model_type.startswith('EleutherAI') or args.model_type.startswith(
            'stanford-crfm'):
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
        'repetition_penalty': args.repetition_penalty
    }
    if args.temperature:
        params['temperature'] = args.temperature
    if args.top_k:
        params['top_k'] = args.top_k
    if args.top_p:
        params['top_p'] = args.top_p
    if args.n_beams:
        params['num_beams'] = args.n_beams
        params['early_stopping'] = True

    model_dir = args.finetuned_model_path.replace('/', '')
    model_dir = model_dir[len('finetuned_bottleself'):] if model_dir.startswith('finetuned_bottleself') else model_dir
    if not os.path.exists(os.path.join(output_dir, model_dir)):
        os.makedirs(os.path.join(output_dir, model_dir), exist_ok=True)
        os.makedirs(os.path.join(output_dir, model_dir, extended_decoding), exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, model_dir, extended_decoding)):
        os.makedirs(os.path.join(output_dir, model_dir, extended_decoding), exist_ok=True)

    delimiter_token_ids = tokenizer.encode(DELIMITER)
    end_token_token_id = 37455  # tokenizer.encode(END_TOKEN_GENERATIONS)  # beware, because encoding 'lambda' will turn into two tokens

    # https://github.com/huggingface/transformers/pull/7552
    for j, compression_rate_bucket in enumerate(range(len(BUCKET_STRUCTURES[args.bucket_structure_id]))):
        if args.enumerated_bucket_ids and str(j) not in args.enumerated_bucket_ids.split(','):
            continue
        for chunk_id in range(args.min_chunk_id, args.max_chunk_id + 1):
            print('chunk_id', chunk_id)
            if args.repeat_bucket_id_to_fixate_idea:
                compression_rate_bucket_repetitions = " ".join([str(compression_rate_bucket) for _ in range(10)])
            else:
                compression_rate_bucket_repetitions = str(compression_rate_bucket)

            if args.filepath_to_summarize:
                dataset, valid_ids_by_chunk_id, invalid_ids_by_chunk_id = load_any_dataset(
                    args.filepath_to_summarize, delim=DELIMITER, end_token=END_TOKEN_GENERATIONS,
                    chunk_id_to_search=chunk_id, compression_rate_bucket=compression_rate_bucket_repetitions,
                    max_sentences=args.max_sentences
                )
            else:
                dataset, valid_ids_by_chunk_id, invalid_ids_by_chunk_id = load_unseen_realnews_dataset(
                    'outputs/realnews_100k', delim=DELIMITER, end_token=END_TOKEN_GENERATIONS,
                    chunk_id_to_search=chunk_id, compression_rate_bucket=compression_rate_bucket_repetitions,
                    max_sentences=args.max_sentences)

            generated_texts = []

            # batched text generation: https://github.com/huggingface/transformers/pull/7552#issue-497255933
            lowerbound = 0
            upperbound = 0
            while upperbound < len(dataset):
                print(lowerbound)
                torch.cuda.empty_cache()

                # adaptative batch size
                lowerbound = upperbound  # new lowerbound is the first element we didn't include before
                max_char_length = 0
                while upperbound < len(dataset):
                    max_char_length = max(max_char_length, len(dataset["text"][upperbound]))
                    if lowerbound < upperbound and max_char_length * (
                            upperbound + 1 - lowerbound) >= args.max_constant:  # prev 1750
                        break
                    upperbound += 1
                print('max_char_length', max_char_length, upperbound + 1 - lowerbound)

                sentences = dataset["text"][lowerbound:upperbound]
                inputs = tokenizer(sentences, max_length=max_sentence_length, truncation=True, padding="longest",
                                   return_tensors='pt')

                new_params = params.copy()
                new_params['input_ids'] = inputs['input_ids'].to(device)
                new_params['attention_mask'] = inputs['attention_mask'].to(device)
                new_params[
                    'num_return_sequences'] = 1 if not args.select_from_top_n_beams else args.n_beams  # sample_output will be [num_return_sequences * batch_size, 2 * max_sentence_len]
                # does not work for lambda because it is two characters :(
                new_params['eos_token_id'] = end_token_token_id  # to stop generating when all batch reached this
                new_params['max_length'] = new_params['input_ids'].shape[1] + max_sentence_length + 2

                with torch.no_grad():
                    sample_output = model.generate(**new_params)

                # tmp = list(dataset.filter(lambda e, i: lowerbound <= i < upperbound, with_indices=True))
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

                print('lala', tmp)
                generated_texts.extend([e['article'] for e in tmp])

            valid_ids = valid_ids_by_chunk_id[chunk_id]
            invalid_ids = invalid_ids_by_chunk_id[chunk_id]
            if args.max_sentences > 0:
                valid_ids = valid_ids[:args.max_sentences]
            max_id = max(max(invalid_ids), max(valid_ids)) if invalid_ids else max(valid_ids)
            final_generated_texts = [INVALID_SYMBOL_IN_ORIGINAL_SENTENCE for _ in range(max_id + 1)]

            assert len(valid_ids) == len(generated_texts) if args.max_sentences == -1 else True
            print(len(valid_ids), len(invalid_ids), len(generated_texts), max_id)
            for i, t in zip(valid_ids, generated_texts):
                final_generated_texts[i] = t

            tmp = BUCKET_STRUCTURES[args.bucket_structure_id][compression_rate_bucket]

            with open(os.path.join(output_dir, model_dir, extended_decoding,
                                   f'summaries_chunk_{chunk_id}_compression_range_{tmp[0]}-{tmp[1]}.txt'),
                      'w') as outfile:
                for entry in final_generated_texts:
                    outfile.write(entry)
                    outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create summaries by getting Realnews100k sentences '
                    'and a finetuned GPT2 model, and completing with a summary.')
    parser.add_argument('--finetuned_model_path', type=str)
    parser.add_argument('--model_type', type=str, default='gpt2')

    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--n_beams', type=int, default=None)

    parser.add_argument('--max_constant', type=int, default=1700)
    parser.add_argument('--max_sentences', type=int, default=-1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--min_chunk_id', type=int, default=0)
    parser.add_argument('--max_chunk_id', type=int, default=99)
    parser.add_argument('--filepath_to_summarize', type=str, default='')
    parser.add_argument('--bucket_structure_id', type=str, default="original_fourway_bucket")
    parser.add_argument('--repeat_bucket_id_to_fixate_idea', action='store_true')
    parser.add_argument('--enumerated_bucket_ids', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--select_from_top_n_beams', action='store_true')

    args = parser.parse_args()

    if args.filepath_to_summarize:
        args.min_chunk_id = 1
        args.max_chunk_id = args.min_chunk_id
    main(args)
