python src/finetune_referee_distill_with_context_filter.py --model_type gpt2-large --batch_size 2 --chunk_list 2,3,4,100,101,102,103,104,105,106 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --summaries_dir summaries_gpt3_curie_realnews_100k --compression_rate 0.3 --custom_token step_1 && \
( python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3 --n_beams 5 --min_chunk_id 5 --max_chunk_id 5 --device "cuda:0" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3 --n_beams 5 --min_chunk_id 6 --max_chunk_id 6 --device "cuda:1" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3 --n_beams 5 --min_chunk_id 7 --max_chunk_id 7 --device "cuda:2" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3 --n_beams 5 --min_chunk_id 8 --max_chunk_id 8 --device "cuda:3" ) && wait && \
python src/finetune_referee_distill_with_context_filter.py --model_type gpt2-large --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3 --batch_size 2 --min_chunk_num 5 --max_chunk_num 8 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --compression_rate 0.5 --summaries_dir generated-datasets/step_1_nli_maxdiffnsp_6_nepochs_5_compression_0.3/p=None-temp=None-k=None-reppen=1.0-nbeams-5/  --custom_token step_2

( python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5 --n_beams 5 --min_chunk_id 9 --max_chunk_id 9 --device "cuda:0" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5 --n_beams 5 --min_chunk_id 10 --max_chunk_id 10 --device "cuda:1" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5 --n_beams 5 --min_chunk_id 11 --max_chunk_id 11 --device "cuda:2" & \
python src/generate_summaries_referee_distill_with_context_filter.py --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5 --n_beams 5 --min_chunk_id 12 --max_chunk_id 12 --device "cuda:3" ) && wait \
python src/finetune_referee_distill_with_context_filter.py --model_type gpt2-large --finetuned_model_path finetuned-models/referee-distill-with-context-filter/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5 --batch_size 2 --min_chunk_num 9 --max_chunk_num 12 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --compression_rate 0.7 --summaries_dir generated-datasets/step_2_nli_maxdiffnsp_6_nepochs_5_compression_0.5/p=None-temp=None-k=None-reppen=1.0-nbeams-5/  --custom_token step_3
