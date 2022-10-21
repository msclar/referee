python finetune_BottleSelf_with_next_sentence.py --model_type gpt2-large --batch_size 2 --chunk_list 2,3,4,100,101,102,103,104,105,106 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --summaries_dir summaries_gpt3_curie_realnews_100k --compression_rate 0.3 --custom_token _chunks2to4-100to106 && \
( python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3 --n_beams 5 --min_chunk_id 5 --max_chunk_id 5 --device "cuda:0" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3 --n_beams 5 --min_chunk_id 6 --max_chunk_id 6 --device "cuda:1" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3 --n_beams 5 --min_chunk_id 7 --max_chunk_id 7 --device "cuda:2" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3 --n_beams 5 --min_chunk_id 8 --max_chunk_id 8 --device "cuda:3" ) && wait && \
python finetune_BottleSelf_with_next_sentence.py --model_type gpt2-large --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3 --batch_size 2 --min_chunk_num 5 --max_chunk_num 8 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --compression_rate 0.5 --summaries_dir generated-datasets/maxdiffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_15295_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3/p=None-temp=None-k=None-reppen=1.0-nbeams-5/  --custom_token _chunks5to8

( python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 9 --max_chunk_id 9 --device "cuda:0" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 10 --max_chunk_id 10 --device "cuda:1" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 11 --max_chunk_id 11 --device "cuda:2" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 12 --max_chunk_id 12 --device "cuda:3" ) && wait \
python finetune_BottleSelf_with_next_sentence.py --model_type gpt2-large --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --batch_size 2 --min_chunk_num 9 --max_chunk_num 12 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --compression_rate 0.7 --summaries_dir generated-datasets/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5/p=None-temp=None-k=None-reppen=1.0-nbeams-5/  --custom_token _chunks9to12

( python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 9 --max_chunk_id 9 --device "cuda:0" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 10 --max_chunk_id 10 --device "cuda:1" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 11 --max_chunk_id 11 --device "cuda:2" & \
python generate_summaries_with_next_sentence.py --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --n_beams 5 --min_chunk_id 12 --max_chunk_id 12 --device "cuda:3") && wait \
python finetune_BottleSelf_with_next_sentence.py --model_type gpt2-large --finetuned_model_path finetuned_bottleself/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5 --batch_size 2 --min_chunk_num 9 --max_chunk_num 12 --filter_by_next_sentence_probability --filter_dataset_based_on_nli --n_epochs 5 --compression_rate 0.7 --summaries_dir generated-datasets/maxdiffnsp_6_chunks5to8_realnews_100k_filtered_size_4172_ffnsp_6_chunks2to4-100to106_realnews_100k_filtered_size_13951_gpt2-large_nepochs_5_lr_6.25e-05_compression_0.3_etal_nepochs_5_lr_6.25e-05_compression_0.5/p=None-temp=None-k=None-reppen=1.0-nbeams-5/  --custom_token _chunks9to12
