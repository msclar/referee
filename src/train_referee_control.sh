# extracted from slurm-4775643.out
# TODO: missing training of step_1, generation of texts from step_1 and training of step_2


for i in 2,3,18,19 3,4,19,20 4,5,20,21 5,6,21,22 6,7,22,23 ; do IFS=","; set -- $i; python finetune_with_control_code.py --model_type gpt2-large --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0  --filter_dataset_based_on_nli --filter_based_on_fluency --fluency_ratio_boundary 1.0  --batch_size 2 --n_epochs 2 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --min_samples_per_class_in_rebalanced_dataset 3000 --custom_model_name gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs --dataset_paths outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.0-0.1.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.1-0.2.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.2-0.3.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.3-0.4.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.4-0.5.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.5-0.6.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.6-0.7.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.7-0.8.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.8-0.9.txt,outputs/realnews_100k/realnews_s1_chunk_$3.txt,generated-datasets/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$1_two_epochs_repeatbucketid_fluencyfilter_1.0/p\=None-temp\=None-k\=None-reppen\=1.0-nbeams-5/summaries_chunk_$3_compression_range_0.9-1.0.txt && \
( python generate_summaries_from_control_code.py --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs_repeatbucketid_fluencyfilter_1.0/ --n_beams 5 --min_chunk_id $4 --max_chunk_id $4 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --enumerated_bucket_ids 0,9 --device "cuda:0" & \
python generate_summaries_from_control_code.py --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs_repeatbucketid_fluencyfilter_1.0/ --n_beams 5 --min_chunk_id $4 --max_chunk_id $4 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --enumerated_bucket_ids 1,8 --device "cuda:1" & \
python generate_summaries_from_control_code.py --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs_repeatbucketid_fluencyfilter_1.0/ --n_beams 5 --min_chunk_id $4 --max_chunk_id $4 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --enumerated_bucket_ids 2,7 --device "cuda:2" & \
python generate_summaries_from_control_code.py --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs_repeatbucketid_fluencyfilter_1.0/ --n_beams 5 --min_chunk_id $4 --max_chunk_id $4 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --enumerated_bucket_ids 3,6 --device "cuda:3" & \
python generate_summaries_from_control_code.py --finetuned_model_path finetuned_bottleself/gpt2-large_control_code_tenway_bucket_chunks_2_16_100_106_step$2_two_epochs_repeatbucketid_fluencyfilter_1.0/ --n_beams 5 --min_chunk_id $4 --max_chunk_id $4 --bucket_structure_id tenway_bucket --repeat_bucket_id_to_fixate_idea --enumerated_bucket_ids 4,5 --device "cuda:4" & echo "lalal" ) && wait ; done