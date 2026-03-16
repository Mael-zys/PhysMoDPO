#!/usr/bin/env bash

generation_gpu='0,1,2,3'

model_path='./save/omnicontrol_smpl/model_last.pt'
output_dir='save/omnicontrol_smpl/inference_rep12_cross'

CUDA_VISIBLE_DEVICES=$generation_gpu python -m sample.generate_smpl_multi_parallel \
--model_path $model_path \
--num_repetitions 12 --text_prompt '' --batch_size 32 --mask_type 'cross' \
--output_dir $output_dir --density 100 --num_samples 20000 --use_smpl --skip_visualization \
--sampler ddim --ddim_eta 0.0 --timestep_respacing ddim50  --data_part 'flat_ground' --max_tasks_per_gpu 2 \
--eval_after_simulation --gpu_ids $generation_gpu


python utils/calculate_dpo_data_stats.py --dpo_root_dir $output_dir