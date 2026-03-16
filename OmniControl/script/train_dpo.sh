#!/usr/bin/env bash


# DPO + SFT
cuda_num=0

dpo_data_root='./save/omnicontrol_smpl/inference_rep12_cross'
resume_checkpoint='./save/omnicontrol_smpl/model_last.pt'

CUDA_VISIBLE_DEVICES=$cuda_num python -m train.train_mdm \
--save_dir save/train_dpo \
--train_platform_type TensorboardPlatform --dataset humanml --num_steps 5000 --batch_size 64 \
--resume_checkpoint $resume_checkpoint --lr 1e-6 --save_interval 200 \
--mask_type 'cross' --use_dpo --dpo_data_root $dpo_data_root \
--sft_scale 2.0 --beta_dpo 20 --log_interval 20 --warmup_steps 200 --eval_during_training \
--sample_strategy 'best_worst' --weight_decay 1e-2 --dpo_final_metric_weight_list 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 --visualize_during_training \
--use_smpl --joints_value_from 'joints' --dpo_final_threshold 0.001 --dpo_random_threshold 0.001 \
--dpo_final_metric_margin_list 0.0 0.0 0.0 0.0 "-100.0" "-100.0" "-100.0" 0.0 "-100.0" --sim_gpu $cuda_num --data_part 'flat_ground' \
--dpo_fusescore_selection_list 1.0 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.1 --early_stop_threshold 1
