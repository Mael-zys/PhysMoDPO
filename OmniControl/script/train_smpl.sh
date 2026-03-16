#!/usr/bin/env bash


# use smpl format

# set the cuda index to use
cuda_num=0

CUDA_VISIBLE_DEVICES=$cuda_num python -m train.train_mdm \
--save_dir save/omnicontrol_smpl --lr 1e-4  --save_interval 10000 \
--train_platform_type TensorboardPlatform --dataset humanml --num_steps 800000 --batch_size 64 \
--mask_type 'original' --log_interval 500 --warmup_steps 2000 --eval_during_training \
--weight_decay 1e-2 --visualize_during_training --use_smpl --joints_value_from 'joints' \
--data_part 'amass' --sim_gpu $cuda_num
