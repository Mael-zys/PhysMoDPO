#!/usr/bin/env bash

cuda=0

# will evaluate the model after SMPL robot simulation

model_path=save/ckpt/DPO_hml3d_original.pt

CUDA_VISIBLE_DEVICES=$cuda python -m eval.eval_humanml --model_path $model_path \
--eval_mode omnicontrol --control_joint 0 --density 100 --mask_type 'cross' --sampler ddim --ddim_eta 0.0 \
--timestep_respacing ddim50 --sim_gpu $cuda --use_smpl --data_part 'flat_ground' --save_figure 