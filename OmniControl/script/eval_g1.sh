#!/usr/bin/env bash


input_folder=$1
GT_folder=$2

# retarget to G1 robot and then run the motion tracking
bash ../third-party/ProtoMotions3/scripts/convert_data_to_motionlib.sh $input_folder $cuda


# evaluation after retargeting to G1 robot
python eval/eval_control_error_g1.py --data_root_dir $input_folder --data_gt_dir $GT_folder
