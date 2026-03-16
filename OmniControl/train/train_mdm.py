# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    # training
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, use_omomo=args.omomo, 
                              use_dpo=args.use_dpo, mask_type=args.mask_type, dpo_data_root=args.dpo_data_root, 
                              multi_text=args.multi_text, sample_strategy=args.sample_strategy,
                              num_workers=args.num_workers, dpo_final_metric_weight_list=args.dpo_final_metric_weight_list,
                              dpo_final_metric_margin_list=args.dpo_final_metric_margin_list,
                              dpo_final_threshold=args.dpo_final_threshold, dpo_random_threshold=args.dpo_random_threshold,
                              use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part,
                              dpo_pair_diff_percentile=args.dpo_pair_diff_percentile)
    
    eval_data = get_dataset_loader(name=args.dataset, batch_size=32, split='val', hml_mode='eval', 
                                   num_frames=args.num_frames, use_omomo=args.omomo, 
                              use_dpo=False, mask_type=args.mask_type, dpo_data_root=args.dpo_data_root,
                              num_workers=args.num_workers, dpo_final_metric_weight_list=args.dpo_final_metric_weight_list,
                              dpo_final_metric_margin_list=args.dpo_final_metric_margin_list,
                              dpo_final_threshold=args.dpo_final_threshold, dpo_random_threshold=args.dpo_random_threshold,
                              use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part,
                              dpo_pair_diff_percentile=args.dpo_pair_diff_percentile)
    gt_data_for_eval = get_dataset_loader(name=args.dataset, batch_size=32, split='val', hml_mode='gt', 
                                   num_frames=args.num_frames, use_omomo=args.omomo, 
                              use_dpo=False, mask_type=args.mask_type, dpo_data_root=args.dpo_data_root,
                              num_workers=args.num_workers, dpo_final_metric_weight_list=args.dpo_final_metric_weight_list,
                              dpo_final_metric_margin_list=args.dpo_final_metric_margin_list,
                              dpo_final_threshold=args.dpo_final_threshold, dpo_random_threshold=args.dpo_random_threshold,
                              use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part,
                              dpo_pair_diff_percentile=args.dpo_pair_diff_percentile)
    
    if args.use_dpo:
        eval_data_best = None
        eval_data_worst = None
    else:
        eval_data_best = None
        eval_data_worst = None

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")

    TrainLoop(args, train_platform, model, diffusion, data, eval_data, gt_data_for_eval, 
              eval_data_best, eval_data_worst, eval_data_gt=None).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
