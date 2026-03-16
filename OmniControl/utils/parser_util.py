from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    # assert os.path.exists(args_path), 'Arguments json file was not found!'
    if os.path.exists(args_path):
        with open(args_path, 'r') as fr:
            model_args = json.load(fr)
    else:
        model_args = {}

    for a in args_to_overwrite:
        if a in model_args.keys():
            if a != 'control_joint' and a != 'density' and a != 'mask_type' and a != 'sampler' \
                and a != 'timestep_respacing' and a != 'ddim_eta' and a != 'data_part' and a != 'sim_gpu':
                setattr(args, a, model_args[a])
        else:
            pass
            # print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--sim_gpu", default=0, type=int,
                       help="GPU ID to use for simulation evaluation (default: 0).")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--sampler", default='ddpm', choices=['ddpm', 'ddim'], type=str,
                       help="Sampler type: ddpm for original sampling, ddim for deterministic sampling")
    group.add_argument("--ddim_eta", default=0.0, type=float,
                       help="DDIM eta parameter. 0.0 for deterministic, 1.0 for stochastic (equivalent to DDPM)")
    group.add_argument("--timestep_respacing", default='', type=str,
                       help="Timestep respacing for faster sampling. Use 'ddimN' for N steps or comma-separated numbers")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru', 'hybrid'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--joints_value_from", default='joints', choices=['joints', 'smpl'], type=str,
                       help="Source for joint values in spatial guidance: 'joints' for direct joint extraction, "
                            "'smpl' for SMPL forward kinematics.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--control_joint", default=0, type=int,
                       help="controlling joint")
    group.add_argument("--density", default=100, type=int,
                       help="density")
    parser.add_argument("--omomo", action="store_true")
    parser.add_argument("--use_dpo", action="store_true")
    parser.add_argument("--dpo_data_root", default="", type=str)
    parser.add_argument("--use_smpl", action="store_true", help="Use SMPL representation for motion data")
    group.add_argument("--mask_type", default='original', choices=['original', 'hands', 'cross', 'random'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_part", default='all', choices=['all', 'flat_ground', 'amass'], type=str,
                       help="Which part of the dataset to use: 'all' for all data, 'flat_ground' for flat ground filtered data, 'amass' for amass filtered data.")
    group.add_argument("--dpo_final_metric_weight_list", default=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], type=float, nargs=9,
                       help="Weight list for final metrics in DPO training: ['mean_error', 'skate_ratio', 'dp_mpjpe', 'dp_mpjpe_max', 'power', 'feet_height', 'jerk', 'm2t_score', 'm2m_score'].")
    group.add_argument("--dpo_final_metric_margin_list", default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], type=float, nargs=9,
                       help="Margin list for final metrics in DPO training: ['mean_error', 'skate_ratio', 'dp_mpjpe', 'dp_mpjpe_max', 'power', 'feet_height', 'jerk', 'm2t_score', 'm2m_score'].")
    group.add_argument("--dpo_final_threshold", default=0.03, type=float,
                       help="Threshold for filtering similar motions in DPO training.")
    group.add_argument("--dpo_random_threshold", default=0.02, type=float,
                       help="Random threshold for DPO training pair selection.")
    group.add_argument("--dpo_pair_diff_percentile", default=1.0, type=float,
                       help="Keep top X%% pairs by pair_diff per prompt (0-1). 1 keeps all.")
    group.add_argument("--early_stop_patience", default=2, type=int,
                       help="Number of consecutive evaluations with Control_l2 above threshold before early stopping. Set to 0 to disable early stopping.")
    group.add_argument("--early_stop_threshold", default=0.1, type=float,
                       help="Percentage threshold for early stopping. If Control_l2 is higher than step0 baseline by this percentage for patience evaluations, stop training. E.g., 0.2 means 20% worse.")
    group.add_argument("--dpo_fusescore_selection_list", default=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], type=float, nargs=9,
                       help="Weight list for final metrics in DPO training: ['mean_error', 'skate_ratio', 'dp_mpjpe', 'dp_mpjpe_max', 'power', 'feet_height', 'jerk', 'm2t_score', 'm2m_score'].")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--warmup_steps", default=0, type=int, 
                       help="Number of warmup steps for learning rate. If > 0, learning rate will linearly increase from 0 to lr over warmup_steps.")
    group.add_argument("--grad_clip", default=1.0, type=float,
                       help="Gradient clipping threshold. If > 0, clip gradients by norm. 0 means no clipping. Recommended: 1.0 for stable training.")
    group.add_argument("--num_workers", default=4, type=int,
                       help="Number of data loading workers. Reduce this if you encounter out-of-memory errors. Default: 4")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--sft_scale", default=0.0, type=float,
                       help="If > 0, will perform SFT training with the specified scale.")
    group.add_argument("--beta_dpo", default=5000.0, type=float,
                       help="Perform DPO training with the specified scale.")
    group.add_argument("--dpo_loss_scale", default=1.0, type=float,
                       help="Scale factor for DPO loss, controls the magnitude of DPO loss.")
    group.add_argument("--dpo_loss_type", default='dpo', choices=['dpo', 'ipo'], type=str,
                       help="Loss type for DPO training: 'dpo' (logsigmoid) or 'ipo' (squared loss).")
    group.add_argument("--eval_during_training", action='store_true', default=False,
                       help="If True, will enable evaluation during training.")
    group.add_argument("--visualize_during_training", action='store_true', default=False,
                       help="If True, will enable visualization during training.")

def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--max_tasks_per_gpu", default=1, type=int,
                       help="Maximum number of tasks that can run simultaneously on each GPU. "
                            "Increase this value if you have sufficient GPU memory.")
    group.add_argument("--max_retries", default=3, type=int,
                       help="Maximum number of retries for failed tasks.")
    group.add_argument("--initial_batch_idx", default=0, type=int,
                       help="Initial batch index for parallel runs on multiple servers. "
                            "Set different values on different servers to avoid file naming conflicts.")
    group.add_argument("--initial_dataset_sample_idx", default=0, type=int,
                       help="Initial dataset sample index for parallel runs on multiple servers. "
                            "Set different values on different servers to avoid sample ID conflicts.")
    group.add_argument("--eval_after_simulation", action='store_true', default=False,
                       help="If true, will run evaluation after simulation.")
    group.add_argument("--gpu_ids", default='0', type=str,
                       help="List of GPU IDs to use for parallel generation, separated by commas (e.g., '0,1,2').")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--cond_mode", default='both_text_spatial', type=str,
                       help="generation mode: both_text_spatial, only_text, only_spatial. Other words will be used as text prompt.")
    group.add_argument("--text_prompt", default='predefined', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--skip_ik", action='store_true', default=False,
                       help="If true, will skip IK processing and will output SMPL parameters only.")
    group.add_argument("--skip_visualization", action='store_true', default=False,
                       help="If true, will skip visualization and video generation to save time.")
    group.add_argument("--multi_text", action='store_true', default=False,
                       help="If true, will use multiple text prompts for generation.")
    group.add_argument("--change_guidance", action='store_true', default=False,
                       help="If true, will change the guidance during generation.")
    group.add_argument("--sample_strategy", default='best_worst', choices=['best_worst', 'random'], type=str,
                       help="Sampling strategy to use during generation.")
    group.add_argument("--visualize_mesh", action='store_true', default=False,
                       help="If true, will visualize the mesh.")
    group.add_argument("--maskedmimic_init_mode", default='zero_pose', choices=['zero_pose', 'zero_pose_warmup', 'ground_truth'], type=str,
                       help="MaskedMimic input initialization mode: 'zero_pose' (default), 'zero_pose_warmup', or 'ground_truth'.")

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='omnicontrol', choices=['omnicontrol', 'maskedmimic'], type=str,
                       help="")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--save_figure", action='store_true', default=False,
                       help="If true, will save evaluation figures.")
    group.add_argument("--g1_format", action='store_true', default=False,
                       help="If true, use G1-format data in evaluation.")
    group.add_argument("--eval_after_simulation", dest="eval_after_simulation", action='store_true',
                       help="If true, evaluate using results after simulation.")
    group.add_argument("--no_eval_after_simulation", dest="eval_after_simulation", action='store_false',
                       help="If set, disable evaluation after simulation.")
    group.set_defaults(eval_after_simulation=True)


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_generate_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)

    return args


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)
