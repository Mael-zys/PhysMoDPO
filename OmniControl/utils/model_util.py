# This code is based on https://github.com/GuyTevet/motion-diffusion-model
from model.cmdm import CMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    """Load model weights from checkpoint, supporting both old and new checkpoint formats.
    
    Args:
        model: The model to load weights into
        state_dict: Either:
            - Old format: Direct state_dict
            - New format: Dict with 'model_state_dict' key containing the state_dict
    """
    # Check if this is a new-style checkpoint with metadata
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        print("Detected new-style checkpoint format (with training state)")
        actual_state_dict = state_dict['model_state_dict']
        
        # Print available metadata
        metadata_keys = [k for k in state_dict.keys() if k != 'model_state_dict']
        if metadata_keys:
            print(f"Available metadata in checkpoint: {metadata_keys}")
            if 'step' in state_dict:
                print(f"  Checkpoint was saved at step: {state_dict['step']}")
            if 'current_epoch' in state_dict:
                print(f"  Checkpoint was saved at epoch: {state_dict['current_epoch']}")
    else:
        print("Detected old-style checkpoint format (model weights only)")
        actual_state_dict = state_dict
    
    missing_keys, unexpected_keys = model.load_state_dict(actual_state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)
    # assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = CMDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        if args.use_smpl:
            data_rep = 'smpl'
            njoints = 205
            nfeats = 1
        else:
            data_rep = 'hml_vec'
            njoints = 263 # + 66
            nfeats = 1
    elif args.dataset == 'kit':
        if args.use_smpl:
            raise NotImplementedError("SMPL representation for KIT dataset is not implemented yet.")
        else:
            data_rep = 'hml_vec'
            njoints = 251
            nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = getattr(args, 'timestep_respacing', '')  # can be used for ddim sampling
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if args.use_smpl:
        data_rep = 'smpl'
    else:
        data_rep = 'hml_vec'

    # Get sampler type and ddim_eta from args
    sampler = getattr(args, 'sampler', 'ddpm')
    ddim_eta = getattr(args, 'ddim_eta', 0.0)

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        dataset=args.dataset,
        joints_value_from=getattr(args, 'joints_value_from', 'joints'),
        data_rep=data_rep,
        sampler=sampler,
        ddim_eta=ddim_eta
    )