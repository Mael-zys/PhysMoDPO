# This code is based on https://github.com/openai/guided-diffusion
"""
A true multi-GPU parallel version using multiprocessing for actual parallel
execution. Each process loads the model independently to avoid deepcopy
overhead.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
import random
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.plot_script_multi import plot_3d_motion_multi
from data_loaders.tensors import collate
from utils.text_control_example_my import collate_all_my
from utils.text_control_example import collate_all
from os.path import join as pjoin
from visualize.simplify_loc2rot import joints2smpl
from eval.eval_humanml import evaluate_control_sample, evaluate_tmr_sample
from scipy.spatial.transform import Rotation as R
import json
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import queue
import shutil
import gc
from tools.extract_joints import extract_joints, extract_joints_batch
from tools.smplrifke_feats import smplrifkefeats_to_smpldata_batch
import subprocess
import yaml
from tools.fix_fps import interpolate_fps_poses, interpolate_fps_trans
from tools.guofeats import joints_to_guofeats
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy
from utils.runtime_paths import PROTO_MOTIONS_ROOT, resolve_omnicontrol_path
os.environ["PYOPENGL_PLATFORM"] = "egl"

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Ignore multiprocessing resource cleanup warnings, mainly FileNotFoundError
# during semaphore cleanup.
import sys
if sys.version_info < (3, 8):
    # In Python 3.7 and earlier, the multiprocessing module may emit resource
    # cleanup warnings when processes are forcefully terminated.
    # This is a known issue that does not affect execution and can be ignored.
    import logging
    logging.getLogger('multiprocessing').setLevel(logging.ERROR)

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))
    
def axis_angle_yup_to_zup(axis_angle, rotate_angle=-90):
    """Converts axis-angle rotation from Y-up to Z-up."""
    axis_angle = np.asarray(axis_angle)
    r = R.from_rotvec(axis_angle)
    R_yup_to_zup = R.from_euler('x', rotate_angle, degrees=True)
    r_new = R_yup_to_zup * r
    return r_new.as_rotvec()


def zup_to_yup(joint_pos):
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]])
    pose_seq_np_n = np.dot(joint_pos, trans_matrix)
    return pose_seq_np_n


def yup_to_zup(joint_pos):
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    pose_seq_np_n = np.dot(joint_pos, trans_matrix)
    return pose_seq_np_n


def worker_process(gpu_id, rep_i, batch_idx, args_dict, model_path, model_kwargs_serialized, 
                   current_batch_size, n_frames, spatial_norm_path, 
                   dataset_sample_idx, t2m_mean, t2m_std, num_actions, original_motion_serialized, result_queue,
                   smplh_dict, skeleton, video_output_dir, sample_file_template,
                   sample_print_template, output_smpl_ik_path, fps):
    """
    Worker process that handles one repetition on the specified GPU.
    Each process loads the model independently to avoid deepcopy overhead.
    The data is already loaded in the main process, so the worker uses the
    provided arguments directly to avoid reinitializing the DataLoader.
    """
    try:
        # Set the GPU.
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        # Synchronize the device used inside dist_util so workers do not all
        # fall back to cuda:0 and block each other.
        dist_util.setup_dist(gpu_id)
        
        # Set an independent random seed.
        seed = batch_idx * args_dict['num_repetitions'] + rep_i + args_dict['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        print(f'[GPU {gpu_id}] Processing rep {rep_i} with seed {seed}')
        
        # Reload the model in this process. This is much faster than deepcopy.
        from utils.parser_util import generate_args
        from utils.model_util import create_model_and_diffusion, load_model_wo_clip
        
        # Rebuild the args object.
        class Args:
            pass
        args = Args()
        for k, v in args_dict.items():
            setattr(args, k, v)
        
        # Create a lightweight mock data object to avoid reloading the
        # DataLoader. It only provides the minimal information required for
        # model initialization.
        class MockDataset:
            def __init__(self, num_actions):
                self.num_actions = num_actions
        
        class MockT2MDataset:
            def __init__(self, mean, std):
                self.mean = torch.from_numpy(mean).float() if mean is not None else None
                self.std = torch.from_numpy(std).float() if std is not None else None
            
            def inv_transform(self, data):
                # Denormalize with: data * std + mean.
                if self.mean is not None and self.std is not None:
                    return data * self.std.to(data.device) + self.mean.to(data.device)
                return data
        
        class MockDataWrapper:
            def __init__(self):
                self.t2m_dataset = MockT2MDataset(t2m_mean, t2m_std)
        
        class MockData:
            def __init__(self, num_actions):
                self.dataset = MockDataWrapper()
                # Used by create_model_and_diffusion.
                mock_dataset = MockDataset(num_actions)
                self.dataset.num_actions = num_actions
        
        data = MockData(num_actions)
        
        tmr_forward = load_tmr_model_easy(device)

        # Create and load the model.
        model, diffusion = create_model_and_diffusion(args, data)
        state_dict = torch.load(model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)
        
        if args.guidance_param != 1:
            from model.cfg_sampler import ClassifierFreeSampleModel
            model = ClassifierFreeSampleModel(model)
        
        model.to(device)
        model.eval()
        
        # Rebuild model_kwargs.
        model_kwargs = {'y': {}}
        for k, v in model_kwargs_serialized.items():
            if isinstance(v, np.ndarray):
                model_kwargs['y'][k] = torch.from_numpy(v).to(device)
            elif isinstance(v, list):
                model_kwargs['y'][k] = v.copy()
            else:
                model_kwargs['y'][k] = v
        
        # Handle multi_text
        if args.multi_text and 'text' in model_kwargs['y']:
            text_batch = model_kwargs['y']['text']
            text_batch_split = [text_sample.split(' ||| ') for text_sample in text_batch]
            current_texts = [text_options[rep_i % len(text_options)] for text_options in text_batch_split]
            model_kwargs['y']['text'] = current_texts
        
        # Add CFG scale
        if args.guidance_param != 1:
            scale_value = args.guidance_param
            if args.change_guidance:
                scale_value += (rep_i - args.num_repetitions // 2 + 1) * 0.5
            model_kwargs['y']['scale'] = torch.ones(current_batch_size, device=device) * scale_value
        
        # Generate noise
        noise = torch.randn((current_batch_size, model.njoints, model.nfeats, n_frames), device=device)
        
        # Sample
        sample_fn = diffusion.p_sample_loop
        with torch.no_grad():
            sample = sample_fn(
                model,
                (current_batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
        
        sample = sample[:, :263]
        output_vector = sample.clone().permute(0, 2, 3, 1).cpu().numpy()[:,0]
        
        vertices_batch = None
        poses_batch = None
        trans_batch = None

        res_metrics = {}

        if model.data_rep == 'smpl':
            n_joints = 24

            sample = sample.permute(0, 2, 3, 1).contiguous()
            sample = sample.squeeze(1)
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu()).float()

            smpldata = smplrifkefeats_to_smpldata_batch(sample.to(dist_util.dev()).clone(), first_angle=np.pi)

            if args_dict['eval_after_simulation']:
                os.makedirs(os.path.join(args_dict['output_eval_sim'], f'{batch_idx}_{rep_i}', 'amass_format', 'ik'), exist_ok=True)
                for bs_i in range(current_batch_size):
                    length = model_kwargs['y']['lengths'][bs_i]
                    save_path = os.path.join(args_dict['output_eval_sim'], f'{batch_idx}_{rep_i}', 'amass_format', 'ik', f'ik_{bs_i}.npz')
                    np.savez(save_path, poses=smpldata['poses'][bs_i].cpu().numpy()[:length], 
                        trans=smpldata['trans'][bs_i].cpu().numpy()[:length],
                        betas=np.zeros(10), num_betas=10, gender='neutral',
                            mocap_frame_rate=20)

                absolute_output_path = resolve_omnicontrol_path(
                    args_dict['output_eval_sim'], f'{batch_idx}_{rep_i}'
                )

                # use specified GPU for simulation evaluation
                sim_gpu_id = args_dict['gpu_ids'].split(',')[gpu_id % len(args_dict['gpu_ids'].split(','))]
                subprocess.call(
                    ["bash", str(PROTO_MOTIONS_ROOT / "run_deepmimic.sh"), absolute_output_path, sim_gpu_id],
                    cwd=str(PROTO_MOTIONS_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                simulation_motion_folder = os.path.join(absolute_output_path, 'deepmimic_output')
                mask_mimic_id_list = yaml.load(open(os.path.join(absolute_output_path, 'amass_format', 'data_list.yaml'), "r"), Loader=yaml.FullLoader)["motions"]
                mask_mimic_name2inx = {}
                for mask_mimic_file in mask_mimic_id_list:
                    mask_mimic_file_name = mask_mimic_file['file'].split('/')[-1].replace('.npy', '')
                    mask_mimic_idx = mask_mimic_file['idx']
                    mask_mimic_name2inx[mask_mimic_file_name] = mask_mimic_idx

                dp_mpjpe = json.load(open(os.path.join(simulation_motion_folder, 'all_motions_with_gt_err_0.json'), 'r'))
                dp_mpjpe_max = json.load(open(os.path.join(simulation_motion_folder, 'all_motions_with_gt_err_max_0.json'), 'r'))
                power = json.load(open(os.path.join(simulation_motion_folder, 'all_motions_with_power_0.json'), 'r'))

                model_kwargs['y']['lengths'] -= 2
                res_metrics['dp_mpjpe'] = {}
                res_metrics['dp_mpjpe_max'] = {}
                res_metrics['power'] = {}
                for bs_i in range(current_batch_size):
                    length = model_kwargs['y']['lengths'][bs_i]
                    maskmimic_id = mask_mimic_name2inx[f'ik_{bs_i}']
                    maskmimic_file = os.path.join(simulation_motion_folder, f'trajectory_pose_aa_{maskmimic_id}_0.npz')
                    sim_smpldata = np.load(maskmimic_file)
                    smpldata['poses'][bs_i,:length,:] = interpolate_fps_poses(torch.from_numpy(sim_smpldata['pose'][0]).float(), 30.0, 20.0)[:length,:66].to(sample.device)
                    smpldata['trans'][bs_i,:length,:] = interpolate_fps_trans(torch.from_numpy(sim_smpldata['trans'][0]).float(), 30.0, 20.0)[:length].to(sample.device)

                    sample_temp_name = f'sample_sample_{bs_i+dataset_sample_idx}_rep_{rep_i}'
                    res_metrics['dp_mpjpe'][sample_temp_name] = dp_mpjpe[f'ik_{bs_i}']
                    res_metrics['dp_mpjpe_max'][sample_temp_name] = dp_mpjpe_max[f'ik_{bs_i}']
                    res_metrics['power'][sample_temp_name] = power[f'ik_{bs_i}']

                joints = extract_joints_batch(smpldata.copy(),
                                            "smpldata",
                                            fps=20,
                                            value_from="smpl",
                                            smpl_layer=smplh_dict['neutral'],keep_torch=True)
            else:
                joints = extract_joints_batch(sample.to(dist_util.dev()),
                                "smplrifke",
                                fps=20,
                                value_from="smpl",
                                smpl_layer=smplh_dict['neutral'],keep_torch=True)
            sample = joints['joints'].cpu()  # (B, T, 24, 3)
            sample = sample.permute(0, 2, 3, 1)  # (B, 24, 3, T)
            vertices_batch = joints['vertices'].cpu().numpy()
            poses_batch = smpldata['poses'].cpu().numpy()
            trans_batch = smpldata['trans'].cpu().numpy()

        else:
            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(current_batch_size, n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)
        
        if model.data_rep == 'smpl':
            motions_temp = sample.clone().cpu().float()
            # numpy array from (B, 22, 3, T) to (B, T, 22, 3)
            motions_temp = motions_temp.permute(0, 3, 1, 2)
            x, y, z = T(motions_temp)
            motions_temp = T(np.stack((x, z, -y), axis=0)) # change to y-up
        else:
            motions_temp = sample.clone().cpu().float()
            # numpy array from (B, 22, 3, T) to (B, T, 22, 3)
            motions_temp = motions_temp.permute(0, 3, 1, 2)
            
        # joints to guo feature for TMR evaluation
        motion_pred_list = torch.from_numpy(np.array([joints_to_guofeats(motions_temp[bs_i]) for bs_i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy())]))
        motion_gt_list = None
        if original_motion_serialized is not None:
            if isinstance(original_motion_serialized, np.ndarray):
                gt_motion = torch.from_numpy(original_motion_serialized)
            elif torch.is_tensor(original_motion_serialized):
                gt_motion = original_motion_serialized
            else:
                gt_motion = torch.from_numpy(np.asarray(original_motion_serialized))

            gt_motion = gt_motion.float()
            if model.data_rep == 'smpl':
                gt_motion = gt_motion.permute(0, 2, 3, 1).contiguous().squeeze(1)
                gt_motion = data.dataset.t2m_dataset.inv_transform(gt_motion.cpu()).float()
                gt_joints = extract_joints_batch(
                    gt_motion.to(dist_util.dev()),
                    "smplrifke",
                    fps=20,
                    value_from="smpl",
                    smpl_layer=smplh_dict['neutral'],
                    keep_torch=True,
                )['joints']
                gt_sample = gt_joints.cpu().permute(0, 2, 3, 1)
            else:
                if model.data_rep == 'hml_vec':
                    n_joints = 22 if gt_motion.shape[1] == 263 else 21
                    gt_sample = data.dataset.t2m_dataset.inv_transform(gt_motion.cpu().permute(0, 2, 3, 1)).float()
                    gt_sample = recover_from_ric(gt_sample, n_joints)
                    gt_sample = gt_sample.view(-1, *gt_sample.shape[2:]).permute(0, 2, 3, 1)
                else:
                    gt_sample = gt_motion.to(device)
                rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
                rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(current_batch_size, n_frames).bool()
                gt_sample = model.rot2xyz(x=gt_sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

            motions_temp_gt = gt_sample.clone().cpu().float()
            motions_temp_gt = motions_temp_gt.permute(0, 3, 1, 2)
            if model.data_rep == 'smpl':
                x, y, z = T(motions_temp_gt)
                motions_temp_gt = T(np.stack((x, z, -y), axis=0))
            motion_gt_list = torch.from_numpy(np.array([joints_to_guofeats(motions_temp_gt[bs_i]) for bs_i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy())]))
        
        res_metrics_tmr = evaluate_tmr_sample(
            tmr_forward,
            motion_pred_list,
            model_kwargs['y']['text'],
            m_lens=model_kwargs['y']['lengths'].cpu().numpy()-1,
            global_sample_idx=dataset_sample_idx,
            rep_i=rep_i,
            gt_motions=motion_gt_list
        )
        
        res_metrics.update(res_metrics_tmr)

        # Process hints if present and evaluate control metrics
        hint_np = None
        hint_for_vis_np = None
        
        if not args.unconstrained and 'hint' in model_kwargs['y']:
            hint = model_kwargs['y']['hint']
            raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))).to(device)
            raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))).to(device)
            mask = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(-1) != 0
            hint = hint * raw_std + raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask.unsqueeze(-1)
            
            # Evaluate control sample to get metrics
            # Create temporary res dict for this repetition
            res_temp = {}
            res_temp['skate_ratio'] = {}
            res_temp['mean_error'] = {}
            res_temp['feet_height'] = {}
            res_temp['jerk'] = {}
            traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
            for key in traj_err_key:
                res_temp[key] = {}
            
            res_updated = evaluate_control_sample(
                sample.clone().cpu(),
                hint.clone().cpu(),
                mask.unsqueeze(-1).clone().cpu(),
                res_temp,
                rep_i,
                dataset_sample_idx,
                m_lens=model_kwargs['y']['lengths'].cpu().numpy(),
                use_smpl=args.use_smpl
            )
            # Aggregate metrics
            res_metrics.update(res_updated)
            
            hint_np = hint.view(hint.shape[0], hint.shape[1], -1).clone().data.cpu().numpy()
            hint_for_vis_np = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).clone().data.cpu().numpy()
        
        sample_np = sample.cpu().numpy()

        # Get the actual text used after resolving multi_text.
        actual_text = None
        if 'text' in model_kwargs['y']:
            actual_text = model_kwargs['y']['text']  # multi_text was already resolved earlier.
        elif 'action_text' in model_kwargs['y']:
            actual_text = model_kwargs['y']['action_text']

        lengths_np = model_kwargs['y']['lengths'].detach().cpu().numpy() \
            if torch.is_tensor(model_kwargs['y']['lengths']) else np.asarray(model_kwargs['y']['lengths'])
        objects_list = model_kwargs['y'].get('object', None)

        # Perform per-sample visualization inside worker to avoid post-processing bottleneck
        if not args.skip_visualization:
            text_for_vis = None
            if args.unconstrained:
                text_for_vis = ['unconstrained'] * current_batch_size
            else:
                if actual_text is not None:
                    text_for_vis = actual_text
                else:
                    text_for_vis = [''] * current_batch_size

            mesh_renderer = None
            if args.visualize_mesh and (args.use_smpl or not args.skip_ik):
                from renderer.humor import HumorRenderer
                mesh_renderer = HumorRenderer(fps=fps)

            try:
                for sample_i in range(current_batch_size):
                    current_dataset_idx = dataset_sample_idx + sample_i
                    caption = text_for_vis[sample_i] if text_for_vis else ''
                    length = int(lengths_np[sample_i])
                    motion = sample_np[sample_i].transpose(2, 0, 1)[:length]
                    hint = hint_for_vis_np[sample_i][:length] if hint_for_vis_np is not None else None
                    object_item = None
                    if isinstance(objects_list, (list, tuple)):
                        object_item = objects_list[sample_i]

                    save_file = sample_file_template.format(current_dataset_idx, rep_i)
                    animation_save_path = os.path.join(video_output_dir, save_file)

                    hint_vis_mesh = None
                    vertices_vis_mesh = None

                    if not args.use_smpl and not args.skip_ik:
                        ik_path = os.path.join(output_smpl_ik_path, f'ik_{current_dataset_idx}_{rep_i}.npz')

                        j2s = joints2smpl(num_frames=length, device_id=gpu_id, cuda=torch.cuda.is_available())
                        opt_dict = j2s.joint2smpl_amass(motion.copy())
                        poses = opt_dict['poses']
                        poses[:,:3] = axis_angle_yup_to_zup(poses[:,:3], 90)
                        np.savez(ik_path, poses=poses, trans=yup_to_zup(opt_dict['trans']),
                                    betas=opt_dict['betas'], num_betas=10, gender='neutral',
                                    mocap_frame_rate=fps, text=caption)

                        if hint is not None:
                            hint_vis_mesh = hint[:length]
                            x, mz, my = T(hint_vis_mesh)
                            hint_vis_mesh = T(np.stack((x, -my, mz), axis=0))

                        offset = motion[0, 0] - opt_dict['new_opt_joints'][0, 0]
                        opt_dict['new_opt_vertices'] += offset

                        x, mz, my = T(opt_dict['new_opt_vertices'])
                        vertices_vis_mesh = T(np.stack((x, -my, mz), axis=0))

                    elif args.use_smpl:
                        hint_vis_mesh = hint[:length].copy() if hint is not None else None
                        if vertices_batch is not None:
                            vertices_vis_mesh = vertices_batch[sample_i][:length]

                        x, y, z = T(motion.copy())
                        motion = T(np.stack((x, z, -y), axis=0))
                        if hint is not None:
                            x, y, z = T(hint.copy())
                            hint = T(np.stack((x, z, -y), axis=0))

                        if poses_batch is not None and trans_batch is not None:
                            ik_path = os.path.join(output_smpl_ik_path, f'ik_{current_dataset_idx}_{rep_i}.npz')
                            np.savez(ik_path, poses=poses_batch[sample_i][:length], 
                                        trans=trans_batch[sample_i][:length],
                                        betas=np.zeros(10), num_betas=10, gender='neutral',
                                        mocap_frame_rate=fps, text=caption)

                    if args.visualize_mesh and (args.use_smpl or not args.skip_ik) and vertices_vis_mesh is not None:
                        if hint_vis_mesh is not None:
                            points_seq = []
                            for frame_idx in range(hint_vis_mesh.shape[0]):
                                frame_points = hint_vis_mesh[frame_idx]
                                mask = np.all(frame_points != 0, axis=1)
                                valid_points = frame_points[mask]
                                points_seq.append(valid_points)
                        else:
                            points_seq = None

                        mesh_output_path = animation_save_path.replace('.mp4', '_mesh.npz')
                        np.savez(mesh_output_path, vertices=vertices_vis_mesh, points_seq=points_seq)

                        if args.use_smpl:
                            cam_rot_matrix = R.from_euler('xz', [90, 180], degrees=True).as_matrix()
                            cam_offset=[0.0, 2.2, 0.9]
                        else:
                            cam_rot_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
                            cam_offset=[0.0, -2.2, 0.9]

                        if mesh_renderer is not None:
                            mesh_renderer(vertices=vertices_vis_mesh, 
                                            output=animation_save_path.replace('.mp4', '_mesh.mp4'),
                                            points_seq=points_seq, cam_rot=cam_rot_matrix, cam_offset=cam_offset,
                                            point_rad=0.10, put_ground=False)

                            mesh_tmp_dir = animation_save_path.replace('.mp4', '_mesh')
                            if os.path.exists(mesh_tmp_dir):
                                shutil.rmtree(mesh_tmp_dir, ignore_errors=True)

                    try:
                        plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset,
                                    title=caption if args.cond_mode != 'only_spatial' else None,
                                    fps=fps, hint=hint if args.cond_mode != 'only_text' else None, 
                                    objects=object_item, elev=-60 if args.use_smpl else 120, 
                                    azim=90 if args.use_smpl else -90)
                    except:
                        try:
                            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset,
                                        title=caption if args.cond_mode != 'only_spatial' else None,
                                        fps=fps, hint=hint if args.cond_mode != 'only_text' else None, 
                                        objects=object_item, elev=-60 if args.use_smpl else 120, 
                                        azim=90 if args.use_smpl else -90)
                        except Exception as plot_err:
                            print(f"[GPU {gpu_id}] Visualization failed for sample {current_dataset_idx}, rep {rep_i}: {plot_err}")
                            continue

                    print(sample_print_template.format(caption, current_dataset_idx, rep_i, save_file))
            finally:
                if mesh_renderer is not None:
                    del mesh_renderer
                import matplotlib.pyplot as plt
                plt.close('all')
        
        # Return the result. Including rep_i is important for sorting.
        result = {
            'rep_i': rep_i,
            'sample': sample_np,
            'output_vector': output_vector,
            'hint_np': hint_np,
            'hint_for_vis_np': hint_for_vis_np,
            'res_metrics': res_metrics,  # Include metrics
            'actual_text': actual_text,  # Actual text used, with multi_text resolved.
            'all_poses': poses_batch,
            'all_trans': trans_batch
        }
        
        # Use a blocking put to ensure the result reaches the queue.
        result_queue.put(result, block=True, timeout=300)  # 5-minute timeout.
        print(f'[GPU {gpu_id}] Rep {rep_i} completed and result queued')
        
    except Exception as e:
        print(f'[GPU {gpu_id}] ERROR in rep {rep_i}: {str(e)}')
        import traceback
        traceback.print_exc()
        try:
            # Error results also need to be put into the queue.
            result_queue.put({'rep_i': rep_i, 'error': str(e)}, block=True, timeout=60)
        except Exception as put_error:
            print(f'[GPU {gpu_id}] CRITICAL: Failed to put error result to queue: {put_error}')
    finally:
        # Thoroughly clean up all resources.
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Release GPU resources.
        try:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except:
            pass
        
        # Force garbage collection.
        try:
            import gc
            gc.collect()
        except:
            pass
        
        print(f'[GPU {gpu_id}] Rep {rep_i} worker process exiting cleanly')


def main():
    args = generate_args()
    fixseed(args.seed)
    
    # Configure retry behavior.
    max_retries = getattr(args, 'max_retries', 3)  # Default: up to 3 retries.
    # Configure the maximum number of concurrent tasks per GPU.
    max_tasks_per_gpu = getattr(args, 'max_tasks_per_gpu', 1)  # Default: 1 task per GPU.
    
    out_path = args.output_dir
    
    mesh_renderer = None
    if args.visualize_mesh:
        from renderer.humor import HumorRenderer
        mesh_renderer = HumorRenderer()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    n_frames = 196
    is_using_data = not any([args.text_prompt])
    
    if args.use_smpl:
        from tools.smpl_layer import SMPLH
        smplh_dict = {
                gender: SMPLH(
                    path="body_models/smplh",
                    jointstype="both",
                    input_pose_rep="axisangle",
                    gender=gender,
                )
                for gender in ["neutral", "male", "female"]
            }
        
        for smpl_layer in smplh_dict.values():
            smpl_layer = smpl_layer.eval()
            # Freeze SMPL layer parameters but allow gradient flow
            for param in smpl_layer.parameters():
                param.requires_grad = False
    else:
        smplh_dict = {}

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'generate_{}_{}_seed{}_randomtype_{}_density_{}_cond_mode_{}'.format(
                                    name, niter, args.seed, args.mask_type, args.density, args.cond_mode))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')

    hints = None
    objects = None
    
    # Load text prompts if specified
    if not args.use_smpl and args.text_prompt != '':
        if 'mypredefined' in args.text_prompt:
            n_frames, texts, hints, objects = collate_all_my(n_frames, args.dataset)
            args.num_samples = len(texts)
            if args.cond_mode == 'only_spatial':
                texts = ['' for i in texts]
            elif args.cond_mode == 'only_text':
                hints = None
        elif args.text_prompt == 'predefined' or args.text_prompt == 'predefined_wo_text':
            texts, hints = collate_all(n_frames, args.dataset, args)
            args.num_samples = len(texts)
            if args.cond_mode == 'only_spatial':
                texts = ['' for i in texts]
            elif args.cond_mode == 'only_text':
                hints = None
        else:
            texts = [args.text_prompt]
            args.num_samples = 1

    elif args.use_smpl and args.text_prompt != '':
        raise NotImplementedError('to be implemented')
    
    if not is_using_data:
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
        args.batch_size = args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    # Detect the number of available GPUs.
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    if num_gpus == 0:
        print("No GPU detected, using CPU (will be slow)")
        num_gpus = 1

    omnicontrol_output_dir = os.path.join(out_path, 'omnicontrol_output')
    video_output_dir = os.path.join(out_path, 'video_output')
    output_smpl_path = os.path.join(out_path, 'amass_format')
    output_smpl_ik_path = os.path.join(output_smpl_path, 'ik')
    
    if args.eval_after_simulation:
        output_eval_sim = os.path.join(out_path, 'eval_after_simulation')
        os.makedirs(output_eval_sim, exist_ok=True)
        args.output_eval_sim = output_eval_sim
                
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(omnicontrol_output_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(output_smpl_path, exist_ok=True)
    os.makedirs(output_smpl_ik_path, exist_ok=True)
    
    res = {}
    res['skate_ratio'] = {}
    res['mean_error'] = {}
    res['feet_height'] = {}
    res['jerk'] = {}
    res['dp_mpjpe'] = {}
    res['dp_mpjpe_max'] = {}
    res['power'] = {}
    res['m2t_score'] = {}
    res['m2m_score'] = {}
    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    for key in traj_err_key:
        res[key] = {}
    
    # Try loading an existing metrics file, if present, to avoid overwriting
    # metrics from previous batches.
    existing_metrics_path = os.path.join(out_path, "omnicontrol_sorted.json")
    if os.path.exists(existing_metrics_path):
        try:
            with open(existing_metrics_path, 'r') as f:
                existing_res = json.load(f)
            # Merge existing metrics.
            for metric_key in existing_res:
                if metric_key in res:
                    res[metric_key].update(existing_res[metric_key])
            print(f"Loaded existing metrics from {existing_metrics_path}")
            print(f"  Existing metrics count: skate_ratio={len(existing_res.get('skate_ratio', {}))}, mean_error={len(existing_res.get('mean_error', {}))}")
        except Exception as e:
            print(f"Warning: Failed to load existing metrics: {e}")

    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    
    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    num_samples_in_out_file = 7
    all_sample_files = []
    
    # Process dataset batch by batch
    if is_using_data:
        data_iterator = iter(data)
        batches_per_epoch = len(data)
        current_epoch = 0  # Track the current epoch to refresh the DataLoader seed.
        print(f"Dataset has {batches_per_epoch} batches per epoch")
        # In data mode, num_batches is only for display, so use an estimate.
        # The actual number of batches is determined by num_samples and batch_size.
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size if args.num_samples > 0 else batches_per_epoch
    else:
        num_batches = 1
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        if hints is not None:
            collate_args = [dict(arg, hint=hint) for arg, hint in zip(collate_args, hints)]
        if objects is not None:
            collate_args = [dict(arg, object=object) for arg, object in zip(collate_args, objects)]
        _, model_kwargs = collate(collate_args)

    dataset_sample_idx = args.initial_dataset_sample_idx
    batch_idx = args.initial_batch_idx  # Global batch counter across epochs.
    if args.dataset == 'humanml':
        if args.use_smpl:
            spatial_norm_path = './dataset/HumanML3D_amass'
        else:
            spatial_norm_path = './dataset/humanml_spatial_norm'
    else:
        spatial_norm_path = None
    
    # Get key dataset parameters and pass them to workers to avoid reloading
    # the DataLoader.
    num_actions = data.dataset.num_actions if hasattr(data.dataset, 'num_actions') else 1
    t2m_mean = data.dataset.t2m_dataset.mean if hasattr(data.dataset.t2m_dataset, 'mean') else None
    t2m_std = data.dataset.t2m_dataset.std if hasattr(data.dataset.t2m_dataset, 'std') else None

    # Convert args to a dictionary for serialization.
    args_dict = vars(args)
    
    # Main loop: process batches until num_samples is reached.
    while True:
        # In data mode, stop once num_samples is reached.
        if is_using_data and args.num_samples > 0 and dataset_sample_idx >= args.num_samples:
            print(f"Reached num_samples: {args.num_samples}")
            break
        
        # In non-data mode, process only once.
        if not is_using_data and batch_idx > 0:
            break
        
        epoch_num = batch_idx // batches_per_epoch if is_using_data else 0
        batch_in_epoch = (batch_idx % batches_per_epoch) + 1 if is_using_data else 1
        
        if is_using_data:
            print(f'\n### Processing batch [{batch_idx + 1}] (Epoch {epoch_num + 1}, Batch {batch_in_epoch}/{batches_per_epoch})')
        else:
            print(f'\n### Processing batch [{batch_idx + 1}/1]')

        batch_npy_path = os.path.join(omnicontrol_output_dir, f'results_batch_{batch_idx:04d}.npy')
        if os.path.exists(batch_npy_path):
            print(f'  Batch results already exist at {batch_npy_path}, skipping...')
            # Load existing batch data to recover batch_size and update
            # dataset_sample_idx.
            try:
                batch_data = np.load(batch_npy_path, allow_pickle=True).item()
                current_batch_size = batch_data['dataset_end_idx'] - batch_data['dataset_start_idx']
                dataset_sample_idx += current_batch_size
                print(f'  Loaded existing batch info: batch_size={current_batch_size}, dataset_idx updated to {dataset_sample_idx}')
                
                # Important: when using data_iterator, this batch must also be
                # consumed from the iterator. Otherwise subsequent batches will
                # read the wrong data.
                if is_using_data:
                    try:
                        _ = next(data_iterator)  # Skip this batch.
                        print(f'  Skipped corresponding batch from data_iterator')
                    except StopIteration:
                        # The DataLoader is exhausted, so refresh the random
                        # seed and recreate the iterator.
                        current_epoch += 1
                        print(f'  DataLoader exhausted while skipping, starting epoch {current_epoch + 1}...')
                        
                        epoch_seed = args.seed + current_epoch * 10000
                        torch.manual_seed(epoch_seed)
                        np.random.seed(epoch_seed)
                        random.seed(epoch_seed)
                        print(f'  Updated DataLoader seed to {epoch_seed} for new epoch')
                        
                        data_iterator = iter(data)
                        try:
                            _ = next(data_iterator)  # Skip this batch.
                            print(f'  Skipped corresponding batch from new epoch data_iterator')
                        except StopIteration:
                            print(f'  Warning: new epoch data_iterator also exhausted')
                            break
                        
            except Exception as e:
                print(f'  Warning: Failed to load batch data for indexing: {e}')
                # If loading fails, try to get batch_size from data_iterator.
                if is_using_data:
                    try:
                        original_motion, model_kwargs = next(data_iterator)
                        current_batch_size = model_kwargs['y']['lengths'].shape[0]
                        dataset_sample_idx += current_batch_size
                        print(f'  Used data_iterator to get batch_size, dataset_idx updated to {dataset_sample_idx}')
                    except StopIteration:
                        # The DataLoader is exhausted, so refresh the random
                        # seed and recreate the iterator.
                        current_epoch += 1
                        print(f'  DataLoader exhausted while loading failed batch, starting epoch {current_epoch + 1}...')
                        
                        epoch_seed = args.seed + current_epoch * 10000
                        torch.manual_seed(epoch_seed)
                        np.random.seed(epoch_seed)
                        random.seed(epoch_seed)
                        print(f'  Updated DataLoader seed to {epoch_seed} for new epoch')
                        
                        data_iterator = iter(data)
                        try:
                            original_motion, model_kwargs = next(data_iterator)
                            current_batch_size = model_kwargs['y']['lengths'].shape[0]
                            dataset_sample_idx += current_batch_size
                            print(f'  Used new epoch data_iterator to get batch_size, dataset_idx updated to {dataset_sample_idx}')
                        except StopIteration:
                            break
            batch_idx += 1  # Skipping a batch should still advance the counter.
            continue
        
        # Get batch data
        if is_using_data:
            try:
                original_motion, model_kwargs = next(data_iterator)
            except StopIteration:
                # The DataLoader is exhausted, so recreate the iterator for a
                # new epoch.
                current_epoch += 1
                print(f'  DataLoader exhausted, starting epoch {current_epoch + 1}...')
                
                # Update the random seed so each epoch uses a different shuffle order.
                epoch_seed = args.seed + current_epoch * 10000  # Use a large gap to avoid collisions.
                torch.manual_seed(epoch_seed)
                np.random.seed(epoch_seed)
                random.seed(epoch_seed)
                print(f'  Updated DataLoader seed to {epoch_seed} for new epoch')
                
                data_iterator = iter(data)
                original_motion, model_kwargs = next(data_iterator)
            current_batch_size = model_kwargs['y']['lengths'].shape[0]
        else:
            current_batch_size = args.batch_size

        if is_using_data:
            if torch.is_tensor(original_motion):
                original_motion_serialized = original_motion.cpu().numpy()
            else:
                original_motion_serialized = np.asarray(original_motion)
        else:
            original_motion_serialized = None

        print(f'  Batch size: {current_batch_size}')
        print(f'  Processing {args.num_repetitions} repetitions in parallel across {num_gpus} GPUs...')
        
        # Serialize model_kwargs.
        model_kwargs_serialized = {}
        for k, v in model_kwargs['y'].items():
            if torch.is_tensor(v):
                model_kwargs_serialized[k] = v.cpu().numpy()
            else:
                model_kwargs_serialized[k] = v
        
        # Process repetitions in parallel with multiprocessing.
        # Use a larger queue to avoid workers blocking on put.
        # The queue capacity should be at least the number of repetitions.
        # Use spawn to avoid resource leaks caused by fork.
        mp_ctx = mp.get_context('spawn')
        result_queue = mp_ctx.Queue(maxsize=max(args.num_repetitions * 2, 100))
        active_processes = {}  # {rep_i: Process}
        finished_processes = {}  # cache finished processes for deferred joins
        retry_count = {}  # {rep_i: int} Retry count for each repetition.
        rep_gpu_history = {}  # {rep_i: [gpu_ids]} GPUs tried by each repetition.
        gpu_to_reps = {i: [] for i in range(num_gpus)}  # {gpu_id: [rep_i]} Repetitions currently assigned to each GPU.
        gpu_usage_count = {i: 0 for i in range(num_gpus)}  # Track GPU usage for load balancing.
        rep_start_time = {}  # {rep_i: float} Start time of each repetition for timeout detection.
        results = []
        
        start_time = time.time()
        
        # Dynamically manage processes by starting new ones and waiting for
        # completed ones.
        rep_i = 0
        last_activity_time = time.time()
        timeout = 900 if args.skip_visualization else 240+120*args.batch_size  # 10-minute timeout.
        
        while rep_i < args.num_repetitions or active_processes:
            # Clean up completed processes. Copy the key list first to avoid
            # mutating the dictionary during iteration.
            # First, try pulling completed results from the queue to avoid
            # blocking workers.
            completed_rep_ids = set()  # Successfully completed repetition IDs.
            
            # Read from the queue multiple times to avoid missing completed results.
            queue_read_attempts = 0
            max_queue_attempts = 10  # Try at most 10 times.
            while queue_read_attempts < max_queue_attempts:
                try:
                    result = result_queue.get(block=True, timeout=1)  # Use blocking mode with a short timeout.
                    if 'error' in result:
                        # Check whether this repetition should be retried.
                        failed_rep_id = result['rep_i']
                        current_retry = retry_count.get(failed_rep_id, 0)
                        
                        if current_retry < max_retries:
                            retry_count[failed_rep_id] = current_retry + 1
                            print(f'  Rep {failed_rep_id} failed (attempt {current_retry + 1}/{max_retries + 1}). Error: {result["error"]}')
                            print(f'  Will retry rep {failed_rep_id}...')
                            # Do not add it to results so it can be rerun.
                        else:
                            print(f'  Rep {failed_rep_id} failed after {max_retries + 1} attempts. Error: {result["error"]}')
                            results.append(result)  # Add the final failed result.
                            completed_rep_ids.add(failed_rep_id)
                    else:
                        results.append(result)
                        completed_rep_ids.add(result['rep_i'])
                    last_activity_time = time.time()
                    queue_read_attempts = 0  # Reset after a successful read and keep trying.
                except queue.Empty:
                    queue_read_attempts += 1
                    # The queue may be empty because a worker is about to put data.

            # Try reading the queue again before checking process state. This
            # avoids classifying a process as unexpectedly dead right after a put.
            extra_queue_check = 0
            while extra_queue_check < 3:
                try:
                    result = result_queue.get(block=False)
                    if 'error' in result:
                        failed_rep_id = result['rep_i']
                        current_retry = retry_count.get(failed_rep_id, 0)
                        if current_retry < max_retries:
                            retry_count[failed_rep_id] = current_retry + 1
                            print(f'  Rep {failed_rep_id} failed (attempt {current_retry + 1}/{max_retries + 1}). Error: {result["error"]}')
                        else:
                            results.append(result)
                            completed_rep_ids.add(failed_rep_id)
                    else:
                        results.append(result)
                        completed_rep_ids.add(result['rep_i'])
                    last_activity_time = time.time()
                except queue.Empty:
                    extra_queue_check += 1
            
            completed_reps = []
            freed_gpus = []
            for rep_id in list(active_processes.keys()):
                p = active_processes[rep_id]
                if not p.is_alive():
                    finished_processes[rep_id] = p  # join later after draining queue
                    completed_reps.append(rep_id)
                    rep_start_time.pop(rep_id, None)
                    
                    # Release the GPU used by this repetition.
                    for gpu_id, assigned_reps in gpu_to_reps.items():
                        if rep_id in assigned_reps:
                            assigned_reps.remove(rep_id)
                            freed_gpus.append(gpu_id)
                            print(f'  GPU {gpu_id} freed (rep {rep_id} completed), current load: {len(assigned_reps)}/{max_tasks_per_gpu}')
                            break
                    
                    last_activity_time = time.time()  # Refresh the activity timestamp.
            
            for rep_id in completed_reps:
                del active_processes[rep_id]
                
                # Check whether this repetition needs a retry because the
                # process died without returning a result.
                if rep_id not in completed_rep_ids:
                    # The process died without a result. Inspect the exit code.
                    exitcode = finished_processes[rep_id].exitcode
                    current_retry = retry_count.get(rep_id, 0)
                    
                    # exitcode != 0 means the process exited abnormally.
                    if exitcode != 0:
                        print(f'  Rep {rep_id} process died with exitcode {exitcode}.')
                    else:
                        print(f'  Rep {rep_id} process completed normally but no result in queue (possible race condition).')
                    
                    if current_retry < max_retries:
                        # Increase the retry count.
                        retry_count[rep_id] = current_retry + 1
                        print(f'  Scheduling retry for rep {rep_id} (attempt {current_retry + 2}/{max_retries + 1})...')
                    else:
                        # The retry limit was reached, so record a final failure.
                        print(f'  Rep {rep_id} failed after {max_retries + 1} attempts.')
                        error_result = {'rep_i': rep_id, 'error': f'Process died with exitcode {exitcode}'}
                        results.append(error_result)
                        completed_rep_ids.add(rep_id)
            
            # Get the currently available GPUs that are not at capacity.
            available_gpus = [gpu_id for gpu_id in range(num_gpus) if len(gpu_to_reps[gpu_id]) < max_tasks_per_gpu]
            
            # First check whether any repetitions need to be retried.
            reps_to_retry = []
            successful_rep_ids = {r['rep_i'] for r in results if 'error' not in r}
            # Get all repetitions that already have a final result, including failures.
            all_completed_rep_ids = {r['rep_i'] for r in results}
            
            for check_rep_id in range(rep_i):
                if check_rep_id not in all_completed_rep_ids and check_rep_id not in active_processes:
                    current_retry = retry_count.get(check_rep_id, 0)
                    # Add it to the retry queue if it has retries remaining.
                    if 0 < current_retry <= max_retries:
                        reps_to_retry.append(check_rep_id)
            
            # Prioritize retries.
            if reps_to_retry and available_gpus:
                retry_rep_id = reps_to_retry[0]
                
                # Get the list of GPUs previously tried by this repetition.
                tried_gpus = rep_gpu_history.get(retry_rep_id, [])
                
                # Prefer GPUs that have not been tried yet.
                untried_gpus = [g for g in available_gpus if g not in tried_gpus]
                
                if untried_gpus:
                    # Among untried GPUs, choose the one with the lowest current load.
                    gpu_id = min(untried_gpus, key=lambda g: (len(gpu_to_reps[g]), gpu_usage_count[g]))
                    print(f'  Retrying rep {retry_rep_id} on different GPU {gpu_id} (previous attempts on GPUs: {tried_gpus})')
                else:
                    # If all GPUs were tried, reuse the available GPU with the lowest load.
                    gpu_id = min(available_gpus, key=lambda g: (len(gpu_to_reps[g]), gpu_usage_count[g]))
                    print(f'  Retrying rep {retry_rep_id} on GPU {gpu_id} (all GPUs tried, reusing)')
                
                gpu_usage_count[gpu_id] += 1
                gpu_to_reps[gpu_id].append(retry_rep_id)
                rep_gpu_history[retry_rep_id].append(gpu_id)
                current_retry = retry_count[retry_rep_id]
                
                print(f'    Attempt {current_retry + 1}/{max_retries + 1}, GPU {gpu_id} load: {len(gpu_to_reps[gpu_id])}/{max_tasks_per_gpu}, total usage: {gpu_usage_count[gpu_id]}, active processes: {len(active_processes)}')
                p = mp_ctx.Process(target=worker_process, args=(
                    gpu_id, retry_rep_id, batch_idx, args_dict, args.model_path,
                    model_kwargs_serialized, current_batch_size, n_frames,
                    spatial_norm_path, dataset_sample_idx, t2m_mean, t2m_std, num_actions, original_motion_serialized, result_queue,
                    smplh_dict, skeleton, video_output_dir, sample_file_template,
                    sample_print_template, output_smpl_ik_path, fps
                ))
                p.start()
                active_processes[retry_rep_id] = p
                rep_start_time[retry_rep_id] = time.time()
                last_activity_time = time.time()  # Refresh the activity timestamp.
            # Start a new process if there are remaining repetitions and an
            # available GPU with remaining capacity.
            elif rep_i < args.num_repetitions and available_gpus:
                # Choose the GPU with the lowest load and total usage for load balancing.
                gpu_id = min(available_gpus, key=lambda g: (len(gpu_to_reps[g]), gpu_usage_count[g]))
                gpu_usage_count[gpu_id] += 1
                gpu_to_reps[gpu_id].append(rep_i)
                retry_count[rep_i] = 0  # Initialize the retry count.
                rep_gpu_history[rep_i] = [gpu_id]  # Record the first GPU used.
                
                print(f'  Starting rep {rep_i} on GPU {gpu_id}, GPU load: {len(gpu_to_reps[gpu_id])}/{max_tasks_per_gpu}, total usage: {gpu_usage_count[gpu_id]}, active processes: {len(active_processes)}')
                p = mp_ctx.Process(target=worker_process, args=(
                    gpu_id, rep_i, batch_idx, args_dict, args.model_path,
                    model_kwargs_serialized, current_batch_size, n_frames,
                    spatial_norm_path, dataset_sample_idx, t2m_mean, t2m_std, num_actions, original_motion_serialized, result_queue,
                    smplh_dict, skeleton, video_output_dir, sample_file_template,
                    sample_print_template, output_smpl_ik_path, fps
                ))
                p.start()
                active_processes[rep_i] = p
                rep_start_time[rep_i] = time.time()
                rep_i += 1
                last_activity_time = time.time()  # Refresh the activity timestamp.
            elif active_processes:
                # Only sleep when there are active processes to avoid busy waiting.
                import time as time_module
                time_module.sleep(0.1)
                
                # Check for timeouts. If a process produces no output for too
                # long, proactively mark it for failure or retry.
                now = time.time()
                stuck_reps = [
                    rep_id for rep_id in active_processes
                    if now - rep_start_time.get(rep_id, last_activity_time) > timeout
                ]
                if stuck_reps:
                    print(f'  WARNING: {len(stuck_reps)} reps exceeded timeout {timeout}s without results: {stuck_reps}')
                    for rep_id in stuck_reps:
                        p = active_processes.pop(rep_id)
                        rep_start_time.pop(rep_id, None)
                        finished_processes[rep_id] = p
                        # Release the GPU allocation.
                        for gpu_id, assigned_reps in gpu_to_reps.items():
                            if rep_id in assigned_reps:
                                assigned_reps.remove(rep_id)
                                print(f'    GPU {gpu_id} freed (timeout rep {rep_id}), current load: {len(assigned_reps)}/{max_tasks_per_gpu}')
                                break
                        
                        # Try terminating the process.
                        print(f'    Terminating rep {rep_id} process (PID: {p.pid})...')
                        try:
                            p.terminate()
                            p.join(timeout=5)
                            
                            # Force kill if graceful termination fails.
                            if p.is_alive():
                                print(f'    Process {rep_id} did not terminate gracefully, forcing kill...')
                                p.kill()
                                p.join(timeout=3)
                                if p.is_alive():
                                    print(f'    WARNING: Process {rep_id} (PID: {p.pid}) still alive after kill!')
                                else:
                                    print(f'    Process {rep_id} killed successfully')
                            else:
                                print(f'    Process {rep_id} terminated successfully')
                        except Exception as e:
                            print(f'    Error terminating process {rep_id}: {e}')
                            # Continue even if cleanup fails so the process is still marked done.

                        current_retry = retry_count.get(rep_id, 0)
                        if current_retry < max_retries:
                            retry_count[rep_id] = current_retry + 1
                            rep_gpu_history.setdefault(rep_id, [])
                            print(f'    Scheduling retry for rep {rep_id} (attempt {current_retry + 2}/{max_retries + 1})')
                        else:
                            error_result = {'rep_i': rep_id, 'error': f'Timeout after {timeout}s'}
                            results.append(error_result)
                            print(f'    Rep {rep_id} failed after timeout and max retries.')

                    # Continue immediately so the retry logic can take effect.
                    last_activity_time = now
                    continue
                else:
                    # There are still active processes and no timeout, so
                    # refresh the activity timestamp to avoid a false
                    # "no activity" condition.
                    last_activity_time = now
            else:
                # No active processes and no new tasks could indicate a dead loop.
                all_completed = {r['rep_i'] for r in results}
                print(f'  WARNING: No active processes and no new tasks to start!')
                print(f'    rep_i={rep_i}/{args.num_repetitions}')
                print(f'    completed_reps={sorted(all_completed)}')
                print(f'    retry_count={dict(sorted(retry_count.items()))}')
                print(f'    Missing reps: {set(range(rep_i)) - all_completed - set(active_processes.keys())}')
                # Loop once more so the exit condition check can run.
                time.sleep(0.1)

            # Exit if all results are collected and there are no active processes.
            # Check whether every repetition has finished, either successfully
            # or with a final failure.
            all_completed_or_failed_rep_ids = {r['rep_i'] for r in results}
            
            # Exit conditions:
            # 1. All repetitions have been started.
            # 2. There are no active processes.
            # 3. Every repetition has a final result.
            if rep_i >= args.num_repetitions and not active_processes and len(all_completed_or_failed_rep_ids) >= args.num_repetitions:
                print(f'  Exit condition met: rep_i={rep_i}, active={len(active_processes)}, completed={len(all_completed_or_failed_rep_ids)}/{args.num_repetitions}')
                break
        
        elapsed_time = time.time() - start_time
        print(f'  All repetitions completed in {elapsed_time:.2f} seconds')
        
        # Print GPU usage statistics.
        print(f'  GPU usage statistics (max_tasks_per_gpu={max_tasks_per_gpu}):')
        total_usage = sum(gpu_usage_count.values())
        for gpu_id in range(num_gpus):
            usage = gpu_usage_count[gpu_id]
            percentage = (usage / total_usage * 100) if total_usage > 0 else 0
            print(f'    GPU {gpu_id}: {usage} tasks ({percentage:.1f}%)')
        
        # Print retry statistics.
        if any(count > 0 for count in retry_count.values()):
            print(f'  Retry statistics:')
            for rep_id, count in sorted(retry_count.items()):
                if count > 0:
                    status = "succeeded" if any(r['rep_i'] == rep_id and 'error' not in r for r in results) else "failed"
                    tried_gpus = rep_gpu_history.get(rep_id, [])
                    print(f'    Rep {rep_id}: {count} retries, GPUs tried: {tried_gpus}, final status: {status}')
        
        # Collect any remaining results.
        while True:
            try:
                result = result_queue.get_nowait()
                if 'error' not in result or retry_count.get(result['rep_i'], 0) >= max_retries:
                    # Only add successful results or failures that reached the retry limit.
                    if result['rep_i'] not in {r['rep_i'] for r in results}:
                        results.append(result)
            except queue.Empty:
                break

        # Join all processes first to ensure they fully exit.
        print(f'  Waiting for all worker processes to complete...')
        all_processes = list(finished_processes.values()) + list(active_processes.values())
        
        for proc in all_processes:
            if proc.is_alive():
                try:
                    proc.join(timeout=15)
                    if proc.is_alive():
                        print(f'    Warning: Process {proc.pid} still alive after 15s, terminating...')
                        proc.terminate()
                        proc.join(timeout=5)
                        if proc.is_alive():
                            print(f'    Warning: Process {proc.pid} still alive after terminate, killing...')
                            proc.kill()
                            proc.join(timeout=2)
                except Exception as e:
                    print(f'    Error joining process {proc.pid}: {e}')
        
        # Wait briefly so the OS can reclaim all process resources.
        import time as time_module
        time_module.sleep(0.5)
        
        # Clean up queue resources. Use cancel_join_thread to avoid blocking.
        try:
            result_queue.cancel_join_thread()  # Cancel join immediately without waiting.
            result_queue.close()
        except Exception as e:
            # Ignore cleanup errors, especially FileNotFoundError.
            if not isinstance(e, FileNotFoundError):
                print(f'    Warning during queue cleanup: {type(e).__name__}')
        
        # Sort by rep_i to preserve metric alignment.
        results.sort(key=lambda x: x['rep_i'])
        print(f'  Results collected and sorted by rep_i: {[r["rep_i"] for r in results]}')
        
        # Check for results that still failed after all retries.
        errors = [r for r in results if 'error' in r]
        if errors:
            print(f"\n  WARNING: Errors occurred in repetitions after {max_retries + 1} attempts: {[r['rep_i'] for r in errors]}")
            for err in errors:
                print(f"    Rep {err['rep_i']}: {err['error']}")
                print(f"    Retry history: attempted {retry_count.get(err['rep_i'], 0) + 1} times")
            print(f"  Skipping batch {batch_idx} due to errors.\n")
            continue
        
        # Merge metrics in rep_i order.
        for result in results:
            if result.get('res_metrics') is not None:
                res_metrics = result['res_metrics']
                # Merge metrics from this repetition into global res
                for metric_key in res_metrics:
                    if isinstance(res_metrics[metric_key], dict):
                        res[metric_key].update(res_metrics[metric_key])
        
        # Organize outputs.
        all_motions = []
        all_output_vectors = []
        all_hint = []
        all_hint_for_vis = []
        all_lengths = []
        all_text = []
        all_tokens = []
        # for smpl formats
        all_poses = []
        all_trans = []
        
        # Organize outputs in sorted result order to preserve rep_i order.
        for result in results:
            all_motions.append(result['sample'])
            all_output_vectors.append(result['output_vector'])
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            
            if result['hint_np'] is not None:
                all_hint.append(result['hint_np'])
                all_hint_for_vis.append(result['hint_for_vis_np'])

            if result['all_poses'] is not None:
                all_poses.append(result['all_poses'])
            
            if result['all_trans'] is not None:
                all_trans.append(result['all_trans'])
            
            if args.unconstrained:
                all_text += ['unconstrained'] * current_batch_size
            else:
                # Use the actual text returned by the worker to avoid mismatches.
                if 'actual_text' in result and result['actual_text'] is not None:
                    all_text += result['actual_text']
                else:
                    # Fallback: use the original text for backward compatibility.
                    text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                    all_text += model_kwargs['y'][text_key]
                
                all_tokens += model_kwargs['y']['tokens']
                
        
        # Concatenate for saving
        all_motions = np.concatenate(all_motions, axis=0)
        all_lengths = np.concatenate(all_lengths, axis=0)
        all_output_vectors = np.concatenate(all_output_vectors, axis=0)
        
        if len(all_hint) > 0:
            all_hint = np.concatenate(all_hint, axis=0)
            all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)
        else:
            all_hint_for_vis = None

        if len(all_poses) > 0:
            all_poses = np.concatenate(all_poses, axis=0)
        else:
            all_poses = None
        
        if len(all_trans) > 0:
            all_trans = np.concatenate(all_trans, axis=0)
        else:
            all_trans = None
        
        # Get original motion
        if is_using_data:
            all_original_motion = original_motion.permute(0, 2, 3, 1).cpu().numpy()[:,0]
        else:
            all_original_motion = None
        
        print(f"Created {len(all_motions)} samples for this batch")

        # Visualize current batch (per-sample renders already done inside workers)
        if not args.skip_visualization:
            print(f"Combining visualizations for batch {batch_idx + 1}/{num_batches}...")
            for sample_i in range(current_batch_size):
                current_dataset_idx = dataset_sample_idx + sample_i
                rep_files = []
                caption_idx = sample_i if len(all_text) >= current_batch_size else 0
                caption = all_text[caption_idx] if not args.unconstrained else 'unconstrained'

                for rep_i in range(args.num_repetitions):
                    save_file = sample_file_template.format(current_dataset_idx, rep_i)
                    animation_save_path = os.path.join(video_output_dir, save_file)
                    if os.path.exists(animation_save_path):
                        rep_files.append(animation_save_path)
                    else:
                        print(f"Warning: missing visualization file {animation_save_path}")
                if len(rep_files) == 0:
                    print(f"Skipping sample {current_dataset_idx} due to missing visualization outputs")
                    continue

                if args.num_repetitions > 1:
                    motion_vis_y_up_to_plot = []
                    hint_vis_y_up_to_plot = []

                    for rep_i in range(args.num_repetitions):
                        idx = rep_i * current_batch_size + sample_i
                        length = all_lengths[idx]
                        motion = all_motions[idx].transpose(2, 0, 1)[:length]
                        hint = all_hint_for_vis[idx] if all_hint_for_vis is not None else None

                        if args.use_smpl:
                            x, y, z = T(motion.copy())
                            motion = T(np.stack((x, z, -y), axis=0))
                            if hint is not None:
                                x, y, z = T(hint.copy()[:length])
                                hint = T(np.stack((x, z, -y), axis=0))
                        else:
                            motion = motion[:length]
                            if hint is not None:
                                hint = hint[:length]

                        motion_vis_y_up_to_plot.append(motion)
                        hint_vis_y_up_to_plot.append(hint)

                    multi_rep_save_file = f'all_sample_{current_dataset_idx}.mp4'
                    multi_rep_save_path = os.path.join(video_output_dir, multi_rep_save_file)
                    hint_arg = hint_vis_y_up_to_plot[0] if hint_vis_y_up_to_plot and hint_vis_y_up_to_plot[0] is not None else None

                    try:
                        plot_3d_motion_multi(multi_rep_save_path, skeleton,
                                            motion_vis_y_up_to_plot,
                                            dataset=args.dataset, title=caption if args.cond_mode != 'only_spatial' else None,
                                            hint=hint_arg,
                                            fps=fps, elev=-60 if args.use_smpl else 120, 
                                            azim=90 if args.use_smpl else -90)
                    except:
                        try:
                            plot_3d_motion_multi(multi_rep_save_path, skeleton,
                                                motion_vis_y_up_to_plot,
                                                dataset=args.dataset, title=caption if args.cond_mode != 'only_spatial' else None,
                                                hint=hint_arg,
                                                fps=fps, elev=-60 if args.use_smpl else 120, 
                                                azim=90 if args.use_smpl else -90)
                        except Exception as e:
                            print(f"Warning: failed to create multi-repetition video for sample {current_dataset_idx}: {e}")
                            multi_rep_save_path = None

                    if multi_rep_save_path is not None and os.path.exists(multi_rep_save_path):
                        print(f"Saved multi-repetition video to [{multi_rep_save_path}]")
                        rep_files.append(multi_rep_save_path)

                        if args.visualize_mesh and (args.use_smpl or not args.skip_ik):
                            vertices_vis_mesh_rep = []
                            points_seq = None
                            mesh_files_missing = False
                            for rep_i in range(args.num_repetitions):
                                single_mesh_path = os.path.join(
                                    video_output_dir,
                                    sample_file_template.format(current_dataset_idx, rep_i)
                                ).replace('.mp4', '_mesh.npz')

                                if not os.path.exists(single_mesh_path):
                                    mesh_files_missing = True
                                    print(f"Warning: missing mesh data {single_mesh_path}")
                                    break

                                with np.load(single_mesh_path, allow_pickle=True) as mesh_data:
                                    vertices_vis_mesh_rep.append(mesh_data['vertices'])
                                    if 'points_seq' in mesh_data.files:
                                        points_seq = mesh_data['points_seq']

                            if not mesh_files_missing and len(vertices_vis_mesh_rep) == args.num_repetitions:
                                if args.use_smpl:
                                    cam_rot_matrix = R.from_euler('xz', [90, 180], degrees=True).as_matrix()
                                    cam_offset=[0.0, 5, 0.001]
                                else:
                                    cam_rot_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
                                    cam_offset=[0.0, -5, 0.001]

                                mesh_renderer(vertices=vertices_vis_mesh_rep, 
                                            output=multi_rep_save_path.replace('.mp4', '_mesh.mp4'),
                                            points_seq=points_seq, cam_rot=cam_rot_matrix, cam_offset=cam_offset,
                                            point_rad=0.10, put_ground=False, follow_camera=False)

                                mesh_dir = multi_rep_save_path.replace('.mp4', '_mesh')
                                if os.path.exists(mesh_dir):
                                    shutil.rmtree(mesh_dir, ignore_errors=True)

                all_sample_files = save_multiple_samples(
                    args, video_output_dir,
                    row_print_template, all_print_template, row_file_template, all_file_template,
                    caption, num_samples_in_out_file, rep_files, all_sample_files, current_dataset_idx
                )
        else:
            for sample_i in range(current_batch_size):
                current_dataset_idx = dataset_sample_idx + sample_i
                
                for rep_i in range(args.num_repetitions):
                    caption = all_text[rep_i * current_batch_size + sample_i]
                    length = all_lengths[rep_i * current_batch_size + sample_i]
                    motion = all_motions[rep_i * current_batch_size + sample_i].transpose(2, 0, 1)[:length]
                                        
                    if not args.use_smpl and not args.skip_ik:
                        ik_path = os.path.join(output_smpl_ik_path, f'ik_{current_dataset_idx}_{rep_i}.npz')
                        
                        j2s = joints2smpl(num_frames=length, device_id=0, cuda=True)
                        opt_dict = j2s.joint2smpl_amass(motion.copy())
                        poses = opt_dict['poses']
                        poses[:,:3] = axis_angle_yup_to_zup(poses[:,:3], 90)
                        np.savez(ik_path, poses=poses, trans=yup_to_zup(opt_dict['trans']),
                                    betas=opt_dict['betas'], num_betas=10, gender='neutral',
                                    mocap_frame_rate=fps, text=caption)

                    elif args.use_smpl:
                        ik_path = os.path.join(output_smpl_ik_path, f'ik_{current_dataset_idx}_{rep_i}.npz')
                        np.savez(ik_path, poses=all_poses[rep_i * current_batch_size + sample_i][:length], 
                                    trans=all_trans[rep_i * current_batch_size + sample_i][:length],
                                    betas=np.zeros(10), num_betas=10, gender='neutral',
                                    mocap_frame_rate=fps, text=caption)
        # Save batch results
        
        print(f"Saving batch results to [{batch_npy_path}]")
        
        batch_data = {
            'motion': all_motions,
            'text': all_text,
            'tokens': all_tokens,
            'lengths': all_lengths,
            'hint': all_hint_for_vis,
            'batch_idx': batch_idx,
            'dataset_start_idx': dataset_sample_idx,
            'dataset_end_idx': dataset_sample_idx + current_batch_size,
            'num_repetitions': args.num_repetitions,
            'output_vectors': all_output_vectors,
            'original_motion': all_original_motion
        }
        np.save(batch_npy_path, batch_data)
        
        # Append text information
        text_path = os.path.join(omnicontrol_output_dir, 'results.txt')
        len_path = os.path.join(omnicontrol_output_dir, 'results_len.txt')
        
        with open(text_path, 'a') as fw:
            for sample_i in range(current_batch_size):
                caption = all_text[sample_i]
                fw.write(f"{caption}\n")
        
        with open(len_path, 'a') as fw:
            for sample_i in range(current_batch_size):
                length = all_lengths[sample_i]
                fw.write(f"{length}\n")
        
        print(f"Batch {batch_idx + 1} completed. Total time: {elapsed_time:.2f}s")
        dataset_sample_idx += current_batch_size
        batch_idx += 1  # Advance the global batch counter.

        # Save intermediate metrics (without sorting for efficiency)
        # Print metric counts before each save.
        metric_list = ['skate_ratio', 'mean_error', 'feet_height', 'jerk'] + traj_err_key
        print(f"  Current metrics count:")
        for metric in metric_list:
            if metric in res and isinstance(res[metric], dict):
                print(f"    {metric}: {len(res[metric])} entries")
        
        metrics_path = os.path.join(out_path, "omnicontrol_sorted.json")
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        print(f"  Intermediate metrics saved to {metrics_path}")
        print(f"  Total accumulated samples in metrics: {dataset_sample_idx}")
        
        # End of the while True loop.

    # Save final metrics (without sorting for efficiency)
    print(f"\n### All batches processed. Total samples: {dataset_sample_idx}")
    print(f"Total generations: {dataset_sample_idx * args.num_repetitions}")
    
    final_metrics_path = os.path.join(out_path, "omnicontrol_sorted_final.json")
    with open(final_metrics_path, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    print(f"Final metrics saved to {final_metrics_path}")
    
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='train',
                              hml_mode='train',
                              use_omomo=args.omomo, use_dpo=False, dpo_data_root=args.dpo_data_root,
                              mask_type=args.mask_type, density=args.density, multi_text=args.multi_text,
                              use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part)
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    # The multiprocessing start method must be set here.
    mp.set_start_method('spawn', force=True)
    main()
