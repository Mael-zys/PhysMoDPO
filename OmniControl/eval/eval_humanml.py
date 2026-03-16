# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import os
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.extract_metrics import extract_metrics
import random
import subprocess 
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy
from utils.calculate_TMR_score.tmr_eval_wrapper import calculate_tmr_metrics
from utils.runtime_paths import (
    PROTO_MOTIONS3_ROOT,
    PROTO_MOTIONS_ROOT,
    resolve_omnicontrol_path,
)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if len(batch) == 7:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                else:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))



def evaluate_tmr_metrics(tmr_forward, motion_loaders, file, calculate_retrieval=False):
    m2m_score_dict = OrderedDict({})
    m2t_score_dict = OrderedDict({})
    tmr_r1_dict = OrderedDict({})
    tmr_r3_dict = OrderedDict({})
    
    print('========== Evaluating TMR Metrics ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        if motion_loader_name == 'ground truth':
            continue

        motion_loader.dataset.output_gt_joints = True

        all_m2t_scores = []
        all_m2m_scores = []
        all_r1_scores = []
        all_r3_scores = []

        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _, hint = batch

                mean_for_eval = motion_loader.dataset.dataloader.dataset.mean_for_eval
                std_for_eval = motion_loader.dataset.dataloader.dataset.std_for_eval
                motions = motions * std_for_eval + mean_for_eval
                motions = motions.float()

                batch_size = motions.shape[0]
                gt_motions = motions[:, motions.shape[1]//2:].clone().reshape(batch_size,-1, 263).cpu().numpy()  # second half is gt
                pred_motions = motions[:, :motions.shape[1]//2].clone().reshape(batch_size,-1, 263).cpu().numpy()  # first half is gen

                motions_guofeats_gt = [gt_motions[ijk][:m_lens[ijk]] for ijk in range(batch_size)]
                motions_guofeats_pred = [pred_motions[ijk][:m_lens[ijk]] for ijk in range(batch_size)]
                # calculate tmr metrics
                metrics = calculate_tmr_metrics(tmr_forward, texts_gt=captions, motions_guofeats_gt=motions_guofeats_gt, 
                                                motions_guofeats_pred=motions_guofeats_pred, calculate_retrieval=calculate_retrieval)
                
                all_m2t_scores.append(metrics['m2t_score'])
                all_m2m_scores.append(metrics['m2m_score'])
                if calculate_retrieval:
                    all_r1_scores.append(metrics['m2t_top_1'])
                    all_r3_scores.append(metrics['m2t_top_3'])
        
        # Calculate average scores
        m2t_score = np.mean(all_m2t_scores)
        m2m_score = np.mean(all_m2m_scores)
        m2t_score_dict[motion_loader_name] = m2t_score
        m2m_score_dict[motion_loader_name] = m2m_score

        print(f'---> [{motion_loader_name}] M2T Score: {m2t_score:.4f}')
        print(f'---> [{motion_loader_name}] M2T Score: {m2t_score:.4f}', file=file, flush=True)

        print(f'---> [{motion_loader_name}] M2M Score: {m2m_score:.4f}')
        print(f'---> [{motion_loader_name}] M2M Score: {m2m_score:.4f}', file=file, flush=True)

        if calculate_retrieval:
            tmr_r1 = np.mean(all_r1_scores)
            tmr_r3 = np.mean(all_r3_scores)
            tmr_r1_dict[motion_loader_name] = tmr_r1
            tmr_r3_dict[motion_loader_name] = tmr_r3

            print(f'---> [{motion_loader_name}] TMR R@1: {tmr_r1:.4f}')
            print(f'---> [{motion_loader_name}] TMR R@1: {tmr_r1:.4f}', file=file, flush=True)

            print(f'---> [{motion_loader_name}] TMR R@3: {tmr_r3:.4f}')
            print(f'---> [{motion_loader_name}] TMR R@3: {tmr_r3:.4f}', file=file, flush=True)
        
        motion_loader.dataset.output_gt_joints = False
    
    return m2m_score_dict, m2t_score_dict, tmr_r1_dict, tmr_r3_dict


def evaluate_control(motion_loaders, file, train_platform=None, training_args=None, training_step=0, 
                     visualize=False, use_smpl=False, cond_mode=None, log_file=None, sim_gpu=None, save_figure=False):
    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})
    simulation_error = OrderedDict({})
    simulation_error_max = OrderedDict({})
    simulation_sr_02 = OrderedDict({})
    simulation_sr_05 = OrderedDict({})
    power = OrderedDict({})
    feet_height_dict = OrderedDict({})
    jerk_dict = OrderedDict({})

    for motion_loader_name in motion_loaders:
        if motion_loader_name == 'ground truth':
            continue
        motion_loader = motion_loaders[motion_loader_name]
        
        if use_smpl:
            motion_loader.dataset.output_format = 'smpl'
        motion_loader.dataset.output_gt_joints = True

        print('========== Evaluating Control ==========')
        # all_dist = []
        all_size = 0
        dist_sum = 0
        skate_ratio_sum = 0
        feet_height_sum = 0
        jerk_sum = 0
        traj_err = []
        traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _, hint = batch
                # process motion
                # sample to motion
                if use_smpl:
                    n_joints = 24
                    motions = motions.float()
                    x, y, z = T(motions)
                    motions = T(np.stack((x, z, -y), axis=0)) # change to y-up
                    motions = torch.from_numpy(motions)

                    gt_motions = motions[:, motions.shape[1]//2:].clone()  # second half is gt
                    motions = motions[:, :motions.shape[1]//2]  # first half is gen
                    
                else:
                    mean_for_eval = motion_loader.dataset.dataloader.dataset.mean_for_eval
                    std_for_eval = motion_loader.dataset.dataloader.dataset.std_for_eval
                    motions = motions * std_for_eval + mean_for_eval
                    motions = motions.float()

                    n_joints = 22 if motions.shape[-1] == 263 else 21

                    gt_motions = motions[:, motions.shape[1]//2:].clone()  # second half is gt
                    gt_motions = recover_from_ric(gt_motions, n_joints)
                    
                    motions = motions[:, :motions.shape[1]//2]  # first half is gen
                    motions = recover_from_ric(motions, n_joints)
                    if n_joints == 21:
                        # kit
                        motions = motions * 0.001
                
                # foot skating error
                if n_joints == 21:
                    skate_ratio, skate_vel = calculate_skating_ratio_kit(motions.permute(0, 2, 3, 1), m_lens)  # [batch_size]
                    feet_height = calculate_feet_height_kit(motions.permute(0, 2, 3, 1), m_lens)  # [batch_size]
                else:
                    skate_ratio, skate_vel = calculate_skating_ratio(motions.permute(0, 2, 3, 1), m_lens)  # [batch_size]
                    feet_height = calculate_feet_height(motions.permute(0, 2, 3, 1), m_lens)  # [batch_size]
                jerk = calculate_jerk(motions.permute(0, 2, 3, 1), m_lens)  # [batch_size]

                skate_ratio_sum += skate_ratio.sum()
                feet_height_sum += feet_height.sum()
                jerk_sum += jerk.sum()

                # control l2 error
                # process hint
                if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text'):
                    mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(dim=-1, keepdim=True) != 0

                    raw_mean = motion_loader.dataset.dataloader.dataset.t2m_dataset.raw_mean
                    raw_std = motion_loader.dataset.dataloader.dataset.t2m_dataset.raw_std

                    hint = hint * raw_std + raw_mean
                    if n_joints == 21:
                        hint = hint * 0.001
                    hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask_hint

                    if use_smpl:
                        # convert hint to y-up
                        x, y, z = T(hint)
                        hint = T(np.stack((x, z, -y), axis=0)) # change to y-up
                        hint = torch.from_numpy(hint)

                    for draw_idx, (motion, h, mask) in enumerate(zip(motions, hint, mask_hint)):
                        control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                        mean_error = control_error.sum() / mask.sum()
                        dist_sum += mean_error
                        control_error = control_error.reshape(-1)
                        mask = mask.reshape(-1)
                        err_np = calculate_trajectory_error(control_error, mean_error, mask)
                        traj_err.append(err_np)

                        if use_smpl and not motion_loader.dataset.eval_after_simulation:
                            # get folder of log_file
                            log_folder = log_file.replace('.log','')
                            ik_folder = os.path.join(log_folder, 'amass_format', 'ik')
                            os.makedirs(ik_folder, exist_ok=True)
                            ik_path = os.path.join(ik_folder, f'ik_{idx}_{draw_idx}.npz')
                            np.savez(ik_path, poses=word_embeddings[draw_idx][:m_lens[draw_idx]], 
                                        trans=pos_one_hots[draw_idx][:m_lens[draw_idx]],
                                        betas=np.zeros(10), num_betas=10, gender='neutral',
                                        mocap_frame_rate=12.5 if n_joints == 21 else 20, 
                                        text=captions[draw_idx] if (training_args is not None and training_args.cond_mode != 'only_spatial') or (cond_mode is not None and cond_mode != 'only_spatial') else None)
                        
                        # temporary add visualization here
                        if ((visualize and train_platform is not None) or save_figure) and draw_idx < 5 and idx == 0:
                            from data_loaders.humanml.utils.plot_script import plot_3d_motion

                            skeleton = paramUtil.kit_kinematic_chain if n_joints == 21 else paramUtil.t2m_kinematic_chain

                            try:
                                if save_figure:
                                    save_folder = os.path.join(log_file.replace('.log',''), 'eval_control_vis')
                                    os.makedirs(save_folder, exist_ok=True)

                                video = plot_3d_motion(os.path.join(save_folder, f'{draw_idx}_{motion_loader_name}.mp4') if save_figure else '', 
                                        skeleton, motion[:m_lens[draw_idx], :].clone().cpu().numpy(), dataset='kit' if n_joints == 21 else 'humanml', figsize=(4, 4), radius=2.5,
                                        title=captions[draw_idx] if (training_args is not None and training_args.cond_mode != 'only_spatial') or (cond_mode is not None and cond_mode != 'only_spatial') else None,
                                        fps=12.5 if n_joints == 21 else 20, hint=h.clone().cpu().numpy() if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text') else None,
                                        tensorboard_vis=True if train_platform is not None else False, elev=-60 if use_smpl else 120, 
                                        azim=90 if use_smpl else -90)
                                if train_platform is not None:
                                    # video (T, H, W, 3) -> (1, 3, T, H, W)
                                    video = video.transpose(0, 3, 1, 2)[None]
                                    train_platform.report_video(f'eval_control_vis/{draw_idx}_' + motion_loader_name, video, training_step, fps=12.5 if n_joints == 21 else 20)


                                # visualize gt as well
                                video = plot_3d_motion(os.path.join(save_folder, f'{draw_idx}_gt_{motion_loader_name}.mp4') if save_figure else '', 
                                        skeleton, gt_motions[draw_idx, :m_lens[draw_idx]].clone().cpu().numpy(), dataset='kit' if n_joints == 21 else 'humanml', figsize=(4, 4), radius=2.5,
                                        title=captions[draw_idx] if (training_args is not None and training_args.cond_mode != 'only_spatial') or (cond_mode is not None and cond_mode != 'only_spatial') else None,
                                        fps=12.5 if n_joints == 21 else 20, hint=h.clone().cpu().numpy() if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text') else None,
                                        tensorboard_vis=True if train_platform is not None else False, elev=-60 if use_smpl else 120, 
                                        azim=90 if use_smpl else -90)
                                if train_platform is not None:
                                    # video (T, H, W, 3) -> (1, 3, T, H, W)
                                    video = video.transpose(0, 3, 1, 2)[None]
                                    train_platform.report_video(f'eval_control_vis/{draw_idx}_gt_' + motion_loader_name, video, training_step, fps=12.5 if n_joints == 21 else 20)
                            
                            except:
                                continue

                # for only_text condition    
                else:
                    for draw_idx, motion in enumerate(motions):
                        if use_smpl and not motion_loader.dataset.eval_after_simulation:
                            # get folder of log_file
                            log_folder = os.path.dirname(log_file)
                            ik_folder = os.path.join(log_folder, 'amass_format')
                            os.makedirs(ik_folder, exist_ok=True)
                            ik_path = os.path.join(ik_folder, f'ik_{idx}_{draw_idx}.npz')
                            np.savez(ik_path, poses=word_embeddings[draw_idx][:m_lens[draw_idx]], 
                                        trans=pos_one_hots[draw_idx][:m_lens[draw_idx]],
                                        betas=np.zeros(10), num_betas=10, gender='neutral',
                                        mocap_frame_rate=12.5 if n_joints == 21 else 20, 
                                        text=captions[draw_idx] if (training_args is not None and training_args.cond_mode != 'only_spatial') or (cond_mode is not None and cond_mode != 'only_spatial') else None)
                        
                        # temporary add visualization here
                        if visualize and train_platform is not None and draw_idx < 5 and idx == 0:
                            from data_loaders.humanml.utils.plot_script import plot_3d_motion

                            skeleton = paramUtil.kit_kinematic_chain if n_joints == 21 else paramUtil.t2m_kinematic_chain

                            try:
                                video = plot_3d_motion('', skeleton, motion[:, :m_lens[draw_idx]].clone().cpu().numpy(), dataset='kit' if n_joints == 21 else 'humanml', figsize=(4, 4), radius=2.5,
                                        title=captions[draw_idx] if training_args.cond_mode != 'only_spatial' else None,
                                        fps=12.5 if n_joints == 21 else 20, hint=h.clone().cpu().numpy() if training_args.cond_mode != 'only_text' else None,
                                        tensorboard_vis=True, elev=-60 if use_smpl else 120, 
                                        azim=90 if use_smpl else -90)
                                # video (T, H, W, 3) -> (1, 3, T, H, W)
                                video = video.transpose(0, 3, 1, 2)[None]
                                train_platform.report_video(f'eval_control_vis/{draw_idx}_' + motion_loader_name, video, training_step, fps=12.5 if n_joints == 21 else 20)


                                # visualize gt as well
                                video = plot_3d_motion('', skeleton, gt_motions[draw_idx, :m_lens[draw_idx]].clone().cpu().numpy(), dataset='kit' if n_joints == 21 else 'humanml', figsize=(4, 4), radius=2.5,
                                        title=captions[draw_idx] if training_args.cond_mode != 'only_spatial' else None,
                                        fps=12.5 if n_joints == 21 else 20, hint=None,
                                        tensorboard_vis=True, elev=-60 if use_smpl else 120, 
                                        azim=90 if use_smpl else -90)
                                # video (T, H, W, 3) -> (1, 3, T, H, W)
                                video = video.transpose(0, 3, 1, 2)[None]
                                train_platform.report_video(f'eval_control_vis/{draw_idx}_gt_' + motion_loader_name, video, training_step, fps=12.5 if n_joints == 21 else 20)
                            
                            except:
                                continue

                all_size += motions.shape[0]          

            
            if use_smpl and not motion_loader.dataset.eval_after_simulation:
                log_folder = log_file.replace('.log','')
                absolute_output_path = resolve_omnicontrol_path(log_folder)

                # use specified GPU for simulation evaluation
                gpu_id = sim_gpu if sim_gpu is not None else (training_args.sim_gpu if training_args is not None and hasattr(training_args, 'sim_gpu') else 0)
                subprocess.call(
                    ["bash", str(PROTO_MOTIONS_ROOT / "run_deepmimic.sh"), absolute_output_path, str(gpu_id)],
                    cwd=str(PROTO_MOTIONS_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                results = extract_metrics(os.path.join(absolute_output_path, 'deepmimic_output/eval_agent.log'))
                
                simulation_error[motion_loader_name] = results['eval/gt_err'][-1]  # only one value

                simulation_error_max[motion_loader_name] = results['eval/gt_err_max'][-1]

                simulation_sr_02[motion_loader_name] = results['eval/tracking_success_rate_0.2'][-1]
                simulation_sr_05[motion_loader_name] = results['eval/tracking_success_rate_0.5'][-1]

                power[motion_loader_name] = results['eval/power'][-1]  # only one value

                print(f'---> [{motion_loader_name}] simulation error: {simulation_error[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation error: {simulation_error[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation error max: {simulation_error_max[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation error max: {simulation_error_max[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation SR 0.2: {simulation_sr_02[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation SR 0.2: {simulation_sr_02[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation SR 0.5: {simulation_sr_05[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation SR 0.5: {simulation_sr_05[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] power: {power[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] power: {power[motion_loader_name]:.4f}', file=file, flush=True)
            elif use_smpl and motion_loader.dataset.eval_after_simulation:
                log_folder = log_file.replace('.log','')
                absolute_output_path = resolve_omnicontrol_path(log_folder)
                # already have results from previous simulation
                # get all folders whose name is like '27_0'
                import re
                
                # Find all folders matching pattern: digit(s)_digit(s)
                pattern = re.compile(r'^\d+_\d+$')
                all_result_folders = []
                
                if os.path.exists(absolute_output_path):
                    for item in os.listdir(absolute_output_path):
                        if pattern.match(item) and os.path.isdir(os.path.join(absolute_output_path, item)):
                            all_result_folders.append(item)
                
                # Read results from all matching folders
                all_gt_err = []
                all_gt_err_max = []
                all_power = []
                
                for folder in all_result_folders:
                    deepmimic_output = os.path.join(absolute_output_path, folder, 'deepmimic_output')
                    
                    # Read gt_err
                    gt_err_file = os.path.join(deepmimic_output, 'all_motions_with_gt_err_0.txt')
                    if os.path.exists(gt_err_file):
                        with open(gt_err_file, 'r') as f:
                            values = [float(line.strip().split(':')[1]) for line in f if line.strip()]
                            all_gt_err.extend(values)
                    
                    # Read gt_err_max
                    gt_err_max_file = os.path.join(deepmimic_output, 'all_motions_with_gt_err_max_0.txt')
                    if os.path.exists(gt_err_max_file):
                        with open(gt_err_max_file, 'r') as f:
                            values = [float(line.strip().split(':')[1]) for line in f if line.strip()]
                            all_gt_err_max.extend(values)
                    
                    # Read power
                    power_file = os.path.join(deepmimic_output, 'all_motions_with_power_0.txt')
                    if os.path.exists(power_file):
                        with open(power_file, 'r') as f:
                            values = [float(line.strip().split(':')[1]) for line in f if line.strip()]
                            all_power.extend(values)
                
                # Calculate mean values
                if all_gt_err:
                    simulation_error[motion_loader_name] = np.mean(all_gt_err)
                else:
                    simulation_error[motion_loader_name] = 0.0
                
                if all_gt_err_max:
                    simulation_error_max[motion_loader_name] = np.mean(all_gt_err_max)
                else:
                    simulation_error_max[motion_loader_name] = 0.0
                
                if all_power:
                    power[motion_loader_name] = np.mean(all_power)
                else:
                    power[motion_loader_name] = 0.0
                
                # For SR metrics, calculate based on gt_err thresholds
                if all_gt_err_max:
                    simulation_sr_02[motion_loader_name] = np.mean([err < 0.2 for err in all_gt_err_max])
                    simulation_sr_05[motion_loader_name] = np.mean([err < 0.5 for err in all_gt_err_max])
                else:
                    simulation_sr_02[motion_loader_name] = 0.0
                    simulation_sr_05[motion_loader_name] = 0.0
                
                print(f'---> [{motion_loader_name}] simulation error: {simulation_error[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation error: {simulation_error[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation error max: {simulation_error_max[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation error max: {simulation_error_max[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation SR 0.2: {simulation_sr_02[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation SR 0.2: {simulation_sr_02[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] simulation SR 0.5: {simulation_sr_05[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] simulation SR 0.5: {simulation_sr_05[motion_loader_name]:.4f}', file=file, flush=True)

                print(f'---> [{motion_loader_name}] power: {power[motion_loader_name]:.4f}')
                print(f'---> [{motion_loader_name}] power: {power[motion_loader_name]:.4f}', file=file, flush=True)

            # Skating evaluation
            skating_score = skate_ratio_sum / all_size
            skating_ratio_dict[motion_loader_name] = skating_score

            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
            print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)

            feet_height_mean = feet_height_sum / all_size
            feet_height_dict[motion_loader_name] = feet_height_mean

            print(f'---> [{motion_loader_name}] Feet Height: {feet_height_mean:.4f}')
            print(f'---> [{motion_loader_name}] Feet Height: {feet_height_mean:.4f}', file=file, flush=True)

            jerk_mean = jerk_sum / all_size
            jerk_dict[motion_loader_name] = jerk_mean

            print(f'---> [{motion_loader_name}] Jerk: {jerk_mean:.4f}')
            print(f'---> [{motion_loader_name}] Jerk: {jerk_mean:.4f}', file=file, flush=True)


            if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text'):
                # l2 dist
                dist_mean = dist_sum / all_size
                l2_dict[motion_loader_name] = dist_mean

                ### For trajecotry evaluation from GMD ###
                traj_err = np.stack(traj_err).mean(0)
                trajectory_score_dict[motion_loader_name] = traj_err

                print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
                print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}', file=file, flush=True)
                
                line = f'---> [{motion_loader_name}] Trajectory Error: '
                for (k, v) in zip(traj_err_key, traj_err):
                    line += '(%s): %.4f ' % (k, np.mean(v))
                print(line)
                print(line, file=file, flush=True)
            else:
                l2_dict[motion_loader_name] = None
                trajectory_score_dict[motion_loader_name] = None
        
        if use_smpl:
            motion_loader.dataset.output_format = 'hml_vec'
        motion_loader.dataset.output_gt_joints = False

    return l2_dict, skating_ratio_dict, trajectory_score_dict, simulation_error, simulation_error_max, simulation_sr_02, simulation_sr_05, power, feet_height_dict, jerk_dict


def evaluate_control_sample(motions, hint, mask_hint, res, rep_i=0, global_sample_idx=0, m_lens=None, use_smpl=False):
    """
    Evaluate control metrics for generated samples.
    
    Args:
        motions: (B, 22, 3, T) - joint positions from rot2xyz
        hint: (B, T, 22, 3) - control hints
        mask_hint: (B, T, 22, 1) - mask for control hints
    """
    traj_err_key  = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    batch_size = motions.shape[0]
    motion_names = ['sample_sample_%d_rep_%d' % (i+global_sample_idx, rep_i) for i in range(batch_size)]

    if use_smpl:
        motions = motions.float()
        # numpy array from (B, 22, 3, T) to (B, T, 22, 3)
        motions = motions.permute(0, 3, 1, 2)
        x, y, z = T(motions)
        motions = T(np.stack((x, z, -y), axis=0)) # change to y-up
        motions = torch.from_numpy(motions)
        # from (B, T, 22, 3) to (B, 22, 3, T)
        motions = motions.permute(0, 2, 3, 1)

        # convert hint to y-up
        x, y, z = T(hint)
        hint = T(np.stack((x, z, -y), axis=0)) # change to y-up
        hint = torch.from_numpy(hint)


    with torch.no_grad():
        # motions shape: (B, 22, 3, T)
        # calculate_skating_ratio expects: (B, 22, 3, T) ✓
        skate_ratio, skate_vel = calculate_skating_ratio(motions.clone().cpu(), m_lens)  # [batch_size]
        feet_height = calculate_feet_height(motions.clone().cpu(), m_lens)  # [batch_size]
        jerk = calculate_jerk(motions.clone().cpu(), m_lens)  # [batch_size]

        skate_ratio_list = skate_ratio.tolist()
        feet_height_list = feet_height.tolist()
        jerk_list = jerk.tolist()
        
        for idx in range(batch_size):
            res['skate_ratio'][motion_names[idx]] = skate_ratio_list[idx]
            res['feet_height'][motion_names[idx]] = feet_height_list[idx]
            res['jerk'][motion_names[idx]] = jerk_list[idx]

        # control l2 error
        # Need to convert motions from (B, 22, 3, T) to (B, T, 22, 3) for control_l2
        motions_for_control = motions.permute(0, 3, 1, 2)  # (B, T, 22, 3)
        
        for idx, (motion, h, mask) in enumerate(zip(motions_for_control, hint, mask_hint)):
            # motion: (T, 22, 3)
            # h: (T, 22, 3)
            # mask: (T, 22, 1)
            # control_l2 expects: (b, seq, 22, 3)
            control_error = control_l2(motion.unsqueeze(0).cpu().numpy(), h.unsqueeze(0).cpu().numpy(), mask.unsqueeze(0).cpu().numpy())
            mean_error = control_error.sum() / mask.sum()

            res['mean_error'][motion_names[idx]] = mean_error.item()

            control_error = control_error.reshape(-1)
            mask = mask.reshape(-1)
            err_np = calculate_trajectory_error(control_error, mean_error, mask)

            for i, key in enumerate(traj_err_key):
                res[key][motion_names[idx]] = err_np[i]
    
    return res


def evaluate_tmr_sample(tmr_forward, motions, texts, m_lens, 
                       gt_motions=None, rep_i=0, global_sample_idx=0):
    """
    Evaluate TMR metrics for generated samples.
    
    Args:
        tmr_forward: TMR model forward function
        motions: (B, T, 263) - generated motion features in normalized space
        texts: list of text descriptions
        m_lens: list of motion lengths
        gt_motions: (B, T, 263) - ground truth motion features (optional)
        rep_i: repetition index
        global_sample_idx: global sample index
    
    Returns:
        res: dict containing TMR scores for each sample
    """
    batch_size = motions.shape[0]
    motion_names = ['sample_sample_%d_rep_%d' % (i+global_sample_idx, rep_i) for i in range(batch_size)]
    
    res = {
        'm2t_score': {},
    }
    
    if gt_motions is not None:
        res['m2m_score'] = {}
    
    with torch.no_grad():
        # Denormalize motions
        motions = motions.float().cpu().numpy()
        
        # Convert to guofeats format (B, T, 263)
        motions_guofeats = [motions[i][:m_lens[i]] for i in range(batch_size)]
        
        # Process ground truth if provided
        if gt_motions is not None:
            gt_motions = gt_motions.float().cpu().numpy()
            motions_guofeats_gt = [gt_motions[i][:m_lens[i]] for i in range(batch_size)]
        else:
            motions_guofeats_gt = None
        
        # Calculate TMR metrics for each sample individually
        for idx in range(batch_size):
            text_list = [texts[idx]]
            motion_pred_list = [motions_guofeats[idx]]
            motion_gt_list = [motions_guofeats_gt[idx]] if motions_guofeats_gt is not None else None
            
            # Calculate TMR metrics
            metrics = calculate_tmr_metrics(
                tmr_forward, 
                texts_gt=text_list, 
                motions_guofeats_gt=motion_gt_list,
                motions_guofeats_pred=motion_pred_list, 
                calculate_retrieval=False
            )
            
            res['m2t_score'][motion_names[idx]] = metrics['m2t_score']
            
            # m2m_score will only be present in metrics if gt was provided
            if 'm2m_score' in metrics:
                res['m2m_score'][motion_names[idx]] = metrics['m2m_score']
    
    return res


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            if len(batch) == 7:
                _, _, _, sent_lens, motions, m_lens, _ = batch
            else:
                _, _, _, sent_lens, motions, m_lens, _, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval



def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, 
               mm_num_times, run_mm=False, train_platform=None, training_args=None, training_step=0, 
               visualize=False, use_smpl=False, cond_mode=None, 
               sim_gpu=None, save_figure=False, tmr_forward=None):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Control_l2': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'Trajectory Error': OrderedDict({}),
                                   'Simulation Error': OrderedDict({}) if use_smpl else None,
                                   'Simulation Error Max': OrderedDict({}) if use_smpl else None,
                                   'Power': OrderedDict({}) if use_smpl else None,
                                   'Simulation SR 0.2': OrderedDict({}) if use_smpl else None,
                                   'Simulation SR 0.5': OrderedDict({}) if use_smpl else None,
                                   'Feet Height': OrderedDict({}),
                                   'Jerk': OrderedDict({}),
                                   'M2M score': OrderedDict({}) if tmr_forward is not None else None,
                                   'M2T score': OrderedDict({}) if tmr_forward is not None else None,
                                   'TMR R1 precision': OrderedDict({}) if tmr_forward is not None else None,
                                   'TMR R3 precision': OrderedDict({}) if tmr_forward is not None else None,
                                   })

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            seed_dict = {
                'vald': replication_times * replication + 0,
                'vald_best': replication_times * replication + 1,
                'vald_worst': replication_times * replication + 2,
                'vald_gt': replication_times * replication + 0, # same as best to ensure reproducibility
                'vald_best_gt': replication_times * replication + 1, # same as best to ensure reproducibility
                'vald_sim': replication_times * replication + 0,
            }
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                # Set both torch and numpy seeds for reproducibility
                if train_platform is not None:
                    # random.seed(seed_dict[motion_loader_name])
                    torch.manual_seed(seed_dict[motion_loader_name])
                    # torch.cuda.manual_seed_all(seed_dict[motion_loader_name])
                    np.random.seed(seed_dict[motion_loader_name])

                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            

            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            # TMR metrics
            if tmr_forward is not None:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                m2m_score_dict, m2t_score_dict, tmr_r1_dict, tmr_r3_dict = evaluate_tmr_metrics(tmr_forward, motion_loaders, f)

                for key, item in m2m_score_dict.items():
                    if key not in all_metrics['M2M score']:
                        all_metrics['M2M score'][key] = [item]
                    else:
                        all_metrics['M2M score'][key] += [item]

                for key, item in m2t_score_dict.items():
                    if key not in all_metrics['M2T score']:
                        all_metrics['M2T score'][key] = [item]
                    else:
                        all_metrics['M2T score'][key] += [item]

                for key, item in tmr_r1_dict.items():
                    if key not in all_metrics['TMR R1 precision']:
                        all_metrics['TMR R1 precision'][key] = [item]
                    else:
                        all_metrics['TMR R1 precision'][key] += [item]

                for key, item in tmr_r3_dict.items():
                    if key not in all_metrics['TMR R3 precision']:
                        all_metrics['TMR R3 precision'][key] = [item]
                    else:
                        all_metrics['TMR R3 precision'][key] += [item]

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            control_l2_dict, skating_ratio_dict, trajectory_score_dict, simulation_error, simulation_error_max, simulation_sr_02, simulation_sr_05, power, feet_height_dict, jerk_dict = evaluate_control(motion_loaders, 
                                                            f, train_platform=train_platform if replication==0 else None,
                                                            training_args=training_args, training_step=training_step,
                                                            visualize=visualize, use_smpl=use_smpl, cond_mode=cond_mode,
                                                            log_file=log_file, sim_gpu=sim_gpu, save_figure=save_figure)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text'):
                for key, item in control_l2_dict.items():
                    if key not in all_metrics['Control_l2']:
                        all_metrics['Control_l2'][key] = [item]
                    else:
                        all_metrics['Control_l2'][key] += [item]

            for key, item in skating_ratio_dict.items():
                if key not in all_metrics['Skating Ratio']:
                    all_metrics['Skating Ratio'][key] = [item]
                else:
                    all_metrics['Skating Ratio'][key] += [item]

            for key, item in feet_height_dict.items():
                if key not in all_metrics['Feet Height']:
                    all_metrics['Feet Height'][key] = [item]
                else:
                    all_metrics['Feet Height'][key] += [item]

            for key, item in jerk_dict.items():
                if key not in all_metrics['Jerk']:
                    all_metrics['Jerk'][key] = [item]
                else:
                    all_metrics['Jerk'][key] += [item]

            if use_smpl:
                for key, item in simulation_error.items():
                    if key not in all_metrics['Simulation Error']:
                        all_metrics['Simulation Error'][key] = [item]
                    else:
                        all_metrics['Simulation Error'][key] += [item]

                for key, item in simulation_error_max.items():
                    if key not in all_metrics['Simulation Error Max']:
                        all_metrics['Simulation Error Max'][key] = [item]
                    else:
                        all_metrics['Simulation Error Max'][key] += [item]

                for key, item in simulation_sr_02.items():
                    if key not in all_metrics['Simulation SR 0.2']:
                        all_metrics['Simulation SR 0.2'][key] = [item]
                    else:
                        all_metrics['Simulation SR 0.2'][key] += [item]

                for key, item in simulation_sr_05.items():
                    if key not in all_metrics['Simulation SR 0.5']:
                        all_metrics['Simulation SR 0.5'][key] = [item]
                    else:
                        all_metrics['Simulation SR 0.5'][key] += [item]

                for key, item in power.items():
                    if key not in all_metrics['Power']:
                        all_metrics['Power'][key] = [item]
                    else:
                        all_metrics['Power'][key] += [item]
            
            if (training_args is not None and training_args.cond_mode != 'only_text') or (cond_mode is not None and cond_mode != 'only_text'):
                for key, item in trajectory_score_dict.items():
                    if key not in all_metrics['Trajectory Error']:
                        all_metrics['Trajectory Error'][key] = [item]
                    else:
                        all_metrics['Trajectory Error'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            # Skip printing if no data collected
            if not metric_dict or metric_dict is None:
                continue
                
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif metric_name == 'Trajectory Error':
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)): # zip(traj_err_key, mean):
                        line += '(%s): Mean: %.4f CInt: %.4f; ' % (traj_err_key[i], mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(
        os.path.dirname(args.model_path),
        'eval_humanml_{}'.format(niter)
    )

    # log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += f'_{args.sampler}'
    log_file += f'_masktype{args.mask_type}'
    log_file += f'_joint{args.control_joint}'
    log_file += f'_density{args.density}'
    log_file += f'_cond{args.cond_mode}'
    log_file += f'_part{args.data_part}'
    if args.omomo:
        log_file += f'_omomo'
    else:
        log_file += f'_humanml3d'

    if args.eval_mode == 'maskedmimic':
        log_file += f'_{args.maskedmimic_init_mode}'
    log_file += f'_{time_str}'

    # log_file += '_cross_random'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'omnicontrol' or args.eval_mode == 'maskedmimic':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 1  # about 3 Hrs
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, 
                                   split=split, hml_mode='gt', control_joint=args.control_joint, 
                                   density=args.density, use_omomo=args.omomo, use_dpo=False, dpo_data_root=args.dpo_data_root,
                                   mask_type=args.mask_type, use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part)
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, 
                                    split=split, hml_mode='eval', control_joint=args.control_joint, 
                                    density=args.density, use_omomo=args.omomo, use_dpo=False, dpo_data_root=args.dpo_data_root,
                                    mask_type=args.mask_type, use_smpl=args.use_smpl, cond_mode=args.cond_mode, data_part=args.data_part)
    num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    log_folder = log_file.replace('.log','')
    os.makedirs(log_folder, exist_ok=True)

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(
            model, diffusion, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, 
            num_samples_limit, args.guidance_param, use_smpl=args.use_smpl, joints_value_from=args.joints_value_from, 
            eval_after_simulation=False if args.eval_mode == 'omnicontrol' else True, sim_gpu=args.sim_gpu, output_path=log_folder, eval_mode=args.eval_mode,
            maskedmimic_init_mode=args.maskedmimic_init_mode
        ),
    }

    if args.use_smpl and args.eval_mode == 'omnicontrol':
        eval_motion_loaders = {}
        eval_motion_loaders['vald_sim'] = lambda: get_mdm_loader(
            model, diffusion, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, 
            num_samples_limit, args.guidance_param, use_smpl=args.use_smpl, joints_value_from=args.joints_value_from,
            eval_after_simulation=args.eval_after_simulation, sim_gpu=args.sim_gpu, output_path=log_folder, eval_mode=args.eval_mode,
            maskedmimic_init_mode=args.maskedmimic_init_mode
        )
    else:
        # as maskedmimic output is already after simulation, no need to eval sim again
        pass

    device = 'cuda'  # or 'cpu'
    # Load TMR model
    tmr_forward = load_tmr_model_easy(device)

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, 
               run_mm=run_mm, use_smpl=args.use_smpl, cond_mode=args.cond_mode, 
               sim_gpu=args.sim_gpu, save_figure=args.save_figure, tmr_forward=tmr_forward)
