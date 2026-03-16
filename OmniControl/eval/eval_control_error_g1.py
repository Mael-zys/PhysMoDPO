import numpy as np
import os
import sys
import torch
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fix_fps import interpolate_fps_joints
from data_loaders.humanml.utils.metrics import (
    calculate_jerk,
    calculate_feet_height,
    calculate_skating_ratio,
)
from utils.calculate_TMR_score.tmr_paths import ensure_tmr_on_path, get_tmr_path

ensure_tmr_on_path()
from TMR.src.process_g1_data import process_joints
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy
from utils.calculate_TMR_score.tmr_eval_wrapper import calculate_tmr_metrics

control_idx_dict = {
    0: 0,
    10: 7,
    11: 13,
    15: 1,
    20: 24,
    21: 32,
}


def parse_args():
    parser = argparse.ArgumentParser(description='output root dir for evaluation')
    parser.add_argument('--data_root_dir', type=str, 
                        default='save/cross_GT_hint',
                        help='Root directory for DPO inference results')
    parser.add_argument('--data_gt_dir', type=str, 
                        default='save/cross_GT_hint',
                        help='Root directory for DPO inference results')
    parser.add_argument('--tmr_mode', type=str, default='guofeats', 
                        choices=['translation_local', 'guofeats', 'velocity_local'], 
                        help='Mode for processing joints for TMR evaluation')
    parser.add_argument('--eval_mode', type=str, default='all', 
                        choices=['all', 'single'], 
                        help='calculate metrics for all motions and average, or calculate metrics for a single motion and save to a json file')
    return parser.parse_args()


def _parse_metric_from_log(log_path, metric_key):
    if not os.path.exists(log_path):
        return None

    float_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    latest_value = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if metric_key not in line:
                continue
            suffix = line.split(metric_key, 1)[1]
            match = re.search(float_pattern, suffix)
            if match:
                try:
                    # Keep scanning and use the latest occurrence in the log.
                    latest_value = float(match.group(0))
                except ValueError:
                    continue
    return latest_value


def _sig4(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        v = float(value)
        if not np.isfinite(v):
            return v
        if v == 0.0:
            return 0.0
        return float(f"{v:.4g}")
    return value


def _sig4_recursive(data):
    if isinstance(data, dict):
        return {k: _sig4_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sig4_recursive(v) for v in data]
    return _sig4(data)


def _find_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_scalar_dict_json(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                out[str(k)] = v
        return out

    if isinstance(data, list):
        out = {}
        for i, v in enumerate(data):
            try:
                out[str(i)] = float(v)
            except (TypeError, ValueError):
                out[str(i)] = v
        return out

    return {"value": data}


def _unique_existing_dirs(candidates):
    out = []
    seen = set()
    for path in candidates:
        if not os.path.isdir(path):
            continue
        real_path = os.path.realpath(path)
        if real_path in seen:
            continue
        seen.add(real_path)
        out.append(path)
    return out


def _dedup_preserve_order(paths):
    out = []
    seen = set()
    for path in paths:
        real_path = os.path.realpath(path)
        if real_path in seen:
            continue
        seen.add(real_path)
        out.append(path)
    return out


def _safe_getmtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _next_available_path(path):
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{path}.{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _archive_files(paths, protected_paths=None, suffix=".merged_shard"):
    protected_real_paths = set()
    if protected_paths:
        protected_real_paths = {os.path.realpath(path) for path in protected_paths}

    archived = []
    for path in _dedup_preserve_order(paths):
        if not os.path.exists(path):
            continue

        if os.path.realpath(path) in protected_real_paths:
            continue

        archive_path = _next_available_path(f"{path}{suffix}")
        try:
            os.rename(path, archive_path)
            archived.append((path, archive_path))
        except OSError as exc:
            print(f"Warning: failed to archive shard file {path}: {exc}")
    return archived


def _select_latest_epoch_paths(search_dirs, ranked_regex, single_regex):
    ranked_re = re.compile(ranked_regex)
    single_re = re.compile(single_regex)
    ranked_by_epoch = {}
    single_by_epoch = {}

    for search_dir in search_dirs:
        for file_name in os.listdir(search_dir):
            ranked_match = ranked_re.fullmatch(file_name)
            if ranked_match:
                epoch = int(ranked_match.group("epoch"))
                rank = int(ranked_match.group("rank"))
                ranked_by_epoch.setdefault(epoch, []).append((rank, os.path.join(search_dir, file_name)))
                continue

            single_match = single_re.fullmatch(file_name)
            if single_match:
                epoch = int(single_match.group("epoch"))
                single_by_epoch.setdefault(epoch, []).append(os.path.join(search_dir, file_name))

    all_epochs = sorted(set(ranked_by_epoch.keys()).union(single_by_epoch.keys()))
    if not all_epochs:
        return [], None, None

    epoch = all_epochs[-1]
    ranked_files = sorted(ranked_by_epoch.get(epoch, []), key=lambda x: x[0])
    ranked_paths = _dedup_preserve_order([path for _, path in ranked_files])
    single_paths = _dedup_preserve_order(sorted(single_by_epoch.get(epoch, [])))

    if single_paths:
        # Reuse cached merged file unless ranked shards are newer.
        single_path = single_paths[0]
        if not ranked_paths:
            return [single_path], epoch, "single"
        single_mtime = _safe_getmtime(single_path)
        newest_ranked_mtime = max(_safe_getmtime(path) for path in ranked_paths)
        if single_mtime >= newest_ranked_mtime:
            return [single_path], epoch, "single"

    if ranked_paths:
        return ranked_paths, epoch, "ranked"

    return [], None, None


def _find_pred_motion_pt_paths(data_root_dir):
    search_dirs = _unique_existing_dirs([
        os.path.join(data_root_dir, "merge_all_g1_output", "results"),
        os.path.join(data_root_dir, "results"),
        data_root_dir,
    ])
    paths, epoch, source_type = _select_latest_epoch_paths(
        search_dirs=search_dirs,
        ranked_regex=r"predicted_motion_lib_epoch_(?P<epoch>\d+)_rank_(?P<rank>\d+)\.pt",
        single_regex=r"predicted_motion_lib_epoch_(?P<epoch>\d+)\.pt",
    )
    merge_output_path = None
    shard_paths = []
    if source_type == "ranked" and paths and epoch is not None:
        merge_output_path = os.path.join(
            os.path.dirname(paths[0]), f"predicted_motion_lib_epoch_{epoch}.pt"
        )
        shard_paths = list(paths)
    if paths:
        print(f"Using predicted motion pt files (epoch={epoch}): {paths}")
    return paths, merge_output_path, shard_paths


def _find_gt_motion_pt_paths(data_gt_dir):
    search_dirs = _unique_existing_dirs([
        os.path.join(data_gt_dir, "merge_all_g1_output"),
        data_gt_dir,
    ])
    ranked_files = []
    single_files = []

    rank_re_list = [
        re.compile(r"retargeted_g1_rank_(\d+)\.pt"),
        re.compile(r"retargeted_g1_(\d+)\.pt"),
    ]

    for search_dir in search_dirs:
        for file_name in os.listdir(search_dir):
            if file_name == "retargeted_g1.pt":
                single_files.append(os.path.join(search_dir, file_name))
                continue

            rank = None
            for rank_re in rank_re_list:
                rank_match = rank_re.fullmatch(file_name)
                if rank_match:
                    rank = int(rank_match.group(1))
                    break
            if rank is not None:
                ranked_files.append((rank, os.path.join(search_dir, file_name)))

    ranked_files = sorted(ranked_files, key=lambda x: x[0])
    ranked_paths = _dedup_preserve_order([path for _, path in ranked_files])
    single_paths = _dedup_preserve_order(sorted(single_files))

    if single_paths:
        single_path = single_paths[0]
        if not ranked_paths:
            print(f"Using GT motion pt file (single): {[single_path]}")
            return [single_path], None, []
        single_mtime = _safe_getmtime(single_path)
        newest_ranked_mtime = max(_safe_getmtime(path) for path in ranked_paths)
        if single_mtime >= newest_ranked_mtime:
            print(f"Using GT motion pt file (single, cached): {[single_path]}")
            return [single_path], None, []

    if ranked_paths:
        merge_output_path = os.path.join(os.path.dirname(ranked_paths[0]), "retargeted_g1.pt")
        print(f"Using GT motion pt files (ranked): {ranked_paths}")
        return ranked_paths, merge_output_path, list(ranked_paths)

    return [], None, []


def _load_motion_lib_pt_files(pt_paths, label, merge_output_path=None, archive_source_paths=None):
    if not pt_paths:
        raise FileNotFoundError(f"No {label} motion pt files found.")

    if len(pt_paths) == 1:
        return torch.load(pt_paths[0], map_location='cpu')

    merged_gts = []
    merged_motion_files = []
    merged_motion_num_frames = []
    merged_length_starts = []
    total_frames = 0
    required_keys = ["gts", "motion_files", "motion_num_frames", "length_starts"]

    for pt_path in pt_paths:
        current = torch.load(pt_path, map_location='cpu')
        for key in required_keys:
            if key not in current:
                raise KeyError(f"Missing key `{key}` in {pt_path}")

        current_gts = current["gts"]
        current_motion_files = list(current["motion_files"])
        current_motion_num_frames = [int(v) for v in current["motion_num_frames"]]
        current_length_starts = [int(v) for v in current["length_starts"]]

        if not (
            len(current_motion_files)
            == len(current_motion_num_frames)
            == len(current_length_starts)
        ):
            raise ValueError(
                f"Inconsistent lengths in {pt_path}: "
                f"motion_files={len(current_motion_files)}, "
                f"motion_num_frames={len(current_motion_num_frames)}, "
                f"length_starts={len(current_length_starts)}"
            )

        merged_gts.append(current_gts)
        merged_motion_files.extend(current_motion_files)
        merged_motion_num_frames.extend(current_motion_num_frames)
        merged_length_starts.extend([start + total_frames for start in current_length_starts])
        total_frames += int(current_gts.shape[0])

    merged = {
        "gts": torch.cat(merged_gts, dim=0),
        "motion_files": merged_motion_files,
        "motion_num_frames": merged_motion_num_frames,
        "length_starts": merged_length_starts,
    }
    print(
        f"Merged {len(pt_paths)} {label} pt files, "
        f"motions={len(merged_motion_files)}, total_frames={total_frames}"
    )

    if merge_output_path:
        os.makedirs(os.path.dirname(merge_output_path), exist_ok=True)
        tmp_path = _next_available_path(f"{merge_output_path}.tmp")
        try:
            torch.save(merged, tmp_path)
            os.replace(tmp_path, merge_output_path)
            print(f"Saved merged {label} pt file to {merge_output_path}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        archived = _archive_files(
            archive_source_paths or pt_paths,
            protected_paths=[merge_output_path],
            suffix=".merged_shard",
        )
        if archived:
            print(f"Archived {len(archived)} {label} shard files after merge")

    return merged


def _find_ranked_metric_json_paths(data_root_dir, metric_prefix):
    search_dirs = _unique_existing_dirs([
        os.path.join(data_root_dir, "merge_all_g1_output", "results"),
        os.path.join(data_root_dir, "results"),
        data_root_dir,
    ])
    file_re = re.compile(rf"{re.escape(metric_prefix)}_(\d+)\.json")
    ranked_files = []
    single_files = []
    for search_dir in search_dirs:
        for file_name in os.listdir(search_dir):
            if file_name == f"{metric_prefix}.json":
                single_files.append(os.path.join(search_dir, file_name))
                continue
            match = file_re.fullmatch(file_name)
            if not match:
                continue
            ranked_files.append((int(match.group(1)), os.path.join(search_dir, file_name)))
    ranked_paths = _dedup_preserve_order([path for _, path in sorted(ranked_files, key=lambda x: x[0])])
    single_paths = _dedup_preserve_order(sorted(single_files))

    if single_paths:
        single_path = single_paths[0]
        if not ranked_paths:
            return [single_path], None, []
        single_mtime = _safe_getmtime(single_path)
        newest_ranked_mtime = max(_safe_getmtime(path) for path in ranked_paths)
        if single_mtime >= newest_ranked_mtime:
            return [single_path], None, []

    if ranked_paths:
        merge_output_path = os.path.join(os.path.dirname(ranked_paths[0]), f"{metric_prefix}.json")
        return ranked_paths, merge_output_path, list(ranked_paths)

    return [], None, []


def _load_and_merge_scalar_dict_jsons(json_paths, merge_output_path=None, archive_source_paths=None, label="metric"):
    if not json_paths:
        return None

    merged = {}
    for json_path in json_paths:
        current = _load_scalar_dict_json(json_path)
        if current is None:
            continue
        overlap_keys = set(merged.keys()).intersection(current.keys())
        if overlap_keys:
            print(
                f"Warning: {len(overlap_keys)} duplicated keys while merging {json_path}; "
                "later file will overwrite previous values"
            )
        merged.update(current)

    if not merged:
        return None

    if merge_output_path:
        os.makedirs(os.path.dirname(merge_output_path), exist_ok=True)
        tmp_path = _next_available_path(f"{merge_output_path}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, merge_output_path)
            print(f"Saved merged {label} json to {merge_output_path}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        archived = _archive_files(
            archive_source_paths or json_paths,
            protected_paths=[merge_output_path],
            suffix=".merged_shard",
        )
        if archived:
            print(f"Archived {len(archived)} {label} json shard files after merge")

    return merged


def _find_eval_log_paths(data_root_dir):
    search_dirs = _unique_existing_dirs([
        os.path.join(data_root_dir, "merge_all_g1_output", "results"),
        os.path.join(data_root_dir, "results"),
        data_root_dir,
    ])
    paths, epoch, _ = _select_latest_epoch_paths(
        search_dirs=search_dirs,
        ranked_regex=r"evaluation_log_epoch_(?P<epoch>\d+)_rank_(?P<rank>\d+)\.txt",
        single_regex=r"evaluation_log_epoch_(?P<epoch>\d+)\.txt",
    )
    if paths:
        print(f"Using evaluation log files (epoch={epoch}): {paths}")
    return paths


def _parse_metric_from_logs(log_paths, metric_key):
    values = []
    for log_path in log_paths:
        metric_value = _parse_metric_from_log(log_path, metric_key)
        if metric_value is not None:
            values.append(float(metric_value))
    if not values:
        return None
    return float(sum(values) / len(values))


def evaluate_control_error_g1(data_root_dir, data_gt_dir, tmr_mode='guofeats', eval_mode='all'):
    is_single = eval_mode == 'single'
    is_all = eval_mode == 'all'

    pred_paths, pred_merge_output_path, pred_shard_paths = _find_pred_motion_pt_paths(data_root_dir)
    gt_paths, gt_merge_output_path, gt_shard_paths = _find_gt_motion_pt_paths(data_gt_dir)

    # fps 50
    pred = _load_motion_lib_pt_files(
        pred_paths,
        label="predicted",
        merge_output_path=pred_merge_output_path,
        archive_source_paths=pred_shard_paths,
    )
    # fps 30
    gt = _load_motion_lib_pt_files(
        gt_paths,
        label="GT",
        merge_output_path=gt_merge_output_path,
        archive_source_paths=gt_shard_paths,
    )

    joints_pred = pred['gts']  # (T, J, 3)
    joints_gt = gt['gts']  # (T, J, 3)
    motion_files = pred['motion_files']  # list of motion file names
    gt_motion_files = gt['motion_files']  # list of motion file names for GT, should correspond to pred_motion_files
    motion_num_frames_pred = pred['motion_num_frames']  # list of frame counts for each motion
    motion_num_frames_gt = gt['motion_num_frames']  # list of frame counts for each motion
    length_starts_pred = pred['length_starts']  # list of start indices for each motion in the concatenated joints
    length_starts_gt = gt['length_starts']  # list of start indices for each motion in the concatenated joints

    # find index corresponding GT motion file for each predicted motion file
    pred_to_gt_file_map = {}
    for idx, pred_file in enumerate(motion_files):
        pred_base_name = os.path.basename(pred_file)
        if len(pred_base_name.split('_')) > 7:
            pred_name_part = pred_base_name.split('_')
            pred_base_name = '_'.join(pred_name_part[:5]+pred_name_part[6:])
        for jdx, gt_file in enumerate(gt_motion_files):
            gt_base_name = os.path.basename(gt_file)
            if pred_base_name == gt_base_name:
                pred_to_gt_file_map[idx] = jdx
                break
    
    tmr_forward = load_tmr_model_easy(
        device='cuda',
        run_dir=get_tmr_path("outputs", f"tmr_g1_{tmr_mode}"),
        ckpt_name="latest-epoch=65",
    )

    mean_error_sum = 0.0
    traj_fail_02_sum = 0.0
    traj_fail_05_sum = 0.0
    jerk_sum = 0.0
    skating_ratio_sum = 0.0
    feet_height_sum = 0.0
    valid_motion_count = 0

    per_sample_metrics = {
        "mean_error": {},
        "traj_fail_02": {},
        "traj_fail_05": {},
        "jerk": {},
        "skate_ratio": {},
        "feet_height": {},
        "power": {},
        "m2t_score": {},
        "m2m_score": {},
    }
    if is_single:
        per_sample_metrics["dp_mpjpe"] = {}
        per_sample_metrics["dp_mpjpe_max"] = {}

    text_list = [] if is_all else None
    joints_feats_pred_list = [] if is_all else None
    joints_feats_gt_list = [] if is_all else None
    sample_names_for_tmr = [] if is_all else None
    # FID, matching score, R-precision, M2T score, M2M score can be calculated here as well using the joints_pred_interp and joints_gt_motion
    for idx in tqdm(range(len(motion_files))):
        if idx not in pred_to_gt_file_map:
            print(f"Warning: No corresponding GT file found for predicted motion file {motion_files[idx]}")
            continue
        
        gt_motion_file = gt_motion_files[pred_to_gt_file_map[idx]]
        start_pred = length_starts_pred[idx]
        end_pred = start_pred + motion_num_frames_pred[idx]
        start_gt = length_starts_gt[pred_to_gt_file_map[idx]]
        end_gt = start_gt + motion_num_frames_gt[pred_to_gt_file_map[idx]]

        joints_pred_motion = joints_pred[start_pred:end_pred].clone()
        joints_gt_motion = joints_gt[start_gt:end_gt].clone()
        
        joints_pred_motion[:,:,:2] = joints_pred[start_pred:end_pred,:,:2] - joints_pred[start_pred:start_pred+1,0:1,:2] # (T_pred, J, 3)
        joints_gt_motion[:,:,:2] = joints_gt[start_gt:end_gt,:,:2] - joints_gt[start_gt:start_gt+1,0:1,:2]  # (T_gt, J, 3)
        
        new_fps = 30
        if '_zero_pose_warmup' in data_root_dir:
            joints_pred_motion = joints_pred_motion[50:]
        if 'maskedmimic_output' in data_root_dir:
            new_fps = 20
        # Interpolate predicted joints to match the number of frames in GT
        joints_pred_interp = interpolate_fps_joints(joints_pred_motion, old_fps=50, new_fps=new_fps)  # (T_interp, J, 3)
        joints_pred_interp[:,:,:2] = joints_pred_interp[:,:,:2] - joints_pred_interp[0:1, 0:1, :2]  # Re-center after interpolation

        file_name = os.path.basename(gt_motion_file).split('_')[2:-2]
        if len(file_name) >= 3:
            hint_file_path = os.path.join(data_gt_dir, f"{file_name[0]}_{file_name[1]}",'amass_format', 'ik', f"ik_{file_name[2]}.npz")
        else:
            hint_file_path = os.path.join(data_gt_dir, 'amass_format', 'ik', f"ik_{file_name[0]}_{file_name[1]}.npz")
        hint = np.load(hint_file_path)
        
        text = str(hint['text'])

        if 'hint' not in hint:
            ori_batch_idx = int(file_name[0]) // 32
            ori_sample_idx = int(file_name[0]) % 32
            # results_batch_0000.npy
            hint_file_path = os.path.join(data_gt_dir, 'omnicontrol_output', f"results_batch_{ori_batch_idx:04d}.npy")
            hint = np.load(hint_file_path, allow_pickle=True).item()
            hint_joints = hint['hint'][ori_sample_idx]
            hint_joints = hint_joints.reshape(hint_joints.shape[0], -1, 3)  # (T_hint, J, 3)
        else:
            hint_joints = hint['hint'].reshape(hint['hint'].shape[0],-1, 3)  # (T_hint, J, 3)

        hint_mask = np.any(hint_joints != 0, axis=-1)  # (T_hint, J_hint)

        length = min(joints_pred_interp.shape[0], joints_gt_motion.shape[0], hint_mask.shape[0])
        num_eval_joints = joints_pred_interp.shape[1]
        g1_hint_mask = np.zeros((length, num_eval_joints), dtype=bool)
        for control_idx, g1_joint_idx in control_idx_dict.items():
            if control_idx >= hint_mask.shape[1] or g1_joint_idx >= num_eval_joints:
                continue
            g1_hint_mask[:, g1_joint_idx] = hint_mask[:length, control_idx]

        if g1_hint_mask.sum() == 0 or length == 0:
            print(f"Warning: Invalid hint mask or length for {gt_motion_file}, skipping")
            continue

        pred_np = joints_pred_interp[:length].clone().cpu().numpy()
        gt_np = joints_gt_motion[:length].numpy()
        joint_error = np.linalg.norm(pred_np - gt_np, axis=-1)  # (T, J)
        control_error = joint_error * g1_hint_mask
        valid_mask_count = g1_hint_mask.sum()

        mean_error = control_error.sum() / valid_mask_count

        traj_fail_02 = 1.0 - ((joint_error <= 0.2) | (~g1_hint_mask)).all()
        traj_fail_05 = 1.0 - ((joint_error <= 0.5) | (~g1_hint_mask)).all()
        motion_name_split = os.path.basename(gt_motion_file).split('_')[2:-2]
        if len(motion_name_split) >= 3:
            motion_name = f'sample_sample_{motion_name_split[0]}_{motion_name_split[2]}_rep_{motion_name_split[1]}'
        else:
            motion_name = f'sample_sample_{motion_name_split[0]}_rep_{motion_name_split[-1]}'

        mean_error_sum += mean_error
        traj_fail_02_sum += traj_fail_02
        traj_fail_05_sum += traj_fail_05
        valid_motion_count += 1

        joints_pred_interp_for_eval = joints_pred_interp[:length].clone().unsqueeze(0).permute(0,2,3,1).cpu()

        jerk_value = float(calculate_jerk(joints_pred_interp_for_eval.clone(), [length], fps=30))
        jerk_sum += jerk_value
        skating_ratio, _ = calculate_skating_ratio(joints_pred_interp_for_eval.clone(), [length], fps=30, feet_idx=[7, 13], height_index=2)    
        skating_ratio_value = float(skating_ratio)
        skating_ratio_sum += skating_ratio_value
        feet_height_value = float(calculate_feet_height(joints_pred_interp_for_eval.clone(), [length], feet_idx=[7, 13], height_index=2))
        feet_height_sum += feet_height_value

        per_sample_metrics["mean_error"][motion_name] = float(mean_error)
        per_sample_metrics["traj_fail_02"][motion_name] = float(traj_fail_02)
        per_sample_metrics["traj_fail_05"][motion_name] = float(traj_fail_05)
        per_sample_metrics["jerk"][motion_name] = jerk_value
        per_sample_metrics["skate_ratio"][motion_name] = skating_ratio_value
        per_sample_metrics["feet_height"][motion_name] = feet_height_value
        per_sample_metrics["power"][motion_name] = 0.0

        joints_pred_for_tmr = process_joints(joints_pred_interp.cpu().numpy(), mode=tmr_mode, align_init_yaw=True, heading_target_axis='y',face_joint_idx=[8, 2, 25, 17])
        joints_gt_for_tmr = process_joints(joints_gt_motion.cpu().numpy(), mode=tmr_mode, align_init_yaw=True, heading_target_axis='y', face_joint_idx=[8, 2, 25, 17])

        # std of dimension 43 and 44 is 0, so set them to 0 to avoid NaN in TMR evaluation
        joints_pred_for_tmr[:,43:45] = 0.0
        joints_gt_for_tmr[:,43:45] = 0.0
        joints_pred_feat = joints_pred_for_tmr.reshape(joints_pred_for_tmr.shape[0], -1)
        joints_gt_feat = joints_gt_for_tmr.reshape(joints_gt_for_tmr.shape[0], -1)

        if is_single:
            sample_tmr_metrics = calculate_tmr_metrics(
                tmr_forward=tmr_forward,
                texts_gt=[text],
                motions_guofeats_pred=[joints_pred_feat],
                motions_guofeats_gt=[joints_gt_feat],
                calculate_retrieval=False,
                calculate_fid=False,
            )
            per_sample_metrics["m2t_score"][motion_name] = float(sample_tmr_metrics.get("m2t_score", 0.0))
            per_sample_metrics["m2m_score"][motion_name] = float(sample_tmr_metrics.get("m2m_score", 0.0))
        else:
            text_list.append(text)
            joints_feats_pred_list.append(joints_pred_feat)
            joints_feats_gt_list.append(joints_gt_feat)
            sample_names_for_tmr.append(motion_name)

    if valid_motion_count == 0:
        raise RuntimeError("No valid motions found for evaluation.")

    num_motions = valid_motion_count
    mean_error_avg = mean_error_sum / num_motions
    traj_fail_02_avg = traj_fail_02_sum / num_motions
    traj_fail_05_avg = traj_fail_05_sum / num_motions

    jerk_avg = jerk_sum / num_motions
    skating_ratio_avg = skating_ratio_sum / num_motions
    feet_height_avg = feet_height_sum / num_motions

    tmr_metrics = {}
    if is_all:
        # Calculate TMR metrics with batch size 32, then average across batches
        tmr_batch_size = 32
        tmr_metrics_sum = {}
        tmr_batch_count = 0

        for batch_start in range(0, len(joints_feats_pred_list), tmr_batch_size):
            batch_end = min(batch_start + tmr_batch_size, len(joints_feats_pred_list))
            batch_joints_feats_pred = joints_feats_pred_list[batch_start:batch_end]
            batch_joints_feats_gt = joints_feats_gt_list[batch_start:batch_end]
            batch_texts = text_list[batch_start:batch_end]
            batch_sample_names = sample_names_for_tmr[batch_start:batch_end]

            if len(batch_joints_feats_pred) == 0:
                continue

            batch_metrics = calculate_tmr_metrics(
                tmr_forward=tmr_forward,
                texts_gt=batch_texts,
                motions_guofeats_pred=batch_joints_feats_pred,
                motions_guofeats_gt=batch_joints_feats_gt,
                calculate_retrieval=True,
                calculate_fid=True,
            )

            # Also save per-sample M2T / M2M in all mode.
            for sample_idx, sample_name in enumerate(batch_sample_names):
                sample_metric = calculate_tmr_metrics(
                    tmr_forward=tmr_forward,
                    texts_gt=[batch_texts[sample_idx]],
                    motions_guofeats_pred=[batch_joints_feats_pred[sample_idx]],
                    motions_guofeats_gt=[batch_joints_feats_gt[sample_idx]],
                    calculate_retrieval=False,
                    calculate_fid=False,
                )
                per_sample_metrics["m2t_score"][sample_name] = float(sample_metric.get("m2t_score", 0.0))
                per_sample_metrics["m2m_score"][sample_name] = float(sample_metric.get("m2m_score", 0.0))

            for metric_name, metric_value in batch_metrics.items():
                tmr_metrics_sum[metric_name] = tmr_metrics_sum.get(metric_name, 0.0) + float(metric_value)
            tmr_batch_count += 1

        if tmr_batch_count > 0:
            tmr_metrics = {metric_name: metric_sum / tmr_batch_count for metric_name, metric_sum in tmr_metrics_sum.items()}

        print("\nTMR Metrics:")
        for metric_name, metric_value in tmr_metrics.items():
            print(f"{metric_name}: {_sig4(metric_value)}")

    print(f"Mean Error: {_sig4(mean_error_avg)}")
    print(f"Trajectory Fail 0.2: {_sig4(traj_fail_02_avg)}")
    print(f"Trajectory Fail 0.5: {_sig4(traj_fail_05_avg)}")
    print(f"Jerk Average: {_sig4(jerk_avg)}")
    print(f"Skating Ratio Average: {_sig4(skating_ratio_avg)}")
    print(f"Feet Height Average: {_sig4(feet_height_avg)}")
    
    summary_result = {
        "mean_error": mean_error_avg,
        "traj_fail_02": traj_fail_02_avg,
        "traj_fail_05": traj_fail_05_avg,
        "jerk": jerk_avg,
        "skating_ratio": skating_ratio_avg,
        "feet_height": feet_height_avg,
        "power": 0.0,
        **tmr_metrics
    }

    if is_single:
        dp_mpjpe_paths, dp_mpjpe_merge_output_path, dp_mpjpe_shard_paths = _find_ranked_metric_json_paths(
            data_root_dir, "all_motions_with_gt_err"
        )
        dp_mpjpe_max_paths, dp_mpjpe_max_merge_output_path, dp_mpjpe_max_shard_paths = _find_ranked_metric_json_paths(
            data_root_dir, "all_motions_with_gt_err_max"
        )

        dp_mpjpe = _load_and_merge_scalar_dict_jsons(
            dp_mpjpe_paths,
            merge_output_path=dp_mpjpe_merge_output_path,
            archive_source_paths=dp_mpjpe_shard_paths,
            label="dp_mpjpe",
        )
        dp_mpjpe_max = _load_and_merge_scalar_dict_jsons(
            dp_mpjpe_max_paths,
            merge_output_path=dp_mpjpe_max_merge_output_path,
            archive_source_paths=dp_mpjpe_max_shard_paths,
            label="dp_mpjpe_max",
        )
        if dp_mpjpe is None:
            print("Warning: dp_mpjpe files not found in expected `results` directories")
        else:
            per_sample_metrics["dp_mpjpe"] = dp_mpjpe
        if dp_mpjpe_max is None:
            print("Warning: dp_mpjpe_max files not found in expected `results` directories")
        else:
            per_sample_metrics["dp_mpjpe_max"] = dp_mpjpe_max

        per_sample_metrics = _sig4_recursive(per_sample_metrics)
        metrics_path = os.path.join(data_root_dir, "omnicontrol_sorted_g1_single.json")
        with open(metrics_path, "w") as f:
            json.dump(per_sample_metrics, f, indent=4, ensure_ascii=False)
        print(f"Per-sample metrics saved to {metrics_path}")
        return per_sample_metrics

    eval_log_paths = _find_eval_log_paths(data_root_dir)
    all_mode_extra_keys = [
        "eval/tracking_success_rate_threshold_0.20",
        "eval/tracking_success_rate_threshold_0.50",
        "eval_mean/gt_err",
    ]
    if not eval_log_paths:
        print("Warning: evaluation log files not found in expected `results` directories")
    else:
        for metric_key in all_mode_extra_keys:
            metric_value = _parse_metric_from_logs(eval_log_paths, metric_key)
            if metric_value is None:
                print(f"Warning: metric `{metric_key}` not found in evaluation logs")
            else:
                summary_result[metric_key] = metric_value

    per_sample_metrics = _sig4_recursive(per_sample_metrics)
    summary_result = _sig4_recursive(summary_result)

    metrics_path = os.path.join(data_root_dir, "omnicontrol_sorted_g1_all.json")
    with open(metrics_path, "w") as f:
        json.dump(per_sample_metrics, f, indent=4, ensure_ascii=False)
    print(f"All-mode per-sample metrics saved to {metrics_path}")

    summary_path = os.path.join(data_root_dir, "omnicontrol_sorted_g1_all_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_result, f, indent=4, ensure_ascii=False)
    print(f"All-mode summary metrics saved to {summary_path}")

    return summary_result


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_control_error_g1(args.data_root_dir, args.data_gt_dir, tmr_mode=args.tmr_mode, eval_mode=args.eval_mode)

    
