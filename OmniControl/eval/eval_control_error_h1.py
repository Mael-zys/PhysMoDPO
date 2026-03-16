import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pickle
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.humanml.utils.metrics import (
    calculate_feet_height,
    calculate_jerk,
    calculate_skating_ratio,
)
from utils.calculate_TMR_score.tmr_paths import ensure_tmr_on_path, get_tmr_path, resolve_tmr_path

ensure_tmr_on_path()
from TMR.src.process_h1_data import process_joints
from tools.fix_fps import interpolate_fps_joints
from utils.calculate_TMR_score.load_tmr_model import load_tmr_model_easy
from utils.calculate_TMR_score.tmr_eval_wrapper import calculate_tmr_metrics


# Control hints (24-joint hint space) -> H1 robot joint index (23 joints).
H1_CONTROL_IDX_DICT = {
    0: 0,   # pelvis
    10: 5,  # left ankle
    11: 10, # right ankle
    15: 22, # head proxy
    20: 20, # left wrist proxy
    21: 21, # right wrist proxy
}

H1_FACE_JOINT_IDX = [8, 3, 16, 12]  # r_hip_pitch, l_hip_pitch, r_shoulder, l_shoulder
H1_FEET_IDX = [5, 10]  # left_ankle, right_ankle
DEFAULT_INFER_FPS = 50.0
TMR_TARGET_FPS = 30.0
DEFAULT_H1_GT_RETARGET_PKL = "save/Eval_h1_fix/gt_retarget_joint_positions.pkl"
# In tmr_h1_guofeats stats, std of dims 34/35 is zero. Clamp them to avoid exploding normalization.
H1_TMR_ZERO_STD_SLICE = slice(34, 36)
PRED_JOINT_KEYS = (
    "body_pos",
    "pred_body_pos",
    "body_pos_pred",
    "joints_pred",
    "pred_joints",
    "joints",
)
GT_JOINT_KEYS = (
    "body_pos_gt",
    "gt_body_pos",
    "body_pos_target",
    "target_body_pos",
    "joints_gt",
    "gt_joints",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate control error for H1 retarget output.")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="save/Eval_h1_fix/gt_retarget_joint_positions.pkl",
        help=(
            "Prediction pkl path or root dir. Supports old *_eval.pkl and "
            "new motion_data_*_joint_positions.pkl formats."
        ),
    )
    parser.add_argument(
        "--data_gt_dir",
        type=str,
        default="save/cross_GT_hint",
        help="GT root dir. Will search retarget_h1_output/motion_data.pkl and hint npz files.",
    )
    parser.add_argument(
        "--gt_retarget_pkl",
        type=str,
        default=DEFAULT_H1_GT_RETARGET_PKL,
        help="GT joints pkl used for all GT body_pos (expects body_pos_gt/gt_body_pos per motion).",
    )
    parser.add_argument(
        "--tmr_mode",
        type=str,
        default="guofeats",
        choices=["translation_local", "guofeats", "velocity_local"],
        help="Mode for processing joints for TMR evaluation.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["all", "single"],
        help="all: aggregate summary + per-sample json, single: per-sample json only.",
    )
    parser.add_argument(
        "--tmr_run_dir",
        type=str,
        default="",
        help="Optional TMR run dir. Default: third-party/TMR/outputs/tmr_h1_<tmr_mode>",
    )
    parser.add_argument(
        "--tmr_ckpt_name",
        type=str,
        default="latest-epoch=362",
        help="TMR checkpoint name. e.g. last, latest-epoch=362, or auto.",
    )
    parser.add_argument(
        "--disable_tmr",
        action="store_true",
        help="Disable TMR metrics and only compute geometric/control metrics.",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=0,
        help="Only evaluate the first N motions for debugging (0 means all).",
    )
    parser.add_argument(
        "--infer_fps",
        type=float,
        default=DEFAULT_INFER_FPS,
        help="FPS of inference joint positions before resampling to GT fps.",
    )
    return parser.parse_args()


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


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_joblib(path):
    return joblib.load(path)


def _safe_getmtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _find_latest_file(file_paths):
    existing = [p for p in file_paths if os.path.isfile(p)]
    if not existing:
        return None
    return max(existing, key=_safe_getmtime)


def _find_latest_h1_eval_pkl(data_root_dir):
    eval_candidates = []
    joint_pos_candidates = []
    generic_pkl_candidates = []
    if os.path.isfile(data_root_dir) and data_root_dir.endswith(".pkl"):
        return data_root_dir

    search_dirs = [
        os.path.join(data_root_dir, "retarget_h1_output", "eval_outputs"),
        os.path.join(data_root_dir, "eval_outputs"),
        data_root_dir,
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for file_name in os.listdir(search_dir):
            if file_name.endswith("_eval.pkl"):
                eval_candidates.append(os.path.join(search_dir, file_name))
            elif file_name.endswith("_joint_positions.pkl"):
                joint_pos_candidates.append(os.path.join(search_dir, file_name))
            elif file_name.endswith(".pkl"):
                generic_pkl_candidates.append(os.path.join(search_dir, file_name))

    for candidates in (eval_candidates, joint_pos_candidates, generic_pkl_candidates):
        latest = _find_latest_file(candidates)
        if latest is not None:
            return latest
    return None


def _find_latest_h1_eval_metrics_txt(data_root_dir):
    candidates = []
    search_dirs = [
        os.path.join(data_root_dir, "retarget_h1_output", "eval_outputs"),
        os.path.join(data_root_dir, "eval_outputs"),
        data_root_dir,
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for file_name in os.listdir(search_dir):
            if file_name.endswith("_eval_metrics.txt"):
                candidates.append(os.path.join(search_dir, file_name))
    return _find_latest_file(candidates)


def _find_gt_motion_data_path(data_gt_dir):
    if os.path.isfile(data_gt_dir) and data_gt_dir.endswith(".pkl"):
        return data_gt_dir
    candidates = [
        os.path.join(data_gt_dir, "retarget_h1_output", "motion_data.pkl"),
        os.path.join(data_gt_dir, "motion_data.pkl"),
    ]
    return _find_latest_file(candidates)


def _parse_metric_text_file(metrics_txt_path):
    if metrics_txt_path is None or not os.path.isfile(metrics_txt_path):
        return {}
    out = {}
    with open(metrics_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "":
                continue
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def _motion_key_to_hint_path(motion_key, data_gt_dir):
    match = re.fullmatch(r"\d+-ik_(\d+)_(\d+)_(.+)", motion_key)
    if not match:
        return None
    folder_i = match.group(1)
    folder_j = match.group(2)
    sample_suffix = match.group(3)
    return os.path.join(
        data_gt_dir,
        f"{folder_i}_{folder_j}",
        "amass_format",
        "ik",
        f"ik_{sample_suffix}.npz",
    )


def _load_hint_and_text(motion_key, data_gt_dir):
    for candidate_key in _iter_motion_key_candidates(motion_key):
        hint_path = _motion_key_to_hint_path(candidate_key, data_gt_dir)
        if hint_path is None or not os.path.isfile(hint_path):
            continue
        with np.load(hint_path, allow_pickle=True) as payload:
            text = str(payload["text"]) if "text" in payload else candidate_key
            hint = payload["hint"].reshape(payload["hint"].shape[0], -1, 3) if "hint" in payload else None
            return hint, text
    return None, motion_key


def _iter_motion_key_candidates(motion_key, max_trim_steps=4):
    key = str(motion_key)
    out = [key]
    current = key
    for _ in range(max_trim_steps):
        if "_" not in current:
            break
        current = current.rsplit("_", 1)[0]
        out.append(current)
    # deduplicate while preserving order
    dedup = []
    seen = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _resolve_gt_motion_key(motion_key, gt_key_set):
    for candidate in _iter_motion_key_candidates(motion_key):
        if candidate in gt_key_set:
            return candidate
    return None


def _extract_joint_array(payload, candidate_keys):
    if not isinstance(payload, dict):
        return None
    for key in candidate_keys:
        if key not in payload:
            continue
        arr = np.asarray(payload[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr
    return None


def _extract_motion_dict(eval_data):
    if not isinstance(eval_data, dict):
        return None

    motions = eval_data.get("motions")
    if isinstance(motions, dict):
        return motions

    if not eval_data:
        return None

    # Fallback: dict-of-motions with motion_key as top-level keys.
    if all(isinstance(v, dict) for v in eval_data.values()):
        for value in eval_data.values():
            if _extract_joint_array(value, PRED_JOINT_KEYS) is not None:
                return eval_data
    return None


def _extract_motion_dict_with_joint_keys(payload, candidate_keys):
    if not isinstance(payload, dict):
        return None

    motions = payload.get("motions")
    if isinstance(motions, dict):
        return motions

    if not payload:
        return None

    if all(isinstance(v, dict) for v in payload.values()):
        for value in payload.values():
            if _extract_joint_array(value, candidate_keys) is not None:
                return payload
    return None


def _parse_h1_prediction_payload(eval_data):
    # Legacy eval.pkl format.
    if isinstance(eval_data, dict) and "motion_keys" in eval_data and "pred_body_pos" in eval_data:
        motion_keys = list(eval_data.get("motion_keys", []))
        pred_body_pos_list = list(eval_data.get("pred_body_pos", []))
        gt_body_pos_list = list(eval_data.get("gt_body_pos", []))
        mpjpe_per_motion = np.asarray(eval_data.get("mpjpe_per_motion", []), dtype=np.float32)
        termination_hist = np.asarray(eval_data.get("termination_hist", []))
        eval_metrics_from_pkl = dict(eval_data.get("metrics", {}))
        return (
            motion_keys,
            pred_body_pos_list,
            gt_body_pos_list,
            mpjpe_per_motion,
            termination_hist,
            eval_metrics_from_pkl,
        )

    motions = _extract_motion_dict(eval_data)
    if motions is None:
        raise RuntimeError(
            "Unsupported prediction pkl format. Expected legacy eval.pkl "
            "or dict with `motions` entries containing joint positions."
        )

    motion_keys = list(motions.keys())
    pred_body_pos_list = []
    gt_body_pos_list = []
    failed_flags = []
    for motion_key in motion_keys:
        payload = motions[motion_key]
        pred_body_pos_list.append(_extract_joint_array(payload, PRED_JOINT_KEYS))
        gt_body_pos_list.append(_extract_joint_array(payload, GT_JOINT_KEYS))
        failed_flags.append(bool(payload.get("failed", False)) if isinstance(payload, dict) else False)

    termination_hist = np.asarray([0.0 if failed else 1.0 for failed in failed_flags], dtype=np.float32)
    eval_metrics_from_pkl = dict(eval_data.get("metrics", {})) if isinstance(eval_data, dict) else {}
    mpjpe_per_motion = np.asarray([], dtype=np.float32)
    return (
        motion_keys,
        pred_body_pos_list,
        gt_body_pos_list,
        mpjpe_per_motion,
        termination_hist,
        eval_metrics_from_pkl,
    )


def _load_h1_gt_joint_map(gt_retarget_pkl_path):
    if gt_retarget_pkl_path is None or not os.path.isfile(gt_retarget_pkl_path):
        raise FileNotFoundError(
            f"GT retarget pkl not found: {gt_retarget_pkl_path}. "
            "Please provide --gt_retarget_pkl with a valid pkl path."
        )

    gt_data = _load_pickle(gt_retarget_pkl_path)
    gt_joint_map = {}

    if isinstance(gt_data, dict) and "motion_keys" in gt_data:
        motion_keys = list(gt_data.get("motion_keys", []))
        gt_list = list(gt_data.get("gt_body_pos", gt_data.get("body_pos_gt", [])))
        for i, motion_key in enumerate(motion_keys):
            if i >= len(gt_list):
                continue
            arr = np.asarray(gt_list[i], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                gt_joint_map[str(motion_key)] = arr
        return gt_joint_map

    motions = _extract_motion_dict_with_joint_keys(gt_data, GT_JOINT_KEYS)
    if motions is None:
        raise RuntimeError(
            "Unsupported GT retarget pkl format. Expected legacy format with "
            "`motion_keys` + `gt_body_pos` or dict with `motions` containing `body_pos_gt`."
        )

    for motion_key, motion_payload in motions.items():
        arr = _extract_joint_array(motion_payload, GT_JOINT_KEYS)
        if arr is None:
            continue
        gt_joint_map[str(motion_key)] = arr
    return gt_joint_map


def _resolve_motion_joints(motion_key, joints_map):
    for candidate in _iter_motion_key_candidates(motion_key):
        arr = joints_map.get(candidate)
        if arr is not None:
            return arr
    return None


def _get_motion_frame_count(motion_payload):
    if not isinstance(motion_payload, dict):
        return None
    for key in ("root_trans_offset", "dof", "pose_aa", "root_rot", "body_pos", "body_pos_gt"):
        value = motion_payload.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
    try:
        return int(motion_payload.get("num_frames"))
    except (TypeError, ValueError):
        return None


def _interpolate_joint_sequence_fps(joints, old_fps, new_fps):
    old_fps = float(old_fps)
    new_fps = float(new_fps)
    if old_fps <= 0 or new_fps <= 0 or np.isclose(old_fps, new_fps):
        return np.asarray(joints, dtype=np.float32)

    joints_t = torch.from_numpy(np.asarray(joints, dtype=np.float32))
    joints_interp = interpolate_fps_joints(joints_t, old_fps=old_fps, new_fps=new_fps)
    return joints_interp.cpu().numpy().astype(np.float32)


def _recenter_xy(joints):
    joints = np.asarray(joints, dtype=np.float32).copy()
    joints[:, :, :2] -= joints[0:1, 0:1, :2]
    return joints


def _safe_filename_token(text):
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _derive_output_tag(eval_pkl_path):
    stem = Path(eval_pkl_path).stem

    motion_match = re.fullmatch(r"motion_data_(.+?)_joint_positions", stem)
    if motion_match:
        return _safe_filename_token(motion_match.group(1))

    if stem == "gt_retarget_joint_positions":
        return "gt"

    eval_match = re.fullmatch(r"(.+)_eval", stem)
    if eval_match:
        return _safe_filename_token(eval_match.group(1))

    return _safe_filename_token(stem)


def _build_output_filename(base_name, tag):
    if tag:
        return f"{base_name}_{tag}.json"
    return f"{base_name}.json"


def _match_motion_length(joints, target_length):
    joints = np.asarray(joints, dtype=np.float32)
    target_length = int(target_length)
    if target_length <= 0:
        return joints[:0].copy()

    current_length = int(joints.shape[0])
    if current_length == target_length:
        return joints
    if current_length > target_length:
        return joints[:target_length].copy()
    if current_length == 0:
        return np.zeros((target_length, joints.shape[1], joints.shape[2]), dtype=np.float32)

    pad_count = target_length - current_length
    last_frame = joints[-1:]
    pad = np.repeat(last_frame, pad_count, axis=0)
    return np.concatenate([joints, pad], axis=0)


def _build_h1_control_mask(hint_joints, length, num_eval_joints):
    mask = np.zeros((length, num_eval_joints), dtype=bool)
    if hint_joints is None:
        for h1_idx in H1_CONTROL_IDX_DICT.values():
            if h1_idx < num_eval_joints:
                mask[:, h1_idx] = True
        return mask

    hint_mask = np.any(hint_joints != 0, axis=-1)
    for control_idx, h1_joint_idx in H1_CONTROL_IDX_DICT.items():
        if control_idx >= hint_mask.shape[1] or h1_joint_idx >= num_eval_joints:
            continue
        mask[:, h1_joint_idx] = hint_mask[:length, control_idx]
    return mask


def _find_best_tmr_ckpt(run_dir, ckpt_name):
    if ckpt_name != "auto":
        return ckpt_name

    last_dir = os.path.join(run_dir, "last_weights")
    if os.path.isdir(last_dir):
        return "last"

    epoch_candidates = []
    if os.path.isdir(run_dir):
        for file_name in os.listdir(run_dir):
            match = re.fullmatch(r"latest-epoch=(\d+)_weights", file_name)
            if not match:
                continue
            epoch_candidates.append((int(match.group(1)), f"latest-epoch={match.group(1)}"))
    if epoch_candidates:
        epoch_candidates.sort(key=lambda x: x[0])
        return epoch_candidates[-1][1]

    return "last"


def _try_init_tmr_forward(tmr_mode, run_dir="", ckpt_name="auto", disable_tmr=False):
    if disable_tmr:
        return None

    resolved_run_dir = resolve_tmr_path(run_dir) or get_tmr_path("outputs", f"tmr_h1_{tmr_mode}")
    resolved_ckpt_name = _find_best_tmr_ckpt(resolved_run_dir, ckpt_name)
    try:
        print(f"Loading TMR model: run_dir={resolved_run_dir}, ckpt={resolved_ckpt_name}")
        return load_tmr_model_easy(
            device="cuda",
            run_dir=resolved_run_dir,
            ckpt_name=resolved_ckpt_name,
        )
    except Exception as exc:
        print(f"Warning: failed to load TMR model, skip TMR metrics. reason: {exc}")
        return None


def evaluate_control_error_h1(
    data_root_dir,
    data_gt_dir,
    gt_retarget_pkl=DEFAULT_H1_GT_RETARGET_PKL,
    tmr_mode="guofeats",
    eval_mode="all",
    tmr_run_dir="",
    tmr_ckpt_name="auto",
    disable_tmr=False,
    max_motions=0,
    infer_fps=DEFAULT_INFER_FPS,
):
    is_single = eval_mode == "single"
    is_all = eval_mode == "all"

    eval_pkl_path = _find_latest_h1_eval_pkl(data_root_dir)
    if eval_pkl_path is None:
        raise FileNotFoundError(
            "No H1 prediction pkl found. Expected a direct pkl path or files under "
            "`<data_root_dir>/retarget_h1_output/eval_outputs` / `eval_outputs` / `<data_root_dir>`."
        )
    eval_metrics_txt_path = _find_latest_h1_eval_metrics_txt(data_root_dir)

    gt_motion_data_path = _find_gt_motion_data_path(data_gt_dir)
    if gt_motion_data_path is None:
        raise FileNotFoundError(
            "No GT motion_data.pkl found. Expected under "
            "`<data_gt_dir>/retarget_h1_output/motion_data.pkl`."
        )

    print(f"Using pred eval pkl: {eval_pkl_path}")
    if eval_metrics_txt_path is not None:
        print(f"Using pred metrics txt: {eval_metrics_txt_path}")
    print(f"Using GT motion_data: {gt_motion_data_path}")
    print(f"Using GT joints pkl: {gt_retarget_pkl}")

    eval_data = _load_pickle(eval_pkl_path)
    motion_data_gt = _load_joblib(gt_motion_data_path)
    gt_joint_map = _load_h1_gt_joint_map(gt_retarget_pkl)

    (
        motion_keys,
        pred_body_pos_list,
        _gt_body_pos_list_unused,
        mpjpe_per_motion,
        termination_hist,
        eval_metrics_from_pkl,
    ) = _parse_h1_prediction_payload(eval_data)
    eval_metrics_from_txt = _parse_metric_text_file(eval_metrics_txt_path)

    if not motion_keys or not pred_body_pos_list:
        raise RuntimeError("Invalid prediction pkl: missing motion keys or predicted joint positions.")
    if len(motion_keys) != len(pred_body_pos_list):
        raise RuntimeError(
            f"Invalid prediction pkl: len(motion_keys)={len(motion_keys)} "
            f"!= len(pred_body_pos)={len(pred_body_pos_list)}"
        )
    if not gt_joint_map:
        raise RuntimeError(
            "GT retarget pkl does not contain valid GT joint positions "
            "(expected keys like `body_pos_gt` / `gt_body_pos`)."
        )

    tmr_forward = _try_init_tmr_forward(
        tmr_mode=tmr_mode,
        run_dir=tmr_run_dir,
        ckpt_name=tmr_ckpt_name,
        disable_tmr=disable_tmr,
    )

    total_eval_num = len(motion_keys) if max_motions <= 0 else min(len(motion_keys), int(max_motions))
    gt_key_set = set(motion_data_gt.keys())
    is_maskedmimic_input = "maskedmimic" in str(data_root_dir).lower()
    output_dir = data_root_dir if os.path.isdir(data_root_dir) else os.path.dirname(data_root_dir)
    output_tag = _derive_output_tag(eval_pkl_path)
    if output_dir == "":
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

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
        per_sample_metrics["mpjpe_per_motion_mm"] = {}

    text_list = [] if (is_all and tmr_forward is not None) else None
    joints_feats_pred_list = [] if (is_all and tmr_forward is not None) else None
    joints_feats_gt_list = [] if (is_all and tmr_forward is not None) else None
    sample_names_for_tmr = [] if (is_all and tmr_forward is not None) else None

    for idx in tqdm(range(total_eval_num), desc="Evaluating H1 motions"):
        motion_key = motion_keys[idx]
        gt_motion_key = _resolve_gt_motion_key(motion_key, gt_key_set)
        if gt_motion_key is None:
            continue

        pred_raw = pred_body_pos_list[idx]
        if pred_raw is None:
            continue
        pred = np.asarray(pred_raw, dtype=np.float32)
        if pred.ndim != 3 or pred.shape[-1] != 3:
            continue
        pred_before_trim = pred
        if is_maskedmimic_input and pred.shape[0] > 50:
            pred = pred[50:]
        elif is_maskedmimic_input and pred.shape[0] <= 50:
            # Keep one frame for failed / very short samples, then pad to GT length below.
            pred = pred_before_trim[-1:].copy()

        gt = _resolve_motion_joints(gt_motion_key, gt_joint_map)
        if gt is None:
            gt = _resolve_motion_joints(motion_key, gt_joint_map)
        if gt is None:
            continue
        gt = np.asarray(gt, dtype=np.float32)
        if gt.ndim != 3 or gt.shape[-1] != 3:
            continue

        gt_frame_count = _get_motion_frame_count(motion_data_gt[gt_motion_key])
        infer_fps_value = float(infer_fps)
        fps_value = float(motion_data_gt[gt_motion_key].get("fps", 20.0))

        pred = _recenter_xy(pred)
        gt = _recenter_xy(gt)

        should_resample_pred = False
        if infer_fps_value > 0 and gt_frame_count is not None and abs(pred.shape[0] - gt_frame_count) > 1:
            projected_pred_frames = int(round(pred.shape[0] * fps_value / infer_fps_value))
            should_resample_pred = abs(projected_pred_frames - gt_frame_count) <= 2
        elif infer_fps_value > 0 and gt_frame_count is None:
            should_resample_pred = not np.isclose(infer_fps_value, fps_value)

        if should_resample_pred:
            pred = _interpolate_joint_sequence_fps(pred, old_fps=infer_fps_value, new_fps=fps_value)
            pred = _recenter_xy(pred)

        if (
            infer_fps_value > 0
            and gt_frame_count is not None
            and abs(gt.shape[0] - gt_frame_count) > 1
        ):
            projected_frames = int(round(gt.shape[0] * fps_value / infer_fps_value))
            if abs(projected_frames - gt_frame_count) <= 2:
                gt = _interpolate_joint_sequence_fps(gt, old_fps=infer_fps_value, new_fps=fps_value)
                gt = _recenter_xy(gt)

        target_motion_length = int(gt_frame_count) if gt_frame_count is not None and int(gt_frame_count) > 0 else int(gt.shape[0])
        pred = _match_motion_length(pred, target_motion_length)
        gt = _match_motion_length(gt, target_motion_length)
        pred = _recenter_xy(pred)
        gt = _recenter_xy(gt)

        hint_joints, text = _load_hint_and_text(gt_motion_key, data_gt_dir)

        length = min(pred.shape[0], gt.shape[0])
        if hint_joints is not None:
            length = min(length, hint_joints.shape[0])
        if length <= 0:
            continue

        num_eval_joints = pred.shape[1]
        h1_hint_mask = _build_h1_control_mask(hint_joints, length, num_eval_joints)
        if h1_hint_mask.sum() == 0:
            continue

        joint_error = np.linalg.norm(pred[:length] - gt[:length], axis=-1)
        control_error = joint_error * h1_hint_mask
        valid_mask_count = int(h1_hint_mask.sum())
        mean_error = float(control_error.sum() / max(valid_mask_count, 1))

        traj_fail_02 = float(1.0 - ((joint_error <= 0.2) | (~h1_hint_mask)).all())
        traj_fail_05 = float(1.0 - ((joint_error <= 0.5) | (~h1_hint_mask)).all())

        pred_torch_eval = (
            torch.from_numpy(pred[:length])
            .float()
            .unsqueeze(0)
            .permute(0, 2, 3, 1)
            .cpu()
        )
        if length >= 4:
            jerk_value = float(calculate_jerk(pred_torch_eval.clone(), [length], fps=fps_value))
            if not np.isfinite(jerk_value):
                jerk_value = 0.0
        else:
            jerk_value = 0.0
        skating_ratio, _ = calculate_skating_ratio(
            pred_torch_eval.clone(),
            [length],
            fps=fps_value,
            feet_idx=H1_FEET_IDX,
            height_index=2,
            thresh_height=0.08,
        )
        skating_ratio_value = float(skating_ratio)
        feet_height_value = float(
            calculate_feet_height(
                pred_torch_eval.clone(),
                [length],
                feet_idx=H1_FEET_IDX,
                height_index=2,
            )
        )

        mean_error_sum += mean_error
        traj_fail_02_sum += traj_fail_02
        traj_fail_05_sum += traj_fail_05
        jerk_sum += jerk_value
        skating_ratio_sum += skating_ratio_value
        feet_height_sum += feet_height_value
        valid_motion_count += 1

        per_sample_metrics["mean_error"][motion_key] = mean_error
        per_sample_metrics["traj_fail_02"][motion_key] = traj_fail_02
        per_sample_metrics["traj_fail_05"][motion_key] = traj_fail_05
        per_sample_metrics["jerk"][motion_key] = jerk_value
        per_sample_metrics["skate_ratio"][motion_key] = skating_ratio_value
        per_sample_metrics["feet_height"][motion_key] = feet_height_value
        per_sample_metrics["power"][motion_key] = 0.0
        if is_single and idx < len(mpjpe_per_motion):
            per_sample_metrics["mpjpe_per_motion_mm"][motion_key] = float(mpjpe_per_motion[idx])

        if tmr_forward is None:
            continue

        pred_for_tmr = pred[:length].copy()
        gt_for_tmr = gt[:length].copy()
        if not np.isclose(fps_value, TMR_TARGET_FPS):
            pred_for_tmr = _interpolate_joint_sequence_fps(
                pred_for_tmr,
                old_fps=fps_value,
                new_fps=TMR_TARGET_FPS,
            )
            gt_for_tmr = _interpolate_joint_sequence_fps(
                gt_for_tmr,
                old_fps=fps_value,
                new_fps=TMR_TARGET_FPS,
            )
            pred_for_tmr = _recenter_xy(pred_for_tmr)
            gt_for_tmr = _recenter_xy(gt_for_tmr)

        if tmr_mode == "guofeats" and len(pred_for_tmr) < 2:
            continue

        joints_pred_for_tmr = process_joints(
            pred_for_tmr,
            mode=tmr_mode,
            align_init_yaw=True,
            heading_target_axis="y",
            face_joint_idx=H1_FACE_JOINT_IDX,
        )
        joints_gt_for_tmr = process_joints(
            gt_for_tmr,
            mode=tmr_mode,
            align_init_yaw=True,
            heading_target_axis="y",
            face_joint_idx=H1_FACE_JOINT_IDX,
        )

        joints_pred_feat = joints_pred_for_tmr.reshape(joints_pred_for_tmr.shape[0], -1)
        joints_gt_feat = joints_gt_for_tmr.reshape(joints_gt_for_tmr.shape[0], -1)
        if tmr_mode == "guofeats" and joints_pred_feat.shape[1] >= H1_TMR_ZERO_STD_SLICE.stop:
            joints_pred_feat[:, H1_TMR_ZERO_STD_SLICE] = 0.0
            joints_gt_feat[:, H1_TMR_ZERO_STD_SLICE] = 0.0

        if is_single:
            sample_tmr_metrics = calculate_tmr_metrics(
                tmr_forward=tmr_forward,
                texts_gt=[text],
                motions_guofeats_pred=[joints_pred_feat],
                motions_guofeats_gt=[joints_gt_feat],
                calculate_retrieval=False,
                calculate_fid=False,
            )
            per_sample_metrics["m2t_score"][motion_key] = float(sample_tmr_metrics.get("m2t_score", 0.0))
            per_sample_metrics["m2m_score"][motion_key] = float(sample_tmr_metrics.get("m2m_score", 0.0))
        else:
            text_list.append(text)
            joints_feats_pred_list.append(joints_pred_feat)
            joints_feats_gt_list.append(joints_gt_feat)
            sample_names_for_tmr.append(motion_key)

    if valid_motion_count == 0:
        raise RuntimeError("No valid motions found for H1 evaluation.")

    num_motions = valid_motion_count
    summary_result = {
        "num_motions_evaluated": int(num_motions),
        "num_motions_total": int(total_eval_num),
        "mean_error": mean_error_sum / num_motions,
        "traj_fail_02": traj_fail_02_sum / num_motions,
        "traj_fail_05": traj_fail_05_sum / num_motions,
        "jerk": jerk_sum / num_motions,
        "skating_ratio": skating_ratio_sum / num_motions,
        "feet_height": feet_height_sum / num_motions,
        "power": 0.0,
    }

    if termination_hist.size > 0:
        summary_result["success_rate_from_termination"] = float(np.mean(termination_hist.astype(np.float32)))

    tmr_metrics = {}
    if is_all and tmr_forward is not None and len(joints_feats_pred_list) > 0:
        tmr_batch_size = 32
        tmr_metrics_sum = {}
        tmr_batch_count = 0

        for batch_start in range(0, len(joints_feats_pred_list), tmr_batch_size):
            batch_end = min(batch_start + tmr_batch_size, len(joints_feats_pred_list))
            batch_pred = joints_feats_pred_list[batch_start:batch_end]
            batch_gt = joints_feats_gt_list[batch_start:batch_end]
            batch_texts = text_list[batch_start:batch_end]
            batch_names = sample_names_for_tmr[batch_start:batch_end]
            if len(batch_pred) == 0:
                continue

            batch_metrics = calculate_tmr_metrics(
                tmr_forward=tmr_forward,
                texts_gt=batch_texts,
                motions_guofeats_pred=batch_pred,
                motions_guofeats_gt=batch_gt,
                calculate_retrieval=True,
                calculate_fid=True,
            )
            for metric_name, metric_value in batch_metrics.items():
                tmr_metrics_sum[metric_name] = tmr_metrics_sum.get(metric_name, 0.0) + float(metric_value)
            tmr_batch_count += 1

            for sample_idx, sample_name in enumerate(batch_names):
                sample_metric = calculate_tmr_metrics(
                    tmr_forward=tmr_forward,
                    texts_gt=[batch_texts[sample_idx]],
                    motions_guofeats_pred=[batch_pred[sample_idx]],
                    motions_guofeats_gt=[batch_gt[sample_idx]],
                    calculate_retrieval=False,
                    calculate_fid=False,
                )
                per_sample_metrics["m2t_score"][sample_name] = float(sample_metric.get("m2t_score", 0.0))
                per_sample_metrics["m2m_score"][sample_name] = float(sample_metric.get("m2m_score", 0.0))

        if tmr_batch_count > 0:
            tmr_metrics = {
                metric_name: metric_sum / tmr_batch_count
                for metric_name, metric_sum in tmr_metrics_sum.items()
            }
            summary_result.update(tmr_metrics)

    # Keep native h1 evaluator metrics (if provided) for direct comparison.
    for k, v in eval_metrics_from_pkl.items():
        try:
            summary_result[k] = float(v)
        except (TypeError, ValueError):
            summary_result[k] = v
    for k, v in eval_metrics_from_txt.items():
        if k not in summary_result:
            summary_result[k] = v

    per_sample_metrics = _sig4_recursive(per_sample_metrics)
    summary_result = _sig4_recursive(summary_result)

    if is_single:
        metrics_path = os.path.join(
            output_dir,
            _build_output_filename("omnicontrol_sorted_h1_single", output_tag),
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(per_sample_metrics, f, indent=4, ensure_ascii=False)
        print(f"Per-sample metrics saved to {metrics_path}")
        return per_sample_metrics

    metrics_path = os.path.join(
        output_dir,
        _build_output_filename("omnicontrol_sorted_h1_all", output_tag),
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_metrics, f, indent=4, ensure_ascii=False)
    print(f"All-mode per-sample metrics saved to {metrics_path}")

    summary_path = os.path.join(
        output_dir,
        _build_output_filename("omnicontrol_sorted_h1_all_summary", output_tag),
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=4, ensure_ascii=False)
    print(f"All-mode summary metrics saved to {summary_path}")
    return summary_result


if __name__ == "__main__":
    args = parse_args()
    evaluate_control_error_h1(
        data_root_dir=args.data_root_dir,
        data_gt_dir=args.data_gt_dir,
        gt_retarget_pkl=args.gt_retarget_pkl,
        tmr_mode=args.tmr_mode,
        eval_mode=args.eval_mode,
        tmr_run_dir=args.tmr_run_dir,
        tmr_ckpt_name=args.tmr_ckpt_name,
        disable_tmr=args.disable_tmr,
        max_motions=args.max_motions,
        infer_fps=args.infer_fps,
    )
