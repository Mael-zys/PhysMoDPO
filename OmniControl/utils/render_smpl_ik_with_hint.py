#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from renderer.humor import HumorRenderer
from tools.fix_fps import interpolate_fps_poses, interpolate_fps_trans
from tools.smpl_layer import SMPLH


def build_current_points_seq(hint):
    points_seq = []
    for frame in hint:
        mask = np.all(frame != 0, axis=1)
        points_seq.append(frame[mask])
    return points_seq


def build_smpl_joints_points_seq(joints):
    points_seq = []
    for frame in joints:
        points_seq.append(frame.astype(np.float32, copy=False))
    return points_seq


def build_full_points_seq_with_time_colors(
    hint,
    past_color,
    current_color,
    future_color,
):
    all_points = []
    all_times = []
    for t, frame in enumerate(hint):
        mask = np.all(frame != 0, axis=1)
        frame_points = frame[mask]
        if frame_points.shape[0] == 0:
            continue
        all_points.append(frame_points)
        all_times.append(np.full((frame_points.shape[0],), t, dtype=np.int32))

    if len(all_points) == 0:
        return build_current_points_seq(hint), None, None

    all_points = np.concatenate(all_points, axis=0).astype(np.float32)
    all_times = np.concatenate(all_times, axis=0)

    num_frames = hint.shape[0]
    points_seq = [all_points.copy() for _ in range(num_frames)]
    points_contact_seq = []
    for t in range(num_frames):
        # 1 => past points, rendered with contact_color in original sphere branch.
        points_contact_seq.append((all_times < t).astype(np.float32))
    points_contact_seq = np.stack(points_contact_seq, axis=0)

    # Keep original sphere rendering branch:
    # - point_color for non-past (current + future)
    # - contact_color for past
    # Sphere branch supports two-color split via contact_seq, so current/future share one color.
    non_past_color = np.asarray(current_color, dtype=np.float32).reshape(3).tolist()
    past_color = np.asarray(past_color, dtype=np.float32).reshape(3).tolist()

    return points_seq, points_contact_seq, (non_past_color, past_color)


def normalize_frame_idx(frame_idx, num_frames):
    idx = int(frame_idx)
    if idx < 0:
        idx = num_frames + idx
    if idx < 0 or idx >= num_frames:
        raise ValueError(f"Frame index out of range: {frame_idx}, valid range is [0, {num_frames - 1}]")
    return idx


def pick_first_rendered_image(render_dir: Path):
    image_files = sorted(render_dir.glob("*.png"))
    if len(image_files) == 0:
        image_files = sorted(render_dir.glob("*.jpg"))
    if len(image_files) == 0:
        image_files = sorted(render_dir.glob("*.jpeg"))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No rendered image found in: {render_dir}")
    return image_files[0]


def build_cam_rot_matrix(cam_rot_order, cam_rot_deg):
    order = str(cam_rot_order).strip()
    angles = np.asarray(cam_rot_deg, dtype=np.float32).reshape(-1)
    if len(order) == 0:
        raise ValueError("--cam-rot-order cannot be empty")
    if angles.shape[0] != len(order):
        raise ValueError(
            f"Length mismatch: --cam-rot-order is '{order}' ({len(order)} axes), "
            f"but --cam-rot-deg has {angles.shape[0]} values"
        )
    return R.from_euler(order, angles, degrees=True).as_matrix()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load ik_{idx}_0.npz + results_batch_{idx//32}.npy hint, "
            "interpolate SMPL 30fps->20fps, truncate by lengths, and render mesh video."
        )
    )
    parser.add_argument("--ik-path", type=str, required=True, help="Path like .../amass_format/ik/ik_{idx}_0.npz")
    parser.add_argument("--idx", type=int, default=None, help="Global sample idx. If not set, parse from ik filename.")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing results_batch_XXXX.npy.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used in saved results_batch files.")
    parser.add_argument("--src-fps", type=float, default=30.0, help="Source fps of SMPL params in ik file.")
    parser.add_argument("--dst-fps", type=float, default=20.0, help="Target fps for rendering.")
    parser.add_argument("--smplh-path", type=str, default="body_models/smplh", help="SMPLH model folder.")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Output mesh mp4 path. Default: beside ik file.")
    parser.add_argument(
        "--cam-offset",
        type=float,
        nargs=3,
        default=[0.0, 2.2, 0.9],
        help="Camera offset [x y z]. For overlap mode this is relative to camera anchor frame.",
    )
    parser.add_argument(
        "--cam-rot-order",
        type=str,
        default="xz",
        help="Euler rotation order for camera rotation, e.g. xz / xyz / zyx",
    )
    parser.add_argument(
        "--cam-rot-deg",
        type=float,
        nargs="+",
        default=[90.0, 180.0],
        help="Euler angles in degrees for camera rotation, matching --cam-rot-order length.",
    )
    parser.add_argument("--overlap-image", action="store_true", help="Render selected frames overlapped into one image.")
    parser.add_argument(
        "--overlap-frame-idxs",
        type=int,
        nargs="+",
        default=None,
        help="Frame indices for overlap image mode, e.g. --overlap-frame-idxs 0 10 20",
    )
    parser.add_argument(
        "--overlap-output",
        type=str,
        default=None,
        help="Overlap image output path (.png recommended). Default: <output_stem>_overlap.png",
    )
    parser.add_argument(
        "--overlap-body-alpha",
        type=float,
        default=0.95,
        help="Max alpha (latest frame) in overlap image mode.",
    )
    parser.add_argument(
        "--overlap-alpha-min",
        type=float,
        default=0.20,
        help="Min alpha (earliest frame) in overlap image mode.",
    )
    parser.add_argument(
        "--overlap-body-color",
        type=float,
        nargs=3,
        default=[0.0390625, 0.4140625, 0.796875],
        help="RGB color (0-1) used for all overlaid bodies in overlap mode.",
    )
    parser.add_argument(
        "--overlap-camera-frame-idx",
        type=int,
        default=None,
        help=(
            "Frame index used to anchor fixed camera center in overlap mode. "
            "Default: first selected overlap frame."
        ),
    )
    parser.add_argument("--point-rad", type=float, default=0.10)
    parser.add_argument(
        "--smpl-joints-point-rad",
        type=float,
        default=None,
        help="Point radius used only when --point-source smpl_joints. Defaults to --point-rad if not set.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Offscreen renderer point size for point-cloud mode (not needed for sphere branch).",
    )
    parser.add_argument(
        "--show-full-control-seq",
        action="store_true",
        help=(
            "Render full sequence control points at every frame. "
            "Past/current/future points use different colors."
        ),
    )
    parser.add_argument("--past-point-color", type=float, nargs=3, default=[1.0, 0.4, 0.0], help="RGB in [0,1].")
    parser.add_argument("--current-point-color", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="RGB in [0,1].")
    parser.add_argument("--future-point-color", type=float, nargs=3, default=[0.2, 0.5, 1.0], help="RGB in [0,1].")
    parser.add_argument("--put-ground", action="store_true", help="Shift motion to floor before rendering.")
    parser.add_argument(
        "--hide-body",
        action="store_true",
        help="Do not render SMPL body mesh. Keep control hints/ground only.",
    )
    parser.add_argument(
        "--hide-hint",
        action="store_true",
        help="Do not render control points (hint or SMPL joints). Keep body mesh/ground only.",
    )
    parser.add_argument(
        "--point-source",
        type=str,
        default="hint",
        choices=["hint", "smpl_joints"],
        help="Choose control points source: sparse hint points or all SMPL joints.",
    )
    parser.add_argument("--save-processed-ik", action="store_true", help="Save interpolated+truncated ik npz.")
    parser.add_argument("--dry-run", action="store_true", help="Only print metadata, do not render.")
    return parser.parse_args()


def infer_results_dir(ik_path: Path) -> Path:
    for parent in ik_path.parents:
        if parent.name == "amass_format":
            return parent.parent / "omnicontrol_output"
    raise ValueError("Cannot infer results dir from ik path. Please pass --results-dir.")


def parse_idx(ik_path: Path, idx: Optional[int]) -> int:
    if idx is not None:
        return idx
    match = re.search(r"ik_(\d+)_\d+\.npz$", ik_path.name)
    if match is None:
        raise ValueError("Cannot parse idx from ik filename. Please pass --idx.")
    return int(match.group(1))


def resolve_batch_file(results_dir: Path, batch_idx: int) -> Path:
    candidates = [
        results_dir / f"results_batch_{batch_idx:04d}.npy",
        results_dir / f"results_batch_{batch_idx}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find results_batch for batch_idx={batch_idx} in {results_dir}. Tried: {candidates}"
    )


def load_ik_data(ik_path: Path):
    data = np.load(ik_path, allow_pickle=True)

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
    elif "pose" in data:
        poses = np.asarray(data["pose"], dtype=np.float32)
        if poses.ndim == 3 and poses.shape[0] == 1:
            poses = poses[0]
    else:
        raise KeyError(f"{ik_path} has neither 'poses' nor 'pose'.")

    if "trans" in data:
        trans = np.asarray(data["trans"], dtype=np.float32)
    elif "transl" in data:
        trans = np.asarray(data["transl"], dtype=np.float32)
    else:
        raise KeyError(f"{ik_path} has neither 'trans' nor 'transl'.")

    if trans.ndim == 3 and trans.shape[0] == 1:
        trans = trans[0]

    if poses.shape[-1] < 66:
        raise ValueError(f"Expected pose dim >=66, got {poses.shape}.")
    poses = poses[:, :66]

    text = ""
    if "text" in data:
        text = str(data["text"])

    return poses, trans, text


def main():
    args = parse_args()
    ik_path = Path(args.ik_path)
    if not ik_path.exists():
        raise FileNotFoundError(f"ik file not found: {ik_path}")

    idx = parse_idx(ik_path, args.idx)
    results_dir = Path(args.results_dir) if args.results_dir else infer_results_dir(ik_path)
    batch_idx = idx // args.batch_size
    local_idx = idx % args.batch_size

    batch_file = resolve_batch_file(results_dir, batch_idx)
    batch_data = np.load(batch_file, allow_pickle=True).item()

    if batch_data.get("hint", None) is None:
        raise ValueError(f"'hint' is None in {batch_file}.")
    if local_idx >= len(batch_data["lengths"]):
        raise IndexError(
            f"local_idx={local_idx} out of range for lengths with size {len(batch_data['lengths'])} in {batch_file}"
        )
    hint = np.asarray(batch_data["hint"][local_idx], dtype=np.float32)
    length = int(np.asarray(batch_data["lengths"])[local_idx])
    batch_text = batch_data["text"][local_idx] if "text" in batch_data and len(batch_data["text"]) > local_idx else ""

    poses, trans, ik_text = load_ik_data(ik_path)
    text = ik_text if ik_text else str(batch_text)

    poses_t = torch.from_numpy(poses).float()
    trans_t = torch.from_numpy(trans).float()
    if args.src_fps != args.dst_fps:
        poses_t = interpolate_fps_poses(poses_t, args.src_fps, args.dst_fps)
        trans_t = interpolate_fps_trans(trans_t, args.src_fps, args.dst_fps)

    max_len = min(length, poses_t.shape[0], trans_t.shape[0], hint.shape[0])
    if max_len <= 0:
        raise ValueError(
            f"Invalid truncated length {max_len}, from length={length}, "
            f"poses={poses_t.shape[0]}, trans={trans_t.shape[0]}, hint={hint.shape[0]}"
        )

    poses_t = poses_t[:max_len, :66]
    trans_t = trans_t[:max_len]
    hint = hint[:max_len]

    output_path = Path(args.output) if args.output else ik_path.with_name(f"{ik_path.stem}_mesh.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"ik_path: {ik_path}")
    print(f"batch_file: {batch_file}")
    print(f"idx/local_idx: {idx}/{local_idx}")
    print(f"text: {text}")
    print(f"length(after truncate): {max_len}")
    print(f"pose shape: {tuple(poses_t.shape)}, trans shape: {tuple(trans_t.shape)}, hint shape: {tuple(hint.shape)}")
    print(f"point source: {args.point_source}")
    print(f"output: {output_path}")

    if args.dry_run:
        return

    cam_offset = np.asarray(args.cam_offset, dtype=np.float32).reshape(3)
    cam_rot_matrix = build_cam_rot_matrix(args.cam_rot_order, args.cam_rot_deg)

    device = torch.device(args.device)
    smplh = SMPLH(
        path=args.smplh_path,
        jointstype="both",
        input_pose_rep="axisangle",
        gender=args.gender,
    ).to(device).eval()

    with torch.no_grad():
        poses_dev = poses_t.to(device)
        trans_dev = trans_t.to(device)
        vertices, joints = smplh(poses_dev, trans_dev)
        offset = joints[0, 0] - trans_dev[0]
        vertices = (vertices - offset[None, None, :]).cpu().numpy()
        joints = (joints - offset[None, None, :]).cpu().numpy()

    if args.overlap_image:
        if not args.overlap_frame_idxs:
            raise ValueError("--overlap-image requires --overlap-frame-idxs")

        overlap_idxs = []
        for raw_idx in args.overlap_frame_idxs:
            idx_norm = normalize_frame_idx(raw_idx, vertices.shape[0])
            if idx_norm not in overlap_idxs:
                overlap_idxs.append(idx_norm)
        if len(overlap_idxs) == 0:
            raise ValueError("No valid frame indices for overlap image.")
        overlap_idxs = sorted(overlap_idxs)

        overlap_vertices = [vertices[fidx:fidx + 1].copy() for fidx in overlap_idxs]

        alpha_min = float(args.overlap_alpha_min)
        alpha_max = float(args.overlap_body_alpha)
        if alpha_min > alpha_max:
            alpha_min, alpha_max = alpha_max, alpha_min
        alpha_min = float(np.clip(alpha_min, 0.0, 1.0))
        alpha_max = float(np.clip(alpha_max, 0.0, 1.0))

        if len(overlap_idxs) == 1:
            alpha_values = np.array([alpha_max], dtype=np.float32)
        else:
            alpha_values = np.linspace(
                alpha_min,
                alpha_max,
                num=len(overlap_idxs),
                dtype=np.float32,
            )
        body_alphas = alpha_values.tolist()
        body_colors = [np.asarray(args.overlap_body_color, dtype=np.float32).reshape(3).tolist() for _ in overlap_vertices]

        if args.overlap_camera_frame_idx is None:
            camera_anchor_idx = overlap_idxs[0]
        else:
            camera_anchor_idx = normalize_frame_idx(args.overlap_camera_frame_idx, vertices.shape[0])
        anchor_mean = vertices[camera_anchor_idx].mean(axis=0)
        cam_offset_anchor = [
            float(anchor_mean[0] + cam_offset[0]),
            float(anchor_mean[1] + cam_offset[1]),
            float(cam_offset[2]),
        ]

        overlap_output_path = (
            Path(args.overlap_output)
            if args.overlap_output
            else output_path.with_name(f"{output_path.stem}_overlap.png")
        )
        overlap_output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_overlap_video = overlap_output_path.with_suffix(".mp4")

        renderer = HumorRenderer(fps=1.0)
        renderer(
            vertices=overlap_vertices,
            output=str(temp_overlap_video),
            cam_rot=cam_rot_matrix,
            cam_offset=cam_offset_anchor,
            follow_camera=False,
            render_body=not args.hide_body,
            body_colors=body_colors,
            body_alphas=body_alphas,
            put_ground=args.put_ground,
        )

        temp_render_dir = temp_overlap_video.with_suffix("")
        first_frame = pick_first_rendered_image(temp_render_dir)
        shutil.copy2(first_frame, overlap_output_path)

        if temp_overlap_video.exists():
            temp_overlap_video.unlink()
        if temp_render_dir.exists() and temp_render_dir.is_dir():
            shutil.rmtree(temp_render_dir, ignore_errors=True)

        print(f"Overlap frame idxs: {overlap_idxs}")
        print(f"Overlap alpha range: [{float(alpha_values.min()):.3f}, {float(alpha_values.max()):.3f}]")
        print(f"Overlap camera anchor frame idx: {camera_anchor_idx}")
        print(f"Overlap camera offset: {cam_offset_anchor}")
        print(f"Overlap camera rotation: order={args.cam_rot_order}, deg={list(np.asarray(args.cam_rot_deg, dtype=float))}")
        print(f"Overlap image saved to: {overlap_output_path}")
        print("Done.")
        return

    if args.hide_hint:
        points_seq = None
        points_contact_seq = None
        point_color = None
        point_contact_color = None
        point_size = args.point_size
        point_rad = args.point_rad
    elif args.point_source == "smpl_joints":
        if args.show_full_control_seq:
            print("Warning: --show-full-control-seq only applies to hint points, ignored for --point-source=smpl_joints.")
        points_seq = build_smpl_joints_points_seq(joints)
        points_contact_seq = None
        point_color = [1.0, 0.0, 0.0]
        point_contact_color = None
        point_size = args.point_size
        point_rad = args.smpl_joints_point_rad if args.smpl_joints_point_rad is not None else args.point_rad
    elif args.show_full_control_seq:
        points_seq, points_contact_seq, color_pair = build_full_points_seq_with_time_colors(
            hint=hint,
            past_color=args.past_point_color,
            current_color=args.current_point_color,
            future_color=args.future_point_color,
        )
        point_color, point_contact_color = color_pair if color_pair is not None else (None, None)
        point_size = args.point_size
        point_rad = args.point_rad
    else:
        points_seq = build_current_points_seq(hint)
        points_contact_seq = None
        point_color = None
        point_contact_color = None
        point_size = args.point_size
        point_rad = args.point_rad

    renderer = HumorRenderer(fps=args.dst_fps)
    render_kwargs = dict(
        vertices=vertices,
        output=str(output_path),
        points_seq=points_seq,
        cam_rot=cam_rot_matrix,
        cam_offset=cam_offset.tolist(),
        render_body=not args.hide_body,
        point_rad=point_rad,
        put_ground=args.put_ground,
    )
    if point_size is not None:
        render_kwargs["point_size"] = point_size
    if point_color is not None:
        render_kwargs["point_color"] = point_color
    if points_contact_seq is not None:
        render_kwargs["points_contact_seq"] = points_contact_seq
    if point_contact_color is not None:
        render_kwargs["contact_color"] = point_contact_color
    print(f"Camera setup: offset={cam_offset.tolist()}, rot_order={args.cam_rot_order}, rot_deg={list(np.asarray(args.cam_rot_deg, dtype=float))}")
    renderer(**render_kwargs)

    mesh_npz_path = output_path.with_suffix(".npz")
    mesh_save_dict = dict(
        vertices=vertices,
        points_seq=np.array([], dtype=object) if points_seq is None else np.array(points_seq, dtype=object),
        text=text,
        idx=idx,
        length=max_len,
        mocap_frame_rate=args.dst_fps,
        show_full_control_seq=args.show_full_control_seq and not args.hide_hint and args.point_source == "hint",
        point_source=args.point_source,
    )
    if point_color is not None:
        mesh_save_dict["point_color"] = np.asarray(point_color, dtype=np.float32)
    if point_contact_color is not None:
        mesh_save_dict["point_contact_color"] = np.asarray(point_contact_color, dtype=np.float32)
    np.savez(mesh_npz_path, **mesh_save_dict)

    if args.save_processed_ik:
        processed_ik_path = output_path.with_name(f"{output_path.stem}_processed_ik.npz")
        np.savez(
            processed_ik_path,
            poses=poses_t.cpu().numpy(),
            trans=trans_t.cpu().numpy(),
            betas=np.zeros(10, dtype=np.float32),
            num_betas=10,
            gender=args.gender,
            mocap_frame_rate=args.dst_fps,
            text=text,
        )

    tmp_dir = output_path.with_suffix("")
    if tmp_dir.exists() and tmp_dir.is_dir():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
