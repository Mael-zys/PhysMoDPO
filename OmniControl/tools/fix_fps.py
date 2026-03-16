# ignore warning
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch
from einops import rearrange, repeat

from tools.slerp import slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_fps_joints(joints, old_fps, new_fps, mode="linear"):
    assert old_fps != 0
    scale_factor = new_fps / old_fps

    # joints: [..., T, J, 3]
    joints = joints.transpose(-3, -1)
    # joints: [..., 3, J, T]
    joints = torch.nn.functional.interpolate(
        joints, scale_factor=scale_factor, mode=mode
    )
    # joints: [..., 3, J, T2]
    joints = joints.transpose(-3, -1)
    # joints: [..., T2, J, 3]
    return joints


def interpolate_fps_trans(trans, old_fps, new_fps, mode="linear"):
    joints = trans[:, None]
    inter_joints = interpolate_fps_joints(joints, old_fps, new_fps, mode=mode)
    return inter_joints[:, 0]


def interpolate_fps_poses(poses, old_fps, new_fps, mode="linear"):
    # slerp interpolation using scipy for better handling of rotation discontinuities
    assert old_fps != 0
    scale_factor = new_fps / old_fps

    # Get back axis angle dimension
    poses_np = poses.cpu().numpy()
    poses_reshaped = rearrange(poses_np, "i (k t) -> i k t", t=3)
    
    nframes = len(poses_reshaped)
    num_joints = poses_reshaped.shape[1]
    
    # Calculate target frame count
    target_frames = int(np.round(nframes * scale_factor))
    
    # Original time points
    original_times = np.linspace(0, 1, nframes)
    # Target time points
    target_times = np.linspace(0, 1, target_frames)
    
    # Initialize output array
    interpolated_poses = np.zeros((target_frames, num_joints, 3))
    
    # Interpolate each joint separately
    for j in range(num_joints):
        # Get axis-angle vectors for this joint across all frames
        joint_poses = poses_reshaped[:, j, :]  # (nframes, 3)
        
        # Convert axis-angle to rotation objects
        # Handle zero rotations (no rotation)
        angles = np.linalg.norm(joint_poses, axis=-1)
        
        # Create rotation objects
        # For very small angles, scipy can handle them as identity rotations
        rotations = R.from_rotvec(joint_poses)
        
        # Create Slerp interpolator
        slerp_interp = Slerp(original_times, rotations)
        
        # Interpolate to target times
        interpolated_rots = slerp_interp(target_times)
        
        # Convert back to axis-angle
        interpolated_poses[:, j, :] = interpolated_rots.as_rotvec()
    
    # Reshape back to original format
    interpolated_poses = rearrange(interpolated_poses, "i k t -> i (k t)")
    
    # Convert back to torch tensor
    return torch.from_numpy(interpolated_poses).to(poses.device).to(poses.dtype)


