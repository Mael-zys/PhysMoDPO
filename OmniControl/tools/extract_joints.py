import numpy as np


def extract_joints(x, featsname, **kwargs):
    if featsname == "smplrifke":
        return extract_joints_smplrifke(x, **kwargs)
    elif featsname == "guoh3dfeats":
        return extract_joints_guoh3dfeats(x, **kwargs)
    else:
        raise NotImplementedError


def extract_joints_batch(x, featsname, **kwargs):
    """
    Batch version of extract_joints.
    
    Args:
        x: Input features of shape (B, T, D) where B is batch size, T is sequence length, D is feature dimension
        featsname: Feature type name
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with output tensors having batch dimension
    """
    if featsname == "smplrifke":
        return extract_joints_smplrifke_batch(x, **kwargs)
    elif featsname == "guoh3dfeats":
        return extract_joints_guoh3dfeats_batch(x, **kwargs)
    elif featsname == "smpldata":
        return extract_joints_smpldata_batch(x, **kwargs)
    else:
        raise NotImplementedError


def extract_joints_smplrifke(
    x, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, keep_torch=False, joints_only=False, **kwargs
):
    assert x.shape[-1] == 205
    if value_from == "smpl":
        assert smpl_layer is not None

    # smplrifke
    from tools.smplrifke_feats import smplrifkefeats_to_smpldata

    smpldata = smplrifkefeats_to_smpldata(x, first_angle=first_angle)

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"] # (T, 66)
    trans = smpldata["trans"] # (T, 3)
    joints = smpldata["joints"]

    if value_from == "smpl":
        # Freeze SMPL layer parameters but allow gradient flow
        for param in smpl_layer.parameters():
            param.requires_grad = False
            
        vertices, joints = smpl_layer(poses, trans)

        # remove offset
        offset = joints[0, 0] - smpldata["trans"][0]
        offset = offset[None, None, :]
        joints -= offset
        vertices -= offset

        if not keep_torch:
            # Handle both CPU and GPU tensors
            if vertices.is_cuda:
                vertices = vertices.cpu().numpy()
                joints = joints.cpu().numpy()
            else:
                vertices = vertices.numpy()
                joints = joints.numpy()
        if joints_only:
            output = {"joints": joints}
        else:
            output = {
                "vertices": vertices,
                "joints": joints,
                "smpldata": smpldata,
            }
    elif value_from == "joints":
        if not keep_torch:
            # Handle both CPU and GPU tensors
            if joints.is_cuda:
                joints = joints.cpu().numpy()
            else:
                joints = joints.numpy()
        output = {"joints": joints}
    else:
        raise NotImplementedError
    return output

def extract_joints_smpldata(
    x, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, keep_torch=False, joints_only=False, **kwargs
):
    if value_from == "smpl":
        assert smpl_layer is not None


    smpldata = x

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"] # (T, 66)
    trans = smpldata["trans"] # (T, 3)
    joints = smpldata["joints"]

    if value_from == "smpl":
        # Freeze SMPL layer parameters but allow gradient flow
        for param in smpl_layer.parameters():
            param.requires_grad = False
            
        vertices, joints = smpl_layer(poses, trans)

        # remove offset
        offset = joints[0, 0] - smpldata["trans"][0]
        offset = offset[None, None, :]
        joints -= offset
        vertices -= offset

        if not keep_torch:
            # Handle both CPU and GPU tensors
            if vertices.is_cuda:
                vertices = vertices.cpu().numpy()
                joints = joints.cpu().numpy()
            else:
                vertices = vertices.numpy()
                joints = joints.numpy()
        if joints_only:
            output = {"joints": joints}
        else:
            output = {
                "vertices": vertices,
                "joints": joints,
                "smpldata": smpldata,
            }
    elif value_from == "joints":
        if not keep_torch:
            # Handle both CPU and GPU tensors
            if joints.is_cuda:
                joints = joints.cpu().numpy()
            else:
                joints = joints.numpy()
        output = {"joints": joints}
    else:
        raise NotImplementedError
    return output

def extract_joints_guoh3dfeats(x, keep_torch=False, **kwargs):
    assert x.shape[-1] == 263
    from tools.guofeats import guofeats_to_joints
    
    joints = guofeats_to_joints(x)
    if not keep_torch:
        # Handle both CPU and GPU tensors
        if joints.is_cuda:
            joints = joints.cpu().numpy()
        else:
            joints = joints.numpy()

    output = {"joints": joints}
    return output


def extract_joints_smplrifke_batch(
    x, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, keep_torch=False, joints_only=False, **kwargs
):
    """
    Batch version of extract_joints_smplrifke.
    
    Args:
        x: Input features of shape (B, T, 205) where B is batch size, T is sequence length
        fps: Frame rate
        value_from: "joints" or "smpl"
        smpl_layer: SMPL layer for computing vertices and joints from poses
        first_angle: Initial angle (scalar or tensor of shape (B,))
        keep_torch: Whether to keep tensors as torch tensors
    
    Returns:
        Dictionary with output tensors having batch dimension
    """
    assert x.shape[-1] == 205, f"Expected last dim to be 205, got {x.shape[-1]}"
    
    if smpl_layer is not None:
        smpl_layer = smpl_layer.eval()
        smpl_layer = smpl_layer.to(x.device)

    # Check if input has batch dimension
    if len(x.shape) == 2:
        # Single sequence, use original function
        return extract_joints_smplrifke(x, fps=fps, value_from=value_from, 
                                       smpl_layer=smpl_layer, first_angle=first_angle, 
                                       keep_torch=keep_torch, joints_only=joints_only, **kwargs)
    
    if value_from == "smpl":
        assert smpl_layer is not None

    # smplrifke
    from tools.smplrifke_feats import smplrifkefeats_to_smpldata_batch

    smpldata = smplrifkefeats_to_smpldata_batch(x, first_angle=first_angle)

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"]  # (B, T, 66)
    trans = smpldata["trans"]  # (B, T, 3)
    joints = smpldata["joints"]  # (B, T, 24, 3)

    if value_from == "smpl":
        batch_size, seq_len = poses.shape[:2]
        
        # Flatten batch and sequence dimensions for SMPL layer
        poses_flat = poses.reshape(batch_size * seq_len, -1)  # (B*T, 66)
        trans_flat = trans.reshape(batch_size * seq_len, -1)  # (B*T, 3)
        
        vertices, joints = smpl_layer(poses_flat, trans_flat)
        
        # Reshape back to batch format
        vertices = vertices.reshape(batch_size, seq_len, *vertices.shape[1:])
        joints = joints.reshape(batch_size, seq_len, *joints.shape[1:])
        
        # remove offset
        offset = joints[:, 0, 0] - smpldata["trans"][:, 0]
        offset = offset[:, None, None, :]
        joints -= offset
        vertices -= offset

        if not keep_torch:
            # Handle both CPU and GPU tensors
            if vertices.is_cuda:
                vertices = vertices.cpu().numpy()
                joints = joints.cpu().numpy()
            else:
                vertices = vertices.numpy()
                joints = joints.numpy()
        if joints_only:
            output = {"joints": joints}
        else:
            output = {
                "vertices": vertices,
                "joints": joints,
                "smpldata": smpldata,
            }
    elif value_from == "joints":
        if not keep_torch:
            # Handle both CPU and GPU tensors
            if joints.is_cuda:
                joints = joints.cpu().numpy()
            else:
                joints = joints.numpy()
        output = {"joints": joints}
    else:
        raise NotImplementedError
    return output

def extract_joints_smpldata_batch(
    x, fps, value_from="joints", smpl_layer=None, first_angle=np.pi, keep_torch=False, joints_only=False, **kwargs
):
    """
    Batch version of extract_joints_smplrifke.
    
    Args:
        x: Input features of shape (B, T, 205) where B is batch size, T is sequence length
        fps: Frame rate
        value_from: "joints" or "smpl"
        smpl_layer: SMPL layer for computing vertices and joints from poses
        first_angle: Initial angle (scalar or tensor of shape (B,))
        keep_torch: Whether to keep tensors as torch tensors
    
    Returns:
        Dictionary with output tensors having batch dimension
    """
    
    if smpl_layer is not None:
        smpl_layer = smpl_layer.eval()
        smpl_layer = smpl_layer.to(x['poses'].device)

    # Check if input has batch dimension
    if len(x['poses'].shape) == 2:
        # Single sequence, use original function
        return extract_joints_smpldata(x, fps=fps, value_from=value_from, 
                                       smpl_layer=smpl_layer, first_angle=first_angle, 
                                       keep_torch=keep_torch, joints_only=joints_only, **kwargs)
    
    if value_from == "smpl":
        assert smpl_layer is not None

    smpldata = x

    smpldata["mocap_framerate"] = fps
    poses = smpldata["poses"]  # (B, T, 66)
    trans = smpldata["trans"]  # (B, T, 3)
    joints = smpldata["joints"]  # (B, T, 24, 3)

    if value_from == "smpl":
        batch_size, seq_len = poses.shape[:2]
        
        # Flatten batch and sequence dimensions for SMPL layer
        poses_flat = poses.reshape(batch_size * seq_len, -1)  # (B*T, 66)
        trans_flat = trans.reshape(batch_size * seq_len, -1)  # (B*T, 3)
        
        vertices, joints = smpl_layer(poses_flat, trans_flat)
        
        # Reshape back to batch format
        vertices = vertices.reshape(batch_size, seq_len, *vertices.shape[1:])
        joints = joints.reshape(batch_size, seq_len, *joints.shape[1:])
        
        # remove offset
        offset = joints[:, 0, 0] - smpldata["trans"][:, 0]
        offset = offset[:, None, None, :]
        joints -= offset
        vertices -= offset

        if not keep_torch:
            # Handle both CPU and GPU tensors
            if vertices.is_cuda:
                vertices = vertices.cpu().numpy()
                joints = joints.cpu().numpy()
            else:
                vertices = vertices.numpy()
                joints = joints.numpy()
        if joints_only:
            output = {"joints": joints}
        else:
            output = {
                "vertices": vertices,
                "joints": joints,
                "smpldata": smpldata,
            }
    elif value_from == "joints":
        if not keep_torch:
            # Handle both CPU and GPU tensors
            if joints.is_cuda:
                joints = joints.cpu().numpy()
            else:
                joints = joints.numpy()
        output = {"joints": joints}
    else:
        raise NotImplementedError
    return output

def extract_joints_guoh3dfeats_batch(x, keep_torch=False, **kwargs):
    """
    Batch version of extract_joints_guoh3dfeats.
    
    Args:
        x: Input features of shape (B, T, 263) where B is batch size, T is sequence length
        keep_torch: Whether to keep tensors as torch tensors
    
    Returns:
        Dictionary with output tensors having batch dimension
    """
    assert x.shape[-1] == 263, f"Expected last dim to be 263, got {x.shape[-1]}"
    
    # Check if input has batch dimension
    if len(x.shape) == 2:
        # Single sequence, use original function
        return extract_joints_guoh3dfeats(x, keep_torch=keep_torch, **kwargs)
    
    from tools.guofeats import guofeats_to_joints_batch
    
    joints = guofeats_to_joints_batch(x)
    if not keep_torch:
        # Handle both CPU and GPU tensors
        if joints.is_cuda:
            joints = joints.cpu().numpy()
        else:
            joints = joints.numpy()

    output = {"joints": joints}
    return output
