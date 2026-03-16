# adapted from HuMoR
import torch
import numpy as np
from tqdm import tqdm
import trimesh
from .parameters import colors, smpl_connections
from .mesh_viewer import MeshViewer

c2c = lambda tensor: tensor.detach().cpu().numpy()  # noqa


def _normalize_color_spec(color_spec):
    """Convert color inputs into a 3/4 channel float list."""
    if color_spec is None:
        return None
    if isinstance(color_spec, str):
        color_key = color_spec.lower()
        if color_key not in colors:
            raise ValueError(f"Unknown color name: {color_spec}")
        return colors[color_key]
    arr = np.asarray(color_spec, dtype=np.float32).reshape(-1)
    if arr.size not in (3, 4):
        raise ValueError(
            "Colors must have 3 (RGB) or 4 (RGBA) components, "
            f"got {arr.size}."
        )
    return arr.tolist()


def _create_body_mesh_seq(body, vertex_color, body_alpha):
    """Create a trimesh sequence for a single body with the desired color."""
    if vertex_color is None:
        vertex_color = colors["vertex"]
    color_arr = np.asarray(vertex_color, dtype=np.float32)
    base_color = np.tile(color_arr[:3], (body.v.size(1), 1))
    alpha = body_alpha
    if alpha is None and color_arr.size == 4:
        alpha = float(color_arr[3])
    if alpha is not None:
        alpha_column = np.ones((base_color.shape[0], 1), dtype=np.float32) * alpha
        vertex_colors = np.concatenate([base_color, alpha_column], axis=1)
    else:
        vertex_colors = base_color

    faces = c2c(body.f)
    return [
        trimesh.Trimesh(
            vertices=c2c(body.v[i]),
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        for i in range(body.v.size(0))
    ]


def _seq_to_numpy_list(seq):
    """Convert sequence data into a python list of numpy arrays."""
    if seq is None:
        return None
    if torch.is_tensor(seq):
        return [c2c(frame_data) for frame_data in seq]
    if isinstance(seq, np.ndarray):
        if seq.ndim == 2 and seq.shape[-1] == 3:
            return [seq]
        return [seq[i] for i in range(seq.shape[0])]
    if isinstance(seq, (list, tuple)):
        converted = []
        for frame_data in seq:
            if frame_data is None:
                converted.append(None)
            elif torch.is_tensor(frame_data):
                converted.append(c2c(frame_data))
            elif isinstance(frame_data, np.ndarray):
                converted.append(frame_data)
            else:
                converted.append(np.asarray(frame_data))
        return converted
    raise TypeError(f"Unsupported sequence type: {type(seq)}")


def _first_valid_point(sequence):
    """Find first valid XYZ point in a frame sequence."""
    if sequence is None:
        return None
    for frame_data in sequence:
        if frame_data is None:
            continue
        arr = np.asarray(frame_data)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)
        return arr[0]
    return None


def _viz_smpl_seq_base(
    pyrender,
    out_path,
    body_specs,
    #
    start=None,
    end=None,
    #
    imw=720,
    imh=720,
    point_size=2.75,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    #
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    points_seq=None,
    points_contact_seq=None,
    points_vel=None,
    line_seq=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    line_rad=0.01,
    skel_connections=smpl_connections,
    img_extn="png",
    ground_alpha=1.0,
    mask_seq=None,
    cam_offset=[0.0, 2.2, 0.9],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    line_color=[1.0, 0.0, 0.0],
    line_text_color=[1.0, 1.0, 0.0],
    show_line_length=True,
    line_text_scale=0.45,
    line_text_thickness=1,
    line_label_offset=0.03,
    line_length_decimals=2,
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
):
    if not body_specs:
        raise ValueError("At least one body specification is required for rendering.")

    body = body_specs[0]["body"]

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    need_body_meshes = render_body or any(
        spec.get("vtx_list") is not None for spec in body_specs
    )
    if need_body_meshes:
        seq_len = body.v.size(0)
        for spec in body_specs:
            cur_body = spec["body"]
            if cur_body.v.size(0) != seq_len:
                raise ValueError("All bodies must have the same number of frames.")
            spec["mesh_seq"] = _create_body_mesh_seq(
                cur_body, spec.get("vertex_color"), spec.get("body_alpha")
            )
    else:
        for spec in body_specs:
            spec["mesh_seq"] = None

    body_mesh_seq_list = [spec.get("mesh_seq") for spec in body_specs]

    if render_joints and joints_seq is None:
        # only body joints
        joints_seq = [c2c(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints:
        joints_seq = _seq_to_numpy_list(joints_seq)

    joints_vel = _seq_to_numpy_list(joints_vel)
    points_vel = _seq_to_numpy_list(points_vel)
    points_seq = _seq_to_numpy_list(points_seq)
    if points_contact_seq is not None:
        if torch.is_tensor(points_contact_seq):
            points_contact_seq = c2c(points_contact_seq)
        elif not isinstance(points_contact_seq, np.ndarray):
            points_contact_seq = np.asarray(points_contact_seq)
    line_seq = _seq_to_numpy_list(line_seq)

    mv = MeshViewer(
        pyrender,
        width=imw,
        height=imh,
        point_size=point_size,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        camera_intrinsics=camera_intrinsics,
        img_extn=img_extn,
        default_cam_offset=cam_offset,
        default_cam_rot=cam_rot,
    )
    if render_body and render_bodies_static is None:
        for idx, mesh_seq in enumerate(body_mesh_seq_list):
            if mesh_seq is None:
                continue
            mv.add_mesh_seq(
                mesh_seq, progress_bar=progress_bar if idx == 0 else None
            )
    elif render_body and render_bodies_static is not None:
        for mesh_seq in body_mesh_seq_list:
            if mesh_seq is None:
                continue
            mv.add_static_meshes(
                [
                    mesh_seq[i]
                    for i in range(len(mesh_seq))
                    if i % render_bodies_static == 0
                ]
            )
    if render_joints and render_skeleton:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            connections=skel_connections,
            connect_color=skel_color,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )
    elif render_joints:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )

    for spec in body_specs:
        vtx_list = spec.get("vtx_list")
        mesh_seq = spec.get("mesh_seq")
        if vtx_list is None or mesh_seq is None:
            continue
        mv.add_smpl_vtx_list_seq(
            mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015
        )

    if points_seq is not None:
        mv.add_point_seq(
            points_seq,
            color=point_color,
            radius=point_rad,
            contact_seq=points_contact_seq,
            vel=points_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )
    if line_seq is not None:
        mv.add_line_seq(
            line_seq,
            color=line_color,
            radius=line_rad,
            show_length_text=show_line_length,
            length_text_color=line_text_color,
            text_scale=line_text_scale,
            text_thickness=line_text_thickness,
            label_offset=line_label_offset,
            length_decimals=line_length_decimals,
            render_static=render_points_static,
        )

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body and body_mesh_seq_list and body_mesh_seq_list[0] is not None:
                xyz_orig = body_mesh_seq_list[0][0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = _first_valid_point(points_seq)
            elif line_seq is not None:
                xyz_orig = _first_valid_point(line_seq)

        mv.add_ground(
            ground_plane=ground_plane,
            xyz_orig=xyz_orig,
            color0=ground_color0,
            color1=ground_color1,
            alpha=ground_alpha,
        )

    mv.set_render_settings(
        out_path=out_path,
        wireframe=wireframe,
        RGBA=RGBA,
        single_frame=(
            render_points_static is not None or render_bodies_static is not None
        ),
    )  # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps, start=start, end=end, progress_bar=progress_bar)
    except RuntimeError as err:
        print("Could not render properly with the error: %s" % (str(err)))

    del mv


def viz_smpl_seq(
    pyrender,
    out_path,
    body,
    #
    start=None,
    end=None,
    #
    imw=720,
    imh=720,
    point_size=2.75,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    #
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    vtx_list=None,
    points_seq=None,
    points_contact_seq=None,
    points_vel=None,
    line_seq=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    line_rad=0.01,
    skel_connections=smpl_connections,
    img_extn="png",
    ground_alpha=1.0,
    body_alpha=None,
    mask_seq=None,
    cam_offset=[0.0, 2.2, 0.9],  # [0.0, 4.0, 1.25],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],  # [0.0, 0.0, 1.0],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    line_color=[1.0, 0.0, 0.0],
    line_text_color=[1.0, 1.0, 0.0],
    show_line_length=True,
    line_text_scale=0.45,
    line_text_thickness=1,
    line_label_offset=0.03,
    line_length_decimals=2,
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    vertex_color=colors["vertex"],
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
):
    """
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - line_seq : list of torch/numpy tensors, each frame is (2,3) or (num_lines,2,3)
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    """

    resolved_color = _normalize_color_spec(vertex_color)
    if resolved_color is None:
        resolved_color = colors["vertex"]

    body_specs = [
        {
            "body": body,
            "vertex_color": resolved_color,
            "body_alpha": body_alpha,
            "vtx_list": vtx_list,
        }
    ]

    _viz_smpl_seq_base(
        pyrender,
        out_path,
        body_specs,
        start=start,
        end=end,
        imw=imw,
        imh=imh,
        point_size=point_size,
        fps=fps,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        progress_bar=progress_bar,
        contacts=contacts,
        render_body=render_body,
        render_joints=render_joints,
        render_skeleton=render_skeleton,
        render_ground=render_ground,
        ground_plane=ground_plane,
        wireframe=wireframe,
        RGBA=RGBA,
        joints_seq=joints_seq,
        joints_vel=joints_vel,
        points_seq=points_seq,
        points_contact_seq=points_contact_seq,
        points_vel=points_vel,
        line_seq=line_seq,
        static_meshes=static_meshes,
        camera_intrinsics=camera_intrinsics,
        img_seq=img_seq,
        point_rad=point_rad,
        line_rad=line_rad,
        skel_connections=skel_connections,
        img_extn=img_extn,
        ground_alpha=ground_alpha,
        mask_seq=mask_seq,
        cam_offset=cam_offset,
        ground_color0=ground_color0,
        ground_color1=ground_color1,
        skel_color=skel_color,
        joint_rad=joint_rad,
        point_color=point_color,
        line_color=line_color,
        line_text_color=line_text_color,
        show_line_length=show_line_length,
        line_text_scale=line_text_scale,
        line_text_thickness=line_text_thickness,
        line_label_offset=line_label_offset,
        line_length_decimals=line_length_decimals,
        joint_color=joint_color,
        contact_color=contact_color,
        render_bodies_static=render_bodies_static,
        render_points_static=render_points_static,
        cam_rot=cam_rot,
    )


def viz_smpl_seq_multi(
    pyrender,
    out_path,
    bodies,
    #
    start=None,
    end=None,
    #
    imw=720,
    imh=720,
    point_size=2.75,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    #
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    vtx_lists=None,
    points_seq=None,
    points_contact_seq=None,
    points_vel=None,
    line_seq=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    line_rad=0.01,
    skel_connections=smpl_connections,
    img_extn="png",
    ground_alpha=1.0,
    body_alpha=None,
    mask_seq=None,
    cam_offset=[0.0, 2.2, 0.9],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    line_color=[1.0, 0.0, 0.0],
    line_text_color=[1.0, 1.0, 0.0],
    show_line_length=True,
    line_text_scale=0.45,
    line_text_thickness=1,
    line_label_offset=0.03,
    line_length_decimals=2,
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    vertex_color=colors["vertex"],
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
    body_colors=None,
    body_alphas=None,
):
    """Render multiple SMPL bodies into the same sequence using per-body colors."""

    if not isinstance(bodies, (list, tuple)):
        raise TypeError("'bodies' must be a list or tuple of SMPL outputs.")
    if len(bodies) == 0:
        raise ValueError("'bodies' must contain at least one element.")
    if vtx_lists is not None and len(vtx_lists) != len(bodies):
        raise ValueError("'vtx_lists' must match the number of bodies.")
    if body_alphas is not None and len(body_alphas) != len(bodies):
        raise ValueError("'body_alphas' must match the number of bodies.")

    default_color = _normalize_color_spec(vertex_color)
    if default_color is None:
        default_color = colors["vertex"]

    palette = [default_color]
    for color_name in ["red", "green", "blue", "yellow", "purple", "cyan", "orange", "brown"]:
        palette.append(_normalize_color_spec(color_name))

    body_specs = []
    for idx, body in enumerate(bodies):
        if body_colors is not None and idx < len(body_colors):
            color = _normalize_color_spec(body_colors[idx])
        else:
            color = None
        if color is None:
            color = palette[idx % len(palette)]

        alpha = body_alpha
        if body_alphas is not None:
            alpha = body_alphas[idx]

        vtx_list = None
        if vtx_lists is not None:
            vtx_list = vtx_lists[idx]

        body_specs.append(
            {
                "body": body,
                "vertex_color": color,
                "body_alpha": alpha,
                "vtx_list": vtx_list,
            }
        )

    _viz_smpl_seq_base(
        pyrender,
        out_path,
        body_specs,
        start=start,
        end=end,
        imw=imw,
        imh=imh,
        point_size=point_size,
        fps=fps,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        progress_bar=progress_bar,
        contacts=contacts,
        render_body=render_body,
        render_joints=render_joints,
        render_skeleton=render_skeleton,
        render_ground=render_ground,
        ground_plane=ground_plane,
        wireframe=wireframe,
        RGBA=RGBA,
        joints_seq=joints_seq,
        joints_vel=joints_vel,
        points_seq=points_seq,
        points_contact_seq=points_contact_seq,
        points_vel=points_vel,
        line_seq=line_seq,
        static_meshes=static_meshes,
        camera_intrinsics=camera_intrinsics,
        img_seq=img_seq,
        point_rad=point_rad,
        line_rad=line_rad,
        skel_connections=skel_connections,
        img_extn=img_extn,
        ground_alpha=ground_alpha,
        mask_seq=mask_seq,
        cam_offset=cam_offset,
        ground_color0=ground_color0,
        ground_color1=ground_color1,
        skel_color=skel_color,
        joint_rad=joint_rad,
        point_color=point_color,
        line_color=line_color,
        line_text_color=line_text_color,
        show_line_length=show_line_length,
        line_text_scale=line_text_scale,
        line_text_thickness=line_text_thickness,
        line_label_offset=line_label_offset,
        line_length_decimals=line_length_decimals,
        joint_color=joint_color,
        contact_color=contact_color,
        render_bodies_static=render_bodies_static,
        render_points_static=render_points_static,
        cam_rot=cam_rot,
    )
