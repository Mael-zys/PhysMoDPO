# adapted from HuMoR
import os
import time

# import math
from tqdm import tqdm
import numpy as np
import trimesh

# import pyrender
import sys
import cv2

from .parameters import colors

# import pyglet

__all__ = ["MeshViewer"]

COMPRESS_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


def pause_play_callback(pyrender_viewer, mesh_viewer):
    mesh_viewer.is_paused = not mesh_viewer.is_paused


def step_callback(pyrender_viewer, mesh_viewer, step_size):
    mesh_viewer.animation_frame_idx = (
        mesh_viewer.animation_frame_idx + step_size
    ) % mesh_viewer.animation_len


class MeshViewer(object):
    def __init__(
        self,
        pyrender,
        width=1200,
        height=800,
        point_size=2.75,
        use_offscreen=False,
        follow_camera=False,
        camera_intrinsics=None,
        img_extn="png",
        default_cam_offset=[0.0, 4.0, 1.25],
        default_cam_rot=None,
    ):
        super().__init__()

        self.pyrender = pyrender
        self.use_offscreen = use_offscreen
        self.follow_camera = follow_camera
        # render settings for offscreen
        self.render_wireframe = False
        self.render_RGBA = False
        self.render_path = "./render_out"
        self.img_extn = img_extn

        # mesh sequences to animate
        self.animated_seqs = []  # the actual sequence of pyrender meshes
        self.animated_seqs_type = []
        self.animated_nodes = []  # the nodes corresponding to each sequence
        self.light_nodes = []
        # they must all be the same length (set based on first given sequence)
        self.animation_len = -1
        # current index in the animation sequence
        self.animation_frame_idx = 0
        # track render time to keep steady framerate
        self.animation_render_time = time.time()
        # background image sequence
        self.img_seq = None
        self.cur_bg_img = None
        # person mask sequence
        self.mask_seq = None
        self.cur_mask = None
        # world-space text labels rendered after rasterization
        self.line_label_tracks = []

        self.single_frame = False

        self.mat_constructor = self.pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = self.pyrender.Mesh.from_trimesh

        self.scene = self.pyrender.Scene(
            bg_color=colors["white"], ambient_light=(0.3, 0.3, 0.3)
        )

        self.default_cam_offset = np.array(default_cam_offset)
        self.default_cam_rot = np.array(default_cam_rot)

        self.default_cam_pose = np.eye(4)
        if default_cam_rot is None:
            self.default_cam_pose = trimesh.transformations.rotation_matrix(
                np.radians(180), (0, 0, 1)
            )
            self.default_cam_pose = np.dot(
                trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)),
                self.default_cam_pose,
            )
        else:
            self.default_cam_pose[:3, :3] = self.default_cam_rot
        self.default_cam_pose[:3, 3] = self.default_cam_offset

        self.use_intrins = False
        if camera_intrinsics is None:
            pc = self.pyrender.PerspectiveCamera(
                yfov=np.pi / 3.0, aspectRatio=float(width) / height
            )
            camera_pose = self.get_init_cam_pose()
            self.camera_node = self.scene.add(pc, pose=camera_pose, name="pc-camera")

            light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            self.scene.add(light, pose=self.default_cam_pose)
        else:
            self.use_intrins = True
            fx, fy, cx, cy = camera_intrinsics
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera = self.pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
            self.camera_node = self.scene.add(
                camera, pose=camera_pose, name="pc-camera"
            )

            light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            self.scene.add(light, pose=camera_pose)

            self.set_background_color([1.0, 1.0, 1.0, 0.0])

        self.figsize = (width, height)

        # key callbacks
        self.is_paused = False
        registered_keys = dict()
        registered_keys["p"] = (pause_play_callback, [self])
        registered_keys["."] = (step_callback, [self, 1])
        registered_keys[","] = (step_callback, [self, -1])

        if self.use_offscreen:
            self.viewer = self.pyrender.OffscreenRenderer(
                *self.figsize, point_size=point_size
            )
            self.use_raymond_lighting(3.5)
        else:
            self.viewer = self.pyrender.Viewer(
                self.scene,
                use_raymond_lighting=(not camera_intrinsics),
                viewport_size=self.figsize,
                cull_faces=False,
                run_in_thread=True,
                registered_keys=registered_keys,
            )

    def get_init_cam_pose(self):
        camera_pose = self.default_cam_pose.copy()
        return camera_pose

    def set_background_color(self, color=colors["white"]):
        self.scene.bg_color = color

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_meshes(self, meshes, group_name="static"):
        for node in self.scene.get_nodes():
            if node.name is not None and "%s-mesh" % group_name in node.name:
                self.scene.remove_node(node)

        for mid, mesh in enumerate(meshes):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = self.pyrender.Mesh.from_trimesh(mesh.copy())
            self.acquire_render_lock()
            self.scene.add(mesh, "%s-mesh-%2d" % (group_name, mid))
            self.release_render_lock()

    def set_static_meshes(self, meshes):
        self.set_meshes(meshes, group_name="static")

    def add_static_meshes(self, meshes):
        for mid, mesh in enumerate(meshes):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = self.pyrender.Mesh.from_trimesh(mesh.copy())
            self.acquire_render_lock()
            self.scene.add(mesh, "%s-mesh-%2d" % ("staticadd", mid))
            self.release_render_lock()

    def add_smpl_vtx_list_seq(
        self, body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015
    ):
        vtx_point_seq = []
        for mesh in body_mesh_seq:
            vtx_point_seq.append(mesh.vertices[vtx_list])
        self.add_point_seq(vtx_point_seq, color=color, radius=radius)

    def set_img_seq(self, img_seq):
        """
        np array of BG images to be rendered in background.
        """
        if not self.use_offscreen:
            print("Cannot render background image if not rendering offscreen")
            return
        # ensure same length as other sequences
        cur_seq_len = len(img_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print(
                    "Unexpected imgage sequence length, all sequences must be the same length!"
                )
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: imge sequence is length 0!")
                return

        self.img_seq = img_seq
        # must have alpha to render background
        self.set_render_settings(RGBA=True)

    def set_mask_seq(self, mask_seq):
        """
        np array of masked images to be rendered in background.
        """
        if not self.use_offscreen:
            print("Cannot render background image if not rendering offscreen")
            return
        # ensure same length as other sequences
        cur_seq_len = len(mask_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print(
                    "Unexpected imgage sequence length, all sequences must be the same length!"
                )
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: imge sequence is length 0!")
                return

        self.mask_seq = mask_seq

    def _normalize_line_frame(self, line_frame):
        if line_frame is None:
            return np.zeros((0, 2, 3), dtype=np.float32)
        arr = np.asarray(line_frame, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 2, 3), dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (2, 3):
            return arr.reshape(1, 2, 3)
        if arr.ndim == 3 and arr.shape[1:] == (2, 3):
            return arr
        raise ValueError(
            f"Each line frame must have shape (2,3) or (N,2,3), got {arr.shape}."
        )

    def _rgb_to_bgr255(self, rgb):
        color = np.asarray(rgb, dtype=np.float32).reshape(-1)
        if color.size < 3:
            return (255, 255, 255)
        color = color[:3]
        if np.max(color) <= 1.0:
            color = color * 255.0
        color = np.clip(color, 0.0, 255.0).astype(np.uint8)
        return (int(color[2]), int(color[1]), int(color[0]))

    def _project_world_to_pixel(self, point_world):
        cam_nodes = list(self.scene.get_nodes(name="pc-camera"))
        if len(cam_nodes) == 0:
            return None

        camera_node = cam_nodes[0]
        camera_pose = camera_node.matrix
        if camera_pose is None:
            return None

        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = point
        point_cam = np.linalg.inv(camera_pose).dot(point_h)

        proj = camera_node.camera.get_projection_matrix(
            self.figsize[0], self.figsize[1]
        )
        clip = proj.dot(point_cam)
        if np.abs(clip[3]) < 1e-8:
            return None
        ndc = clip[:3] / clip[3]
        if not np.isfinite(ndc).all():
            return None
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None

        px = (ndc[0] * 0.5 + 0.5) * (self.figsize[0] - 1)
        py = (1.0 - (ndc[1] * 0.5 + 0.5)) * (self.figsize[1] - 1)
        return int(round(px)), int(round(py))

    def _draw_line_labels(self, output_img):
        if len(self.line_label_tracks) == 0:
            return output_img
        if output_img.ndim != 3 or output_img.shape[2] < 3:
            return output_img

        has_alpha = output_img.shape[2] == 4
        if has_alpha:
            # cv2.putText requires a cv::Mat-compatible contiguous layout.
            draw_img = np.ascontiguousarray(output_img[:, :, :3])
        else:
            draw_img = (
                output_img
                if output_img.flags.c_contiguous
                else np.ascontiguousarray(output_img)
            )

        img_h, img_w = draw_img.shape[:2]
        for track in self.line_label_tracks:
            labels_seq = track["labels_seq"]
            if self.animation_frame_idx >= len(labels_seq):
                continue

            for item in labels_seq[self.animation_frame_idx]:
                pixel = self._project_world_to_pixel(item["position"])
                if pixel is None:
                    continue
                x, y = pixel
                if x < 0 or x >= img_w or y < 0 or y >= img_h:
                    continue

                org = (
                    min(max(0, x + 4), img_w - 1),
                    min(max(0, y - 4), img_h - 1),
                )
                cv2.putText(
                    draw_img,
                    item["text"],
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    track["text_scale"],
                    track["text_color_bgr"],
                    track["text_thickness"],
                    lineType=cv2.LINE_AA,
                )
        if has_alpha:
            output_img = output_img.copy()
            output_img[:, :, :3] = draw_img
            return output_img
        if draw_img is not output_img:
            return draw_img
        return output_img

    def add_line_seq(
        self,
        line_seq,
        color=[1.0, 0.0, 0.0],
        radius=0.01,
        show_length_text=True,
        length_text_color=[1.0, 1.0, 0.0],
        text_scale=0.45,
        text_thickness=1,
        label_offset=0.03,
        length_decimals=2,
        render_static=None,
    ):
        """
        Add a sequence of line segments.

        - line_seq : list, each frame is (2,3) for one line or (N,2,3) for multiple lines.
        """
        cur_seq_len = len(line_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print(
                    "Unexpected sequence length, all sequences must be the same length!"
                )
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: line sequence is length 0!")
                return

        pyrender_line_seq = []
        labels_seq = []
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for pid, line_frame in enumerate(line_seq):
            if pid % 200 == 0:
                print("Caching pyrender lines mesh %d/%d..." % (pid, len(line_seq)))

            frame_lines = self._normalize_line_frame(line_frame)
            cyl_mesh_list = []
            frame_labels = []
            for line_pts in frame_lines:
                p1 = line_pts[0]
                p2 = line_pts[1]
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-8:
                    continue

                segment = np.array([p1, p2])
                cyl_mesh = trimesh.creation.cylinder(
                    radius=radius, height=None, segment=segment
                )
                cyl_mesh.visual.vertex_colors = color
                cyl_mesh_list.append(cyl_mesh.copy())

                if show_length_text:
                    side_dir = np.cross(direction, world_up)
                    side_norm = np.linalg.norm(side_dir)
                    if side_norm < 1e-8:
                        side_dir = np.cross(direction, world_y)
                        side_norm = np.linalg.norm(side_dir)
                    if side_norm < 1e-8:
                        side_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        side_dir = side_dir / side_norm

                    label_pos = 0.5 * (p1 + p2) + side_dir * label_offset
                    frame_labels.append(
                        {
                            "position": label_pos,
                            "text": f"{length:.{length_decimals}f}",
                        }
                    )

            if len(cyl_mesh_list) > 0:
                m = self.pyrender.Mesh.from_trimesh(cyl_mesh_list)
            else:
                sm = trimesh.creation.uv_sphere(radius=max(radius * 0.05, 1e-4))
                sm.visual.vertex_colors = color
                tfs = np.eye(4).reshape((1, 4, 4))
                tfs[0, :3, 3] = np.array([0, 0, 30.0], dtype=np.float32)
                m = self.pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs)
            pyrender_line_seq.append(m)
            labels_seq.append(frame_labels)

        if render_static is None:
            self.add_pyrender_mesh_seq(pyrender_line_seq, seq_type="line")
        else:
            self.add_static_meshes(
                [
                    pyrender_line_seq[i]
                    for i in range(len(pyrender_line_seq))
                    if i % render_static == 0
                ]
            )

        if show_length_text:
            self.line_label_tracks.append(
                {
                    "labels_seq": labels_seq,
                    "text_color_bgr": self._rgb_to_bgr255(length_text_color),
                    "text_scale": text_scale,
                    "text_thickness": text_thickness,
                }
            )

    def add_point_seq(
        self,
        point_seq,
        color=[1.0, 0.0, 0.0],
        radius=0.015,
        contact_seq=None,
        contact_color=[0.0, 1.0, 0.0],
        connections=None,
        connect_color=[0.0, 0.0, 1.0],
        vel=None,
        render_static=None,
    ):
        """
        Add a sequence of points that will be visualized as spheres.

        - points : List of Nx3 numpy arrays of point locations to visualize as sequence.
        - color : list of 3 RGB values
        - radius : radius of each point
        - contact_seq : an array of num_frames x num_points indicatin "contacts" i.e. points that should be colored
                        differently at different time steps.
        - connections : array of point index pairs, draws a cylinder between each pair to create skeleton
        - vel : list of Nx3 numpy arrays for the velocities of corresponding sequence points
        """
        # ensure same length as other sequences
        cur_seq_len = len(point_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print(
                    "Unexpected sequence length, all sequences must be the same length!"
                )
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: points sequence is length 0!")
                return

        num_joints = point_seq[0].shape[0]
        if contact_seq is not None and contact_seq.shape[1] != num_joints:
            print(num_joints)
            print(contact_seq.shape)
            print(
                "Contact sequence must have the same number of points as the input joints!"
            )
            return
        if contact_seq is not None and contact_seq.shape[0] != cur_seq_len:
            print(
                "Contact sequence must have the same number of frames as the input sequence!"
            )
            return

        # add skeleton
        if connections is not None:
            pyrender_skeleton_seq = []
            for pid, points in enumerate(point_seq):
                if pid % 200 == 0:
                    print(
                        "Caching pyrender connections mesh %d/%d..."
                        % (pid, len(point_seq))
                    )

                cyl_mesh_list = []
                for point_pair in connections:
                    # print(point_pair)
                    p1 = points[point_pair[0]]
                    p2 = points[point_pair[1]]
                    if np.linalg.norm(p1 - p2) < 1e-6:
                        segment = np.array([[-1.0, -1.0, -1.0], [-1.01, -1.01, -1.01]])
                    else:
                        segment = np.array([p1, p2])
                    # print(segment)

                    cyl_mesh = trimesh.creation.cylinder(
                        radius * 0.35, height=None, segment=segment
                    )
                    cyl_mesh.visual.vertex_colors = connect_color
                    cyl_mesh_list.append(cyl_mesh.copy())

                # combine
                m = self.pyrender.Mesh.from_trimesh(cyl_mesh_list)
                pyrender_skeleton_seq.append(m)

            if render_static is None:
                self.add_pyrender_mesh_seq(pyrender_skeleton_seq)
            else:
                self.add_static_meshes(
                    [
                        pyrender_skeleton_seq[i]
                        for i in range(len(pyrender_skeleton_seq))
                        if i % render_static == 0
                    ]
                )

        # add velocities
        if vel is not None:
            print("Caching pyrender velocities mesh...")
            pyrender_vel_seq = []

            point_vel_pairs = zip(point_seq, vel)
            for pid, point_vel_pair in enumerate(point_vel_pairs):
                cur_point_seq, cur_vel_seq = point_vel_pair

                cyl_mesh_list = []
                for cur_point, cur_vel in zip(cur_point_seq, cur_vel_seq):
                    p1 = cur_point
                    p2 = cur_point + cur_vel * 0.1
                    segment = np.array([p1, p2])
                    if np.linalg.norm(p1 - p2) < 1e-6:
                        continue
                    cyl_mesh = trimesh.creation.cylinder(
                        radius * 0.1, height=None, segment=segment
                    )
                    cyl_mesh.visual.vertex_colors = [0.0, 0.0, 1.0]
                    cyl_mesh_list.append(cyl_mesh.copy())

                # combine
                m = self.pyrender.Mesh.from_trimesh(cyl_mesh_list)
                pyrender_vel_seq.append(m)

            if render_static is None:
                self.add_pyrender_mesh_seq(pyrender_vel_seq)
            else:
                self.add_static_meshes(
                    [
                        pyrender_vel_seq[i]
                        for i in range(len(pyrender_vel_seq))
                        if i % render_static == 0
                    ]
                )

        # create spheres with trimesh
        if contact_seq is None:
            contact_seq = [
                np.zeros((point_seq[t].shape[0])) for t in range(cur_seq_len)
            ]
        pyrender_non_contact_point_seq = []
        pyrender_contact_point_seq = []
        for pid, points in enumerate(point_seq):
            if pid % 200 == 0:
                print("Caching pyrender points mesh %d/%d..." % (pid, len(point_seq)))

            # first non-contacting points
            if len(color) > 3:
                pyrender_non_contact_point_seq.append(
                    self.pyrender.Mesh.from_points(points, color[pid])
                )
            else:
                sm = trimesh.creation.uv_sphere(radius=radius)
                sm.visual.vertex_colors = color
                non_contact_points = points[contact_seq[pid] == 0]
                if len(non_contact_points) > 0:
                    tfs = np.tile(np.eye(4), (len(non_contact_points), 1, 1))
                    tfs[:, :3, 3] = non_contact_points.copy()
                    m = self.pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs)
                    pyrender_non_contact_point_seq.append(m)
                else:
                    tfs = np.eye(4).reshape((1, 4, 4))
                    tfs[0, :3, 3] = np.array([0, 0, 30.0])
                    pyrender_non_contact_point_seq.append(
                        self.pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs)
                    )
            # then contacting points
            sm = trimesh.creation.uv_sphere(radius=radius)
            sm.visual.vertex_colors = contact_color
            contact_points = points[contact_seq[pid] == 1]
            if len(contact_points) > 0:
                tfs = np.tile(np.eye(4), (len(contact_points), 1, 1))
                tfs[:, :3, 3] = contact_points.copy()
                m = self.pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs)
                pyrender_contact_point_seq.append(m)
            else:
                tfs = np.eye(4).reshape((1, 4, 4))
                tfs[0, :3, 3] = np.array([0, 0, 30.0])
                pyrender_contact_point_seq.append(
                    self.pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs)
                )

        if len(pyrender_non_contact_point_seq) > 0:
            if render_static is None:
                self.add_pyrender_mesh_seq(
                    pyrender_non_contact_point_seq, seq_type="point"
                )
            else:
                self.add_static_meshes(
                    [
                        pyrender_non_contact_point_seq[i]
                        for i in range(len(pyrender_non_contact_point_seq))
                        if i % render_static == 0
                    ]
                )
        if len(pyrender_contact_point_seq) > 0:
            if render_static is None:
                self.add_pyrender_mesh_seq(pyrender_contact_point_seq, seq_type="point")
            else:
                self.add_static_meshes(
                    [
                        pyrender_contact_point_seq[i]
                        for i in range(len(pyrender_contact_point_seq))
                        if i % render_static == 0
                    ]
                )

    def add_mesh_seq(self, mesh_seq, progress_bar=tqdm):
        """
        Add a sequence of trimeshes to render.

        - meshes : List of trimesh.trimesh objects giving each frame of the sequence.
        """

        # ensure same length as other sequences
        cur_seq_len = len(mesh_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print(
                    "Unexpected sequence length, all sequences must be the same length!"
                )
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: mesh sequence is length 0!")
                return

        # print("Adding mesh sequence with %d frames..." % (cur_seq_len))

        # create sequence of pyrender meshes and save
        pyrender_mesh_seq = []

        iterator = enumerate(mesh_seq)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Import meshes in pyrender")

        for mid, mesh in iterator:
            if isinstance(mesh, trimesh.Trimesh):
                mesh = self.pyrender.Mesh.from_trimesh(mesh.copy())
                pyrender_mesh_seq.append(mesh)
            else:
                print("Meshes must be from trimesh!")
                return

        self.add_pyrender_mesh_seq(pyrender_mesh_seq, seq_type="mesh")

    def add_pyrender_mesh_seq(self, pyrender_mesh_seq, seq_type="default"):
        # add to the list of sequences to render
        seq_id = len(self.animated_seqs)
        self.animated_seqs.append(pyrender_mesh_seq)
        self.animated_seqs_type.append(seq_type)

        # create the corresponding node in the scene
        self.acquire_render_lock()
        anim_node = self.scene.add(pyrender_mesh_seq[0], "anim-mesh-%2d" % (seq_id))
        self.animated_nodes.append(anim_node)
        self.release_render_lock()

    def add_ground(
        self,
        ground_plane=None,
        length=25.0,
        color0=[0.8, 0.9, 0.9],
        color1=[0.6, 0.7, 0.7],
        tile_width=0.5,
        xyz_orig=None,
        alpha=1.0,
    ):
        """
        If ground_plane is none just places at origin with +z up.
        If ground_plane is given (a, b, c, d) where a,b,c is the normal, then this is rendered. To more accurately place the floor
        provid an xyz_orig = [x,y,z] that we expect to be near the point of focus.
        """
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        # make checkerboard
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices = []
        faces = []
        face_colors = []
        for i in range(num_rows):
            for j in range(num_cols):
                start_loc = [-radius + j * tile_width, radius - i * tile_width]
                cur_verts = np.array(
                    [
                        [start_loc[0], start_loc[1], 0.0],
                        [start_loc[0], start_loc[1] - tile_width, 0.0],
                        [start_loc[0] + tile_width, start_loc[1] - tile_width, 0.0],
                        [start_loc[0] + tile_width, start_loc[1], 0.0],
                    ]
                )
                cur_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=int)
                cur_faces += 4 * (
                    i * num_cols + j
                )  # the number of previously added verts
                use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
                cur_color = color0 if use_color0 else color1
                cur_face_colors = np.array([cur_color, cur_color])

                vertices.append(cur_verts)
                faces.append(cur_faces)
                face_colors.append(cur_face_colors)

        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        face_colors = np.concatenate(face_colors, axis=0)

        if ground_plane is not None:
            # compute transform between identity floor and passed in floor
            a, b, c, d = ground_plane
            # rotation
            old_normal = np.array([0.0, 0.0, 1.0])
            new_normal = np.array([a, b, c])
            new_normal = new_normal / np.linalg.norm(new_normal)
            v = np.cross(old_normal, new_normal)
            ang_sin = np.linalg.norm(v)
            ang_cos = np.dot(old_normal, new_normal)
            skew_v = np.array(
                [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
            )
            R = (
                np.eye(3)
                + skew_v
                + np.matmul(skew_v, skew_v) * ((1.0 - ang_cos) / (ang_sin**2))
            )
            # translation
            # project point of focus onto plane
            if xyz_orig is None:
                xyz_orig = np.array([0.0, 0.0, 0.0])
            # project origin onto plane
            plane_normal = np.array([a, b, c])
            plane_off = d
            direction = -plane_normal
            s = (plane_off - np.dot(plane_normal, xyz_orig)) / np.dot(
                plane_normal, direction
            )
            itsct_pt = xyz_orig + s * direction
            t = itsct_pt

            # transform floor
            vertices = np.dot(R, vertices.T).T + t.reshape((1, 3))

        ground_tri = trimesh.creation.Trimesh(
            vertices=vertices, faces=faces, face_colors=face_colors, process=False
        )
        ground_mesh = self.pyrender.Mesh.from_trimesh(ground_tri, smooth=False)

        self.acquire_render_lock()
        anim_node = self.scene.add(ground_mesh, "ground-mesh")
        self.release_render_lock()

        # update light nodes (if using raymond lighting) to be in this frame
        if ground_plane is not None:
            for lnode in self.light_nodes:
                new_lpose = np.eye(4)
                new_lrot = np.dot(R, lnode.matrix[:3, :3])
                new_ltrans = t
                new_lpose[:3, :3] = new_lrot
                new_lpose[:3, 3] = new_ltrans
                self.acquire_render_lock()
                self.scene.set_pose(lnode, new_lpose)
                self.release_render_lock()

    def update_frame(self):
        """
        Update frame to show the current self.animation_frame_idx
        """
        for seq_idx in range(len(self.animated_seqs)):
            cur_mesh = self.animated_seqs[seq_idx][self.animation_frame_idx]
            # render the current frame of eqch sequence
            self.acquire_render_lock()

            # replace the old mesh
            anim_node = list(self.scene.get_nodes(name="anim-mesh-%2d" % (seq_idx)))
            anim_node = anim_node[0]
            anim_node.mesh = cur_mesh
            # update camera pc-camera
            if (
                self.follow_camera and not self.use_intrins
            ):  # don't want to reset if we're going from camera view
                if self.animated_seqs_type[seq_idx] == "mesh":
                    cam_node = list(self.scene.get_nodes(name="pc-camera"))
                    cam_node = cam_node[0]
                    mesh_mean = np.mean(cur_mesh.primitives[0].positions, axis=0)
                    camera_pose = self.get_init_cam_pose()
                    camera_pose[:3, 3] = camera_pose[:3, 3] + np.array(
                        [mesh_mean[0], mesh_mean[1], 0.0]
                    )
                    self.scene.set_pose(cam_node, camera_pose)

            self.release_render_lock()

        # update background img
        if self.img_seq is not None:
            self.acquire_render_lock()
            self.cur_bg_img = self.img_seq[self.animation_frame_idx]
            self.release_render_lock

        # update mask
        if self.mask_seq is not None:
            self.acquire_render_lock()
            self.cur_mask = self.mask_seq[self.animation_frame_idx]
            self.release_render_lock

    def animate(self, fps=30, start=None, end=None, progress_bar=tqdm):
        """
        Starts animating any given mesh sequences. This should be called last after adding
        all desired components to the scene as it is a blocking operation and will run
        until the user exits (or the full video is rendered if offline).
        """
        if not self.use_offscreen:
            print("=================================")
            print("VIEWER CONTROLS")
            print("p - pause/play")
            print('"," and "." - step back/forward one frame')
            print("w - wireframe")
            print("h - render shadows")
            print("q - quit")
            print("=================================")

        # print("Animating...")
        # frame_dur = 1.0 / float(fps)

        # set up init frame
        self.update_frame()

        # only support offscreen in this script
        assert self.use_offscreen

        assert self.animation_frame_idx == 0

        iterator = range(self.animation_len)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="SMPL rendering")

        for frame_idx in iterator:
            self.animation_frame_idx = frame_idx
            # render frame
            if not os.path.exists(self.render_path):
                os.mkdir(self.render_path)
                # print("Rendering frames to %s!" % (self.render_path))
            cur_file_path = os.path.join(
                self.render_path,
                "frame_%08d.%s" % (self.animation_frame_idx, self.img_extn),
            )
            if start is None or end is None:
                self.save_snapshot(cur_file_path)
            else:
                # only do it for a crop
                if start <= self.animation_frame_idx < end:
                    self.save_snapshot(cur_file_path)

            if self.animation_frame_idx + 1 >= self.animation_len:
                continue  # last iteration anyway

            self.animation_render_time = time.time()
            # if self.is_paused:
            # self.update_frame()  # just in case there's a single frame update
            # continue

            self.animation_frame_idx = self.animation_frame_idx + 1
            # % self.animation_len
            self.update_frame()

            # if self.single_frame:
            # break

        self.animation_frame_idx = 0
        return True

    def _add_raymond_light(self):
        DirectionalLight = self.pyrender.light.DirectionalLight
        Node = self.pyrender.node.Node
        # from pyrender.light import DirectionalLight
        # from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes

    def use_raymond_lighting(self, intensity=1.0):
        if not self.use_offscreen:
            sys.stderr.write("Interactive viewer already uses raymond lighting!\n")
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)  # , parent_node=pc)

            self.light_nodes.append(n)

    def set_render_settings(
        self, wireframe=None, RGBA=None, out_path=None, single_frame=None
    ):
        if wireframe is not None and wireframe == True:
            self.render_wireframe = True
        if RGBA is not None and RGBA == True:
            self.render_RGBA = True
        if out_path is not None:
            self.render_path = out_path
        if single_frame is not None:
            self.single_frame = single_frame

    def render(self):
        RenderFlags = self.pyrender.constants.RenderFlags
        # from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if self.render_RGBA:
            flags |= RenderFlags.RGBA
        if self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        output_img = color_img
        if self.cur_bg_img is not None:
            color_img = color_img.astype(np.float32) / 255.0
            person_mask = None
            if self.cur_mask is not None:
                person_mask = self.cur_mask[:, :, np.newaxis]
                color_img = color_img * (1.0 - person_mask)
            valid_mask = (color_img[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = self.cur_bg_img
            if color_img.shape[2] == 4:
                output_img = (
                    color_img[:, :, :-1] * color_img[:, :, 3:]
                    + (1.0 - color_img[:, :, 3:]) * input_img
                )
            else:
                output_img = (
                    color_img[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img
                )

            output_img = (output_img * 255.0).astype(np.uint8)

        output_img = self._draw_line_labels(output_img)
        return output_img

    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write(
                "Currently saving snapshots only works with off-screen renderer!\n"
            )
            return
        color_img = self.render()
        if color_img.shape[-1] == 4:
            img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGRA)
        else:
            img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, img_bgr, COMPRESS_PARAMS)

    def acquire_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.acquire()

    def release_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.release()
