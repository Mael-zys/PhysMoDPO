import numpy as np
import torch
from .humor_render_tools.tools import viz_smpl_seq, viz_smpl_seq_multi
from smplx.utils import Struct
from .video import Video
import os
from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import Process

# os.environ["PYOPENGL_PLATFORM"] = "egl"


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
FACE_PATH = os.path.join(THIS_FOLDER, "humor_render_tools/smplh.faces")
FACES = torch.from_numpy(np.int32(np.load(FACE_PATH)))


class HumorRenderer:
    def __init__(self, fps=20.0, **kwargs):
        self.kwargs = kwargs
        self.fps = fps

    def __call__(self, vertices, output, **kwargs):
        params = {**self.kwargs, **kwargs}
        fps = self.fps
        if "fps" in params:
            fps = params.pop("fps")
        render(vertices, output, fps, **params)


def _shift_sequence_z(sequence, z_offset):
    """Shift z value for frame-wise sequences like points_seq/line_seq."""
    if sequence is None:
        return sequence

    if isinstance(sequence, (list, tuple)):
        adjusted_sequence = []
        for frame_data in sequence:
            if frame_data is None:
                adjusted_sequence.append(frame_data)
                continue
            if torch.is_tensor(frame_data):
                adjusted_frame = frame_data.clone()
            else:
                adjusted_frame = np.array(frame_data, copy=True)
            if adjusted_frame.size > 0:
                adjusted_frame[..., 2] -= z_offset
            adjusted_sequence.append(adjusted_frame)
        return adjusted_sequence

    if torch.is_tensor(sequence):
        sequence[..., 2] -= z_offset
        return sequence

    if isinstance(sequence, np.ndarray):
        if sequence.dtype == object:
            adjusted_sequence = []
            for frame_data in sequence:
                if frame_data is None:
                    adjusted_sequence.append(frame_data)
                    continue
                adjusted_frame = np.array(frame_data, copy=True)
                if adjusted_frame.size > 0:
                    adjusted_frame[..., 2] -= z_offset
                adjusted_sequence.append(adjusted_frame)
            return adjusted_sequence
        sequence[..., 2] -= z_offset
        return sequence

    return sequence


def render(vertices, out_path, fps, progress_bar=tqdm, put_ground=True, **kwargs):
    # Put the vertices at the floor level
    if put_ground:
        ground = vertices[..., 2].min()
        vertices[..., 2] -= ground

        if "points_seq" in kwargs and kwargs["points_seq"] is not None:
            kwargs["points_seq"] = _shift_sequence_z(kwargs["points_seq"], ground)
        if "line_seq" in kwargs and kwargs["line_seq"] is not None:
            kwargs["line_seq"] = _shift_sequence_z(kwargs["line_seq"], ground)

    import pyrender

    # remove title if it exists
    kwargs.pop("title", None)

    # vertices: SMPL-H vertices
    # verts = np.load("interval_2_verts.npy")
    out_folder = os.path.splitext(out_path)[0]

    if isinstance(vertices, (list, tuple)):
        body_pred = [Struct(v=torch.from_numpy(v), f=FACES) for v in vertices]
        viz_smpl_seq_multi(
            pyrender, out_folder, body_pred, fps=fps, progress_bar=progress_bar, **kwargs
        )

    else:
        verts = torch.from_numpy(vertices)
        body_pred = Struct(v=verts, f=FACES)

        # out_folder, body_pred, start, end, fps, kwargs = args
        viz_smpl_seq(
            pyrender, out_folder, body_pred, fps=fps, progress_bar=progress_bar, **kwargs
        )

    video = Video(out_folder, fps=fps)
    video.save(out_path)


def render_offset(args):
    import pyrender

    out_folder, body_pred, start, end, fps, kwargs = args
    viz_smpl_seq(
        pyrender, out_folder, body_pred, start=start, end=end, fps=fps, **kwargs
    )
    return 0


def render_multiprocess(vertices, out_path, fps, **kwargs):
    # WIP: does not work yet

    # remove title if it exists
    kwargs.pop("title", None)

    # vertices: SMPL-H vertices
    # verts = np.load("interval_2_verts.npy")
    out_folder = os.path.splitext(out_path)[0]

    verts = torch.from_numpy(vertices)
    body_pred = Struct(v=verts, f=FACES)

    # faster rendering
    # by rendering part of the sequence in parallel
    # still work in progress, use one process for now
    n_processes = 1

    verts_lst = np.array_split(verts, n_processes)
    len_split = [len(x) for x in verts_lst]
    starts = [0] + np.cumsum([x for x in len_split[:-1]]).tolist()
    ends = np.cumsum([x for x in len_split]).tolist()
    out_folders = [out_folder for _ in range(n_processes)]
    fps_s = [fps for _ in range(n_processes)]
    kwargs_s = [kwargs for _ in range(n_processes)]
    body_pred_s = [body_pred for _ in range(n_processes)]

    arguments = [out_folders, body_pred_s, starts, ends, fps_s, kwargs_s]
    # sanity
    # lst = [verts[start:end] for start, end in zip(starts, ends)]
    # assert (torch.cat(lst) == verts).all()

    processes = []
    for _, args in zip(range(n_processes), zip(*arguments)):
        process = Process(target=render_offset, args=(args,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    if False:
        # start 4 worker processes
        with Pool(processes=n_processes) as pool:
            # print "[0, 1, 4,..., 81]"
            # print same numbers in arbitrary order
            print(f"0/{n_processes} rendered")
            i = 0
            for _ in pool.imap_unordered(render_offset, zip(*arguments)):
                i += 1
                print(f"i/{n_processes} rendered")

    video = Video(out_folder, fps=fps)
    video.save(out_path)
