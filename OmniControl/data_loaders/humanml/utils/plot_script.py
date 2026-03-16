# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/shared/centos7/ffmpeg/20190305/bin/ffmpeg'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from io import BytesIO
from PIL import Image
import tempfile, os

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], hint=None, objects=None, elev=120, azim=-90, tensorboard_vis=False):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20)) if title is not None else ''

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
        if hint is not None:
            mask = hint.sum(-1) != 0
            hint = hint[mask]
            hint *= 0.003
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
        if hint is not None:
            mask = hint.sum(-1) != 0
            hint = hint[mask]
            hint *= 1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def plot_objects(objects, trajec, index, height_offset):
        if objects is not None:
            for obj in objects:
                name = obj['name']
                position = obj['position']
                dimensions = obj['dimensions']
                if dataset == 'kit':
                    scale = 0.003  # scale for visualization 
                elif dataset == 'humanml':
                    scale = 1.3  # scale for visualization

                pos_x, pos_y, pos_z = position['x']*scale, position['z']*scale, -position['y']*scale
                dim_len, dim_wid, dim_hei = dimensions['length']*scale, dimensions['width']*scale, dimensions['height']*scale
                # Create vertices for the cuboid
                verts = [
                    [pos_x - dim_wid/2 - trajec[index, 0], pos_y - dim_hei/2 - height_offset, pos_z - dim_len/2 - trajec[index, 1]],
                    [pos_x - dim_wid/2 - trajec[index, 0], pos_y + dim_hei/2 - height_offset, pos_z - dim_len/2 - trajec[index, 1]],
                    [pos_x + dim_wid/2 - trajec[index, 0], pos_y + dim_hei/2 - height_offset, pos_z - dim_len/2 - trajec[index, 1]],
                    [pos_x + dim_wid/2 - trajec[index, 0], pos_y - dim_hei/2 - height_offset, pos_z - dim_len/2 - trajec[index, 1]],
                    [pos_x - dim_wid/2 - trajec[index, 0], pos_y - dim_hei/2 - height_offset, pos_z + dim_len/2 - trajec[index, 1]],
                    [pos_x - dim_wid/2 - trajec[index, 0], pos_y + dim_hei/2 - height_offset, pos_z + dim_len/2 - trajec[index, 1]],
                    [pos_x + dim_wid/2 - trajec[index, 0], pos_y + dim_hei/2 - height_offset, pos_z + dim_len/2 - trajec[index, 1]],
                    [pos_x + dim_wid/2 - trajec[index, 0], pos_y - dim_hei/2 - height_offset, pos_z + dim_len/2 - trajec[index, 1]]
                ]
                faces = [
                    [verts[0], verts[1], verts[5], verts[4]],
                    [verts[7], verts[6], verts[2], verts[3]],
                    [verts[0], verts[3], verts[7], verts[4]],
                    [verts[1], verts[2], verts[6], verts[5]],
                    [verts[0], verts[1], verts[2], verts[3]],
                    [verts[4], verts[5], verts[6], verts[7]]
                ]
                cuboid = Poly3DCollection(faces)
                cuboid.set_facecolor((0.5, 0.5, 0.5, 0.5))
                ax.add_collection3d(cuboid)
                # Add object name
                ax.text(pos_x - trajec[index, 0], pos_y - height_offset, pos_z - trajec[index, 1], name, color='black')       
    def update(index):
        # Reset axes for this frame
        ax.cla()
        init()
        
        ax.view_init(elev=elev, azim=azim)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        
        # Plot objects if provided
        plot_objects(objects, trajec, index, height_offset)

        if hint is not None:
            ax.scatter(hint[..., 0] - trajec[index, 0], hint[..., 1], hint[..., 2] - trajec[index, 1], color="#80B79A")

            # # link a path for hints
            # ax.plot3D(hint[..., 0] - trajec[index, 0], hint[..., 1], hint[..., 2] - trajec[index, 1], linewidth=1.0,
            #           color="#34C1E2")

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # Create line without reusing artists
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
                      linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    if tensorboard_vis:
        
        buf = BytesIO()
        writer = PillowWriter(fps=fps)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name
        ani.save(gif_path, writer=writer)

        with open(gif_path, "rb") as f:
            buf = BytesIO(f.read())
        buf.seek(0)
        image = Image.open(buf)
        plt.close('all')
        
        frames = []
        try:
            while True:
                frame = np.array(image.convert('RGB'))  # Convert to RGB and then to a NumPy array.
                frames.append(frame)
                image.seek(image.tell() + 1)  # Move to the next frame.
        except EOFError:
            pass  # Exit when the last frame is reached.

        frames = np.stack(frames)  # (num_frames, H, W, 3)
        os.remove(gif_path)
        return frames
    else:
        ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)
        plt.close('all')
