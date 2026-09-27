"""Render one drone flight with the MuJoCo OpenGL renderer and the MuJoCo Warp sensor side by side.

The drone follows a figure-8 whose attitude is derived from the path's acceleration, the way a
quadrotor banks into turns. Its pose is set directly every frame, so both renderers see exactly the
same trajectory. Each frame is rendered with ``Sim.render`` (MuJoCo's OpenGL renderer) and with
:func:`crazyflow.sim.sensors.warp.build_render_warp_fn` (MuJoCo Warp ray tracer), and the two views
are written next to each other into a video together with their per-frame render times.

Requires the warp extra, a CUDA-capable GPU, and ffmpeg. Run with::

    python benchmark/render_compare.py --output render_compare.mp4 --camera fpv_cam --supersample 3
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # Headless OpenGL for Sim.render
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.sensors.warp import build_render_warp_fn

GRAVITY = 9.81


def figure_8(t: float, period: float, radius: float, height: float) -> tuple[np.ndarray, ...]:
    """Position, velocity, and acceleration on a horizontal figure-8 at time t."""
    w = 2 * np.pi / period
    pos = np.array([radius * np.sin(w * t), 0.5 * radius * np.sin(2 * w * t), height])
    vel = np.array([radius * w * np.cos(w * t), radius * w * np.cos(2 * w * t), 0.0])
    acc = np.array([-radius * w**2 * np.sin(w * t), -2 * radius * w**2 * np.sin(2 * w * t), 0.0])
    return pos, vel, acc


def flat_attitude(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] that points the thrust along acc + g and the nose along vel."""
    z = acc + np.array([0.0, 0.0, GRAVITY])
    z /= np.linalg.norm(z)
    yaw = np.arctan2(vel[1], vel[0])
    y = np.cross(z, [np.cos(yaw), np.sin(yaw), 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return R.from_matrix(np.stack([x, y, z], axis=1)).as_quat()


def set_pose(sim: Sim, pos: np.ndarray, quat: np.ndarray):
    """Place drone 0 of world 0 at the given pose."""
    states = sim.data.states.replace(
        pos=sim.data.states.pos.at[0, 0].set(jnp.asarray(pos)),
        quat=sim.data.states.quat.at[0, 0].set(jnp.asarray(quat)),
    )
    # Clear the sync flag like Sim.step does, or Sim.render keeps drawing the last synced pose
    core = sim.data.core.replace(mjx_synced=False)
    sim.data = sim.data.replace(states=states, core=core)


def main(
    output: str = "render_compare.mp4",
    camera: str = "fpv_cam",
    resolution: tuple[int, int] = (320, 240),
    duration: float = 10.0,
    fps: int = 30,
    period: float = 8.0,
    radius: float = 1.5,
    height: float = 1.0,
    supersample: int = 3,
):
    """Render the figure-8 with both renderers and write the side-by-side video to ``output``."""
    sim = Sim(n_worlds=1, n_drones=1, device="gpu")
    width, height_px = resolution
    render_warp = build_render_warp_fn(
        sim, resolution=resolution, supersample=supersample, camera_prefix=camera
    )
    render_warp(sim.data).block_until_ready()  # The first call captures the CUDA graph

    fig, axes = plt.subplots(1, 2, figsize=(2 * width / 80 + 0.5, height_px / 80 + 0.9), dpi=100)
    blank = np.zeros((height_px, width, 3))
    ims = [ax.imshow(blank) for ax in axes]
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    suptitle = fig.suptitle("")

    n_frames = int(duration * fps)
    times = {"opengl": [], "warp": []}
    writer = FFMpegWriter(fps=fps, metadata={"title": f"OpenGL vs Warp, {camera}:0"})
    with writer.saving(fig, output, dpi=100):
        for i in range(n_frames):
            t = i / fps
            pos, vel, acc = figure_8(t, period, radius, height)
            set_pose(sim, pos, flat_attitude(vel, acc))

            t0 = time.perf_counter()
            img_gl = sim.render(
                mode="rgb_array", camera=f"{camera}:0", width=width, height=height_px
            )
            t1 = time.perf_counter()
            img_warp = np.asarray(render_warp(sim.data)[0, 0])  # np.asarray waits for the GPU
            t2 = time.perf_counter()
            times["opengl"].append(t1 - t0)
            times["warp"].append(t2 - t1)

            ims[0].set_data(img_gl)
            ims[1].set_data(img_warp)
            axes[0].set_title(f"MuJoCo OpenGL (Sim.render)  {1e3 * (t1 - t0):5.1f} ms")
            ssaa = f", {supersample}x{supersample} SSAA" if supersample > 1 else ""
            axes[1].set_title(f"MuJoCo Warp{ssaa}  {1e3 * (t2 - t1):5.1f} ms")
            suptitle.set_text(f"{camera}:0   t = {t:4.1f} s")
            writer.grab_frame()
    plt.close(fig)
    sim.close()

    for name, ts in times.items():
        ts = np.asarray(ts[1:])  # The first OpenGL frame creates the renderer
        print(f"{name:>6}: median {1e3 * np.median(ts):.2f} ms/frame over {len(ts)} frames")
    print(f"Wrote {n_frames} frames to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="render_compare.mp4", help="Output video path")
    parser.add_argument("--camera", default="fpv_cam", help="fpv_cam or track_cam")
    parser.add_argument("--resolution", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    parser.add_argument("--duration", type=float, default=10.0, help="Video length in seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--supersample", type=int, default=3, help="Warp rays per pixel per axis")
    args = parser.parse_args()
    main(
        args.output,
        args.camera,
        tuple(args.resolution),
        args.duration,
        args.fps,
        supersample=args.supersample,
    )
