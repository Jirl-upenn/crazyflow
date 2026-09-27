"""Unit tests for the MuJoCo Warp camera sensor."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import available_backends
from mujoco import mjx
from mujoco.mjx import warp as mjxw
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.sensors import warp as warp_sensor
from crazyflow.sim.sensors.warp import (
    build_render_warp_fn,
    build_render_warp_rgbd_fn,
    render_warp_rgb,
    render_warp_rgbd,
)

requires_warp = pytest.mark.skipif(
    not mjxw.WARP_INSTALLED or "gpu" not in available_backends(),
    reason="requires the warp extra and a CUDA GPU",
)

RES = (32, 24)

# Fixed, track and trackcom cameras and lights on a mocap body with a jointed child body
CAMLIGHT_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <body name="base" mocap="true">
      <geom type="box" size=".1 .1 .1"/>
      <camera name="fixed" pos=".1 .2 .3" euler="10 20 30"/>
      <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="track"/>
      <camera name="trackcom" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <light pos=".1 0 .2" dir="1 0 -1" mode="fixed"/>
      <light pos="0 1 1" dir="0 -1 -1" mode="trackcom"/>
      <body name="arm" pos=".3 0 0">
        <joint type="hinge" axis="0 0 1"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .4 0 0" mass="2"/>
        <camera name="arm_cam" pos=".4 0 0" euler="0 90 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _set_poses(sim: Sim, seed: int, pitch: tuple[float, float] = (0.2, 0.6)):
    """Place every drone at a random position and yaw, pitched down to see the floor."""
    rng = np.random.default_rng(seed)
    shape = (sim.n_worlds, sim.n_drones)
    pos = rng.uniform([-1, -1, 0.5], [1, 1, 1.5], (*shape, 3))
    eul = np.stack([np.zeros(shape), rng.uniform(*pitch, shape), rng.uniform(-np.pi, np.pi, shape)])
    quat = R.from_euler("xyz", np.moveaxis(eul, 0, -1).reshape(-1, 3)).as_quat().reshape(*shape, 4)
    states = sim.data.states.replace(pos=jnp.asarray(pos), quat=jnp.asarray(quat))
    sim.data = sim.data.replace(states=states)


@pytest.mark.unit
def test_render_warp_requires_gpu():
    sim = Sim(n_worlds=1, n_drones=1, device="cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        render_warp_rgb(sim)
    with pytest.raises(RuntimeError, match="GPU"):
        build_render_warp_rgbd_fn(sim)


@pytest.mark.unit
@requires_warp
def test_camlight_matches_mujoco():
    """The batched camlight reproduces mj_camlight for every camera and light mode we support."""
    mj_model = mujoco.MjModel.from_xml_string(CAMLIGHT_XML)
    mj_data = mujoco.MjData(mj_model)
    mj_data.mocap_pos[0] = [0.3, -0.2, 1.0]
    mj_data.mocap_quat[0] = R.from_euler("xyz", [0.3, -0.4, 1.2]).as_quat(scalar_first=True)
    mj_data.qpos[0] = 0.7
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    mujoco.mj_camlight(mj_model, mj_data)

    data = jax.vmap(lambda _: mjx.make_data(mj_model, impl="warp"))(jnp.arange(1))
    data = data.replace(
        xpos=jnp.asarray(mj_data.xpos)[None],
        xmat=jnp.asarray(mj_data.xmat).reshape(1, -1, 3, 3),
        subtree_com=jnp.asarray(mj_data.subtree_com)[None],
    )
    data = warp_sensor._camlight_fn(mj_model)(data)
    tol = dict(atol=1e-5)
    assert np.allclose(data.cam_xpos[0], mj_data.cam_xpos, **tol)
    assert np.allclose(data.cam_xmat[0].reshape(-1, 9), mj_data.cam_xmat, **tol)
    assert np.allclose(data._impl.light_xpos[0], mj_data.light_xpos, **tol)
    assert np.allclose(data._impl.light_xdir[0], mj_data.light_xdir, **tol)


@pytest.mark.unit
@requires_warp
def test_render_warp_rgb():
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    _set_poses(sim, 0)
    # Every drone's fpv camera renders into a (n_worlds, n_drones, H, W, 3) stack
    img = render_warp_rgb(sim, resolution=RES)
    assert img.shape == (2, 2, RES[1], RES[0], 3)
    assert img.dtype == jnp.float32
    assert np.all((img >= 0.0) & (img <= 1.0))
    assert img.max() > 0.0, "Nothing is visible in the image"
    assert not np.allclose(img[0, 0], img[1, 0]), "Worlds with different poses must differ"
    # Selecting a single drone matches that slice of the full stack
    one = render_warp_rgb(sim, drones=1, resolution=RES)
    assert one.shape == (2, 1, RES[1], RES[0], 3)
    assert np.allclose(one[:, 0], img[:, 1], atol=1e-6)
    # The compiled variant renders the same images
    render_fn = build_render_warp_fn(sim, resolution=RES)
    assert np.allclose(img, render_fn(sim.data), atol=1e-6)


@pytest.mark.unit
@requires_warp
def test_render_warp_follows_state():
    """Images track sim.data on every call, including through the captured CUDA graph."""
    sim = Sim(n_worlds=2, n_drones=1, device="gpu")
    render_fn = build_render_warp_fn(sim, resolution=RES)
    frames = []
    for seed in range(3):
        _set_poses(sim, seed)
        frames.append(np.asarray(render_fn(sim.data)))
    assert all(not np.allclose(a, b) for a, b in zip(frames, frames[1:]))
    # Rendering an earlier state again reproduces its image exactly
    _set_poses(sim, 0)
    assert np.array_equal(np.asarray(render_fn(sim.data)), frames[0])


@pytest.mark.unit
@requires_warp
def test_render_warp_camera_prefix():
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    _set_poses(sim, 0)
    fpv = render_warp_rgb(sim, resolution=RES)
    track = render_warp_rgb(sim, resolution=RES, camera_prefix="track_cam")
    assert track.shape == fpv.shape
    assert not np.allclose(fpv, track), "track_cam and fpv_cam must give different views"
    with pytest.raises(ValueError, match="not found"):
        render_warp_rgb(sim, resolution=RES, camera_prefix="does_not_exist")


@pytest.mark.unit
@requires_warp
def test_render_warp_rgbd():
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    _set_poses(sim, 0)
    max_range = 5.0
    img = render_warp_rgbd(sim, resolution=RES, max_range=max_range)
    assert img.shape == (2, 2, RES[1], RES[0], 4)
    # The color channels match the RGB renderer
    assert np.allclose(img[..., :3], render_warp_rgb(sim, resolution=RES), atol=1e-6)
    depth = np.asarray(img[..., 3])
    assert np.all((depth > 0.0) & (depth <= max_range))
    assert depth.min() < max_range, "The drones look at the floor, some pixels must hit it"
    render_fn = build_render_warp_rgbd_fn(sim, resolution=RES, max_range=max_range)
    assert np.allclose(img, render_fn(sim.data), atol=1e-6)


@pytest.mark.unit
@requires_warp
def test_render_warp_supersample():
    """Supersampling equals rendering at k times the resolution and pooling each k x k block."""
    sim = Sim(n_worlds=2, n_drones=1, device="gpu")
    _set_poses(sim, 0)
    k, (w, h) = 2, RES
    fine = np.asarray(render_warp_rgbd(sim, resolution=(w * k, h * k)))
    blocks = fine.reshape(2, 1, h, k, w, k, 4)
    img = np.asarray(render_warp_rgbd(sim, resolution=RES, supersample=k))
    assert img.shape == (2, 1, h, w, 4)
    assert np.allclose(img[..., :3], blocks[..., :3].mean(axis=(3, 5)), atol=1e-6)
    assert np.allclose(img[..., 3], blocks[..., 3].min(axis=(3, 5)), atol=1e-6)
    rgb = render_warp_rgb(sim, resolution=RES, supersample=k)
    assert np.allclose(rgb, img[..., :3], atol=1e-6)
    with pytest.raises(ValueError, match="supersample"):
        render_warp_rgb(sim, resolution=RES, supersample=0)


@pytest.mark.unit
@requires_warp
def test_render_warp_geom_rgba(tmp_path: Path):
    """Geom colors can be replaced for all worlds at once or per world."""
    scene = (Path(__file__).parents[2] / "crazyflow" / "scene.xml").read_text()
    wall = '<geom name="wall" type="box" pos="1 0 1" size="0.05 3 3" rgba="0 0 1 1" contype="0"/>'
    (tmp_path / "scene.xml").write_text(scene.replace("</worldbody>", f"{wall}</worldbody>"))
    sim = Sim(n_worlds=2, n_drones=1, device="gpu", xml_path=tmp_path / "scene.xml")
    # Hover in front of the wall, which then fills the center of the fpv image
    states = sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(1.0))
    sim.data = sim.data.replace(states=states)
    wall_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    center = (slice(None), 0, slice(8, 16), slice(12, 20))  # (world, cam, rows, cols)

    default = np.asarray(render_warp_rgb(sim, resolution=RES))[center]
    assert np.all(default[..., 2].mean(axis=(1, 2)) > 0.3), "The wall must look blue by default"
    assert np.all(default[..., 0].mean(axis=(1, 2)) < 0.1)
    # One color table for every world
    rgba = np.array(sim.mj_model.geom_rgba)
    rgba[wall_id] = [1.0, 0.0, 0.0, 1.0]
    red = np.asarray(render_warp_rgb(sim, resolution=RES, geom_rgba=rgba))[center]
    assert np.all(red[..., 0].mean(axis=(1, 2)) > 0.3)
    assert np.all(red[..., 2].mean(axis=(1, 2)) < 0.1)
    # A table per world recolors only world 1, and the built renderer takes the same argument
    per_world = np.stack([np.array(sim.mj_model.geom_rgba), rgba])
    render_fn = build_render_warp_fn(sim, resolution=RES)
    mixed = np.asarray(render_fn(sim.data, per_world))[center]
    assert np.allclose(mixed[0], default[0], atol=1e-6)
    assert np.allclose(mixed[1], red[1], atol=1e-6)
    assert np.allclose(np.asarray(render_fn(sim.data))[center], default, atol=1e-6)


@pytest.mark.unit
@requires_warp
def test_render_warp_background():
    sim = Sim(n_worlds=1, n_drones=1, device="gpu")
    _set_poses(sim, 0, pitch=(-1.2, -1.2))  # Nose up, the camera sees mostly sky
    red = np.asarray(render_warp_rgb(sim, resolution=RES, background=(1.0, 0.0, 0.0)))
    sky = np.asarray(render_warp_rgb(sim, resolution=RES))
    is_red = np.all(red == np.array([1.0, 0.0, 0.0]), axis=-1)
    assert is_red.any(), "Missed rays must take the background color"
    assert not np.all(sky == np.array([1.0, 0.0, 0.0]), axis=-1).any()


@pytest.mark.unit
@requires_warp
def test_render_warp_reuses_renderers():
    sim = Sim(n_worlds=1, n_drones=1, device="gpu")
    render_warp_rgb(sim, resolution=RES)
    render_warp_rgb(sim, resolution=RES)
    assert len(warp_sensor._RENDERERS[sim.mj_model]) == 1
    render_warp_rgb(sim, resolution=(16, 12))
    assert len(warp_sensor._RENDERERS[sim.mj_model]) == 2
