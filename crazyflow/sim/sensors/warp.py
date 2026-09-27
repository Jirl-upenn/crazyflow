"""Batched camera sensors built on the MuJoCo Warp ray tracer.

Renders the MuJoCo geometry of the simulation as RGB(-D) images from any model camera, in all worlds
at once, through :func:`mujoco.mjx.render`. This module requires the ``warp`` extra and a simulation
constructed with ``device="gpu"``.

Body, camera, and light poses are computed with MuJoCo Warp rather than with
:func:`~crazyflow.sim.sim.sync_sim2mjx`. MJX's JAX kinematics leaves the children of mocap bodies,
such as the propellers, at their initial pose, and does not compute the subtree centers of mass
that ``trackcom`` cameras follow.
"""

from __future__ import annotations

import copy
import weakref
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from crazyflow.sim.sensors._common import camera_id, requires_gpu, resolve_drones

if TYPE_CHECKING:
    from typing import Callable, Sequence

    from jax import Array

    from crazyflow.sim.data import SimData
    from crazyflow.sim.sim import Sim

_FIXED = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
_TRACK = int(mujoco.mjtCamLight.mjCAMLIGHT_TRACK)
_TRACKCOM = int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)

# Renderers built by render_warp_rgb(d), per model. Keyed weakly on the model so that rebuilding the
# MuJoCo model (e.g. Sim.build_mjx) discards the renderers of the old one.
_RENDERERS: weakref.WeakKeyDictionary[mujoco.MjModel, dict[tuple, Callable[..., Array]]] = (
    weakref.WeakKeyDictionary()
)


@requires_gpu
def render_warp_rgb(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] | None = None,
    supersample: int = 1,
    camera_prefix: str = "fpv_cam",
    geom_rgba: Array | None = None,
) -> Array:
    """Render RGB images of the MuJoCo scene in all worlds.

    The renderer for a given set of arguments is built on the first call and reused afterwards. Use
    :func:`build_render_warp_fn` to hold on to the renderer explicitly, e.g. inside a jitted loop.

    Args:
        sim: The simulation to render.
        drones: Drones whose cameras are rendered. ``None`` renders every drone, an int renders one,
            and a sequence renders that subset in order.
        resolution: Image resolution as (width, height).
        background: RGB color with values in [0, 1] for rays that hit nothing. ``None`` renders the
            model's skybox if it has one.
        supersample: Rays per pixel along each image axis. The image is rendered at ``supersample``
            times the resolution and each block is averaged, which suppresses the flicker of
            textures finer than a pixel. The cost grows with ``supersample**2``.
        camera_prefix: Camera name prefix, resolved to ``{camera_prefix}:{drone}`` for each drone.
        geom_rgba: Per-geom RGBA colors replacing the model's, of shape (ngeom, 4) or, to color
            each world differently, (n_worlds, ngeom, 4). As in MuJoCo, only geoms without a
            material take their color from it.

    Returns:
        RGB images with values in [0, 1] of shape (n_worlds, n_selected, height, width, 3).
    """
    render = _cached_renderer(
        sim, drones, resolution, background, supersample, camera_prefix, depth=False
    )
    return render(sim.data, geom_rgba)


@requires_gpu
def render_warp_rgbd(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] | None = None,
    max_range: float = 10.0,
    supersample: int = 1,
    camera_prefix: str = "fpv_cam",
    geom_rgba: Array | None = None,
) -> Array:
    """Render RGB-D images of the MuJoCo scene in all worlds.

    Adds the depth along the camera's optical axis as a fourth channel. Pixels whose ray hits
    nothing report ``max_range``.

    Args:
        sim: The simulation to render.
        drones: Drones whose cameras are rendered. ``None`` renders every drone, an int renders one,
            and a sequence renders that subset in order.
        resolution: Image resolution as (width, height).
        background: RGB color with values in [0, 1] for rays that hit nothing. ``None`` renders the
            model's skybox if it has one.
        max_range: Sensor range in meters. Reported where nothing is hit, and depth clips to it.
        supersample: Rays per pixel along each image axis. The image is rendered at ``supersample``
            times the resolution and each block is averaged, which suppresses the flicker of
            textures finer than a pixel. The cost grows with ``supersample**2``. Depth reports
            the nearest sample in each block.
        camera_prefix: Camera name prefix, resolved to ``{camera_prefix}:{drone}`` for each drone.
        geom_rgba: Per-geom RGBA colors replacing the model's, of shape (ngeom, 4) or, to color
            each world differently, (n_worlds, ngeom, 4). As in MuJoCo, only geoms without a
            material take their color from it.

    Returns:
        RGB in [0, 1] followed by depth in meters along the camera's optical axis, of shape
        (n_worlds, n_selected, height, width, 4).
    """
    render = _cached_renderer(
        sim, drones, resolution, background, supersample, camera_prefix, True, max_range
    )
    return render(sim.data, geom_rgba)


@requires_gpu
def build_render_warp_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] | None = None,
    supersample: int = 1,
    camera_prefix: str = "fpv_cam",
) -> Callable[..., Array]:
    """Build a Warp RGB renderer for a drone selection, camera prefix, and resolution.

    The renderer is fixed to the simulation's model and number of worlds, and holds its own MuJoCo
    Warp render buffers for as long as the returned function is alive. Build once and reuse it.

    Returns:
        A function ``render(data, geom_rgba=None)`` mapping the simulation data, and optionally
        per-geom colors as in :func:`render_warp_rgb`, to images. It can be called inside jit.
    """
    drone_ids = resolve_drones(sim, drones)
    return _build(sim, drone_ids, resolution, background, supersample, camera_prefix, depth=False)


@requires_gpu
def build_render_warp_rgbd_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] | None = None,
    max_range: float = 10.0,
    supersample: int = 1,
    camera_prefix: str = "fpv_cam",
) -> Callable[..., Array]:
    """Build a Warp RGB-D renderer for a drone selection, camera prefix, and resolution.

    Mirrors :func:`build_render_warp_fn`.
    """
    drone_ids = resolve_drones(sim, drones)
    return _build(
        sim, drone_ids, resolution, background, supersample, camera_prefix, True, max_range
    )


def _cached_renderer(
    sim: Sim,
    drones: int | Sequence[int] | None,
    resolution: tuple[int, int],
    background: tuple[float, float, float] | None,
    supersample: int,
    camera_prefix: str,
    depth: bool,
    max_range: float = 10.0,
) -> Callable[..., Array]:
    """Look up the renderer for these arguments, building it on first use."""
    drone_ids = resolve_drones(sim, drones)
    background = None if background is None else tuple(float(x) for x in background)
    max_range = float(max_range)
    key = (drone_ids, tuple(resolution), background, supersample, camera_prefix, depth, max_range)
    renderers = _RENDERERS.setdefault(sim.mj_model, {})
    if key not in renderers:
        renderers[key] = _build(
            sim, drone_ids, resolution, background, supersample, camera_prefix, depth, max_range
        )
    return renderers[key]


def _build(
    sim: Sim,
    drone_ids: tuple[int, ...],
    resolution: tuple[int, int],
    background: tuple[float, float, float] | None,
    supersample: int,
    camera_prefix: str,
    depth: bool,
    max_range: float = 10.0,
) -> Callable[..., Array]:
    """Set up the MuJoCo Warp model, data, and render context, and return the jitted renderer."""
    from mujoco.mjx import warp as mjxw

    if not mjxw.WARP_INSTALLED:
        raise RuntimeError("Warp rendering requires the warp extra: pip install 'crazyflow[warp]'")
    from mujoco.mjx.warp import render_context
    from mujoco.mjx.warp import smooth as warp_smooth
    from mujoco.mjx.warp.types import GraphMode

    # Rendering needs no contacts, and mj_forward (run by the render context setup) raises on
    # contacts between two static bodies such as drones welded to the world
    mj_model = copy.deepcopy(sim.mj_model)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    cam_ids = tuple(camera_id(mj_model, camera_prefix, d) for d in drone_ids)
    _check_camlight_modes(mj_model, cam_ids)
    if not isinstance(supersample, (int, np.integer)) or supersample < 1:
        raise ValueError(f"supersample must be a positive integer, got {supersample}")
    n_worlds, (width, height), k = sim.n_worlds, resolution, int(supersample)

    mx = mjx.put_model(mj_model, impl="warp", device=sim.device)

    def with_graph_mode(mode: GraphMode) -> mjx.Model:
        return mx.replace(opt=mx.opt.replace(_impl=mx.opt._impl.replace(graph_mode=mode)))

    # GraphMode.WARP caches one CUDA graph per set of input buffer addresses, and each new sim
    # state lives at new addresses, so it would re-capture on most calls. WARP_STAGED copies the
    # inputs into fixed buffers and captures once. put_model treats GraphMode.NONE (== 0) as unset.
    mx, mx_eager = with_graph_mode(GraphMode.WARP_STAGED), with_graph_mode(GraphMode.NONE)
    template = jax.vmap(lambda _: mjx.make_data(mj_model, impl="warp"))(jnp.arange(n_worlds))
    # Start from the sim's current MuJoCo state so that non-drone mocap bodies render where they are
    template = jax.device_put(
        template.replace(
            qpos=sim.mjx_data.qpos,
            mocap_pos=sim.mjx_data.mocap_pos,
            mocap_quat=sim.mjx_data.mocap_quat,
        ),
        sim.device,
    )

    selected = [i in cam_ids for i in range(mj_model.ncam)]
    has_skybox = bool(np.any(mj_model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX))
    ctx = mjx.create_render_context(
        mj_model,
        nworld=n_worlds,
        devices=[f"cuda:{sim.device.id}"],
        cam_res=(width * k, height * k),
        render_rgb=selected,
        render_depth=selected if depth else False,
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
        render_skybox=background is None and has_skybox,
        **({} if background is None else {"background_color": (*background, 1.0)}),
    )
    warp_rc = render_context.get(ctx.pytree())
    rgb_adr = tuple(int(warp_rc.rgb_adr.numpy()[c]) for c in cam_ids)
    depth_adr = tuple(int(warp_rc.depth_adr.numpy()[c]) for c in cam_ids)
    camlight = _camlight_fn(mj_model)
    n_pixels = width * height * k * k
    # Images are rendered at (height * k, width * k). Split out each pixel's k x k block of samples
    block_shape = (n_worlds, len(cam_ids), height, k, width, k)

    rgba_shape = (n_worlds, mj_model.ngeom, 4)
    base_rgba = jax.device_put(jnp.asarray(mj_model.geom_rgba), sim.device)

    def render(data: SimData, geom_rgba: Array, mx: mjx.Model) -> Array:
        # MuJoCo quat is [w, x, y, z], ours is [x, y, z, w]
        quat = jnp.roll(data.states.quat, 1, axis=-1)
        ids = data.core.drone_mocap_ids
        d = template.replace(
            mocap_pos=template.mocap_pos.at[:, ids].set(data.states.pos),
            mocap_quat=template.mocap_quat.at[:, ids].set(quat),
        )
        d = jax.vmap(mjx.kinematics, in_axes=(None, 0))(mx, d)
        d = jax.vmap(warp_smooth.com_pos, in_axes=(None, 0))(mx, d)
        d = camlight(d)  # MJX has no Warp camlight
        rc = ctx.pytree()  # Referencing ctx keeps the render buffers alive with the renderer
        d = mjx.refit_bvh(mx, d, rc)
        # The renderer reads geom colors per world (world % leading axis), so a batch recolors
        geom_rgba = jnp.broadcast_to(jnp.asarray(geom_rgba, base_rgba.dtype), rgba_shape)
        rgb_data, depth_data = mjx.render(mx.replace(geom_rgba=geom_rgba), d, rc)
        rgb = jnp.stack([_unpack_rgb(rgb_data[:, a : a + n_pixels]) for a in rgb_adr], axis=1)
        rgb = rgb.reshape(*block_shape, 3).mean(axis=(3, 5))
        if not depth:
            return rgb
        raw = jnp.stack([depth_data[:, a : a + n_pixels] for a in depth_adr], axis=1)
        raw = raw.reshape(*block_shape, 1)
        # The renderer leaves 0 where a ray hits nothing. The nearest sample of a block avoids the
        # in-between depths that averaging across an object's silhouette would invent.
        metric = jnp.where(raw <= 0.0, max_range, jnp.minimum(raw, max_range)).min(axis=(3, 5))
        return jnp.concatenate([rgb, metric], axis=-1)

    # Warp captures each render into a CUDA graph. Loading a kernel module during capture requires
    # CUDA driver 12.3+, so render once without graphs to load all kernels before the first capture.
    jax.block_until_ready(jax.jit(lambda data: render(data, base_rgba, mx_eager))(sim.data))
    render_jit = jax.jit(lambda data, geom_rgba: render(data, geom_rgba, mx))

    def render_fn(data: SimData, geom_rgba: Array | None = None) -> Array:
        return render_jit(data, base_rgba if geom_rgba is None else geom_rgba)

    return render_fn


def _unpack_rgb(packed: Array) -> Array:
    """Unpack the renderer's uint32 pixels (0xAARRGGBB) into float RGB in [0, 1]."""
    channels = [(packed >> shift) & 0xFF for shift in (16, 8, 0)]
    return jnp.stack(channels, axis=-1).astype(jnp.float32) / 255.0


def _check_camlight_modes(mj_model: mujoco.MjModel, cam_ids: tuple[int, ...]):
    """Raise if a rendered camera or any light uses a mode that camlight does not implement."""
    supported = (_FIXED, _TRACK, _TRACKCOM)
    for i in cam_ids:
        if mj_model.cam_mode[i] not in supported:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            raise NotImplementedError(f"Camera '{name}' uses an unsupported (target) mode")
    if np.any(~np.isin(mj_model.light_mode, supported)):
        raise NotImplementedError("Lights with target modes are not supported")


def _camlight_fn(mj_model: mujoco.MjModel) -> Callable[[mjx.Data], mjx.Data]:
    """Build a batched ``mj_camlight`` for fixed, track, and trackcom cameras and lights.

    Target modes are left at their fixed pose. :func:`_check_camlight_modes` rejects them for every
    camera and light the renderer reads.
    """
    # Copy the fields out so the closure does not keep the MjModel (a weak cache key) alive
    anchor = ("bodyid", "mode", "pos", "pos0", "poscom0")
    cam = {k: np.array(getattr(mj_model, f"cam_{k}")) for k in (*anchor, "mat0")}
    cam["mat0"] = cam["mat0"].reshape(-1, 3, 3)
    cam["rot"] = np.stack([_quat2mat(q) for q in mj_model.cam_quat]).reshape(-1, 3, 3)
    light = {k: np.array(getattr(mj_model, f"light_{k}")) for k in (*anchor, "dir", "dir0")}

    def camlight(d: mjx.Data) -> mjx.Data:
        cam_pos = _anchor(d, cam["bodyid"], cam["mode"], cam["pos"], cam["pos0"], cam["poscom0"])
        cam_mat = jnp.where(
            np.isin(cam["mode"], (_TRACK, _TRACKCOM))[None, :, None, None],
            cam["mat0"][None],
            jnp.einsum("wcij,cjk->wcik", d.xmat[:, cam["bodyid"]], cam["rot"]),
        )
        light_pos = _anchor(
            d, light["bodyid"], light["mode"], light["pos"], light["pos0"], light["poscom0"]
        )
        light_dir = jnp.where(
            np.isin(light["mode"], (_TRACK, _TRACKCOM))[None, :, None],
            light["dir0"][None],
            jnp.einsum("wlij,lj->wli", d.xmat[:, light["bodyid"]], light["dir"]),
        )
        impl = d._impl.replace(light_xpos=light_pos, light_xdir=light_dir)
        return d.replace(cam_xpos=cam_pos, cam_xmat=cam_mat, _impl=impl)

    return camlight


def _anchor(
    d: mjx.Data,
    body: np.ndarray,
    mode: np.ndarray,
    local_pos: np.ndarray,
    pos0: np.ndarray,
    poscom0: np.ndarray,
) -> Array:
    """World positions of cameras or lights attached to ``body`` in the given camlight modes."""
    fixed = d.xpos[:, body] + jnp.einsum("wbij,bj->wbi", d.xmat[:, body], local_pos)
    track = d.xpos[:, body] + pos0
    trackcom = d.subtree_com[:, body] + poscom0
    mode = mode[None, :, None]
    return jnp.where(mode == _TRACK, track, jnp.where(mode == _TRACKCOM, trackcom, fixed))


def _quat2mat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a MuJoCo [w, x, y, z] quaternion."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)
