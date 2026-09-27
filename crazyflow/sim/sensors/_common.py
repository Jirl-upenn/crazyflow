"""Helpers shared by the batched camera sensors."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import mujoco
import numpy as np

if TYPE_CHECKING:
    from typing import Sequence

    from crazyflow.sim.sim import Sim

Params = ParamSpec("Params")
Return = TypeVar("Return")


def requires_gpu(fn: Callable[Params, Return]) -> Callable[Params, Return]:
    """Decorator to ensure that the simulation is running on the GPU."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Any, **kwargs: Any) -> Return:
        if sim.device.platform != "gpu":
            raise RuntimeError(f"{fn.__name__} requires a simulation running on the GPU.")
        return fn(sim, *args, **kwargs)

    return wrapper


def resolve_drones(sim: Sim, drones: int | Sequence[int] | None) -> tuple[int, ...]:
    """Normalize a drone selection to a tuple of drone indices."""
    if isinstance(drones, (int, np.integer)):
        return (int(drones),)
    ids = range(sim.n_drones) if drones is None else drones
    return tuple(int(d) for d in ids)


def camera_id(mj_model: mujoco.MjModel, prefix: str, drone: int) -> int:
    """Camera index of a drone for the given camera name prefix."""
    name = f"{prefix}:{drone}"
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"Camera '{name}' not found in the model")
    return cam_id


def camera_intrinsics(
    mj_model: mujoco.MjModel, camera_id: int, resolution: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pinhole intrinsics of a model camera for a given image resolution.

    Args:
        mj_model: MuJoCo model containing the camera.
        camera_id: Camera index.
        resolution: Image resolution as (width, height).

    Returns:
        Focal lengths (fx, fy) and principal point (cx, cy) in pixels.
    """
    width, height = resolution
    fov_y = np.deg2rad(mj_model.cam_fovy[camera_id])
    focal = float(height / (2.0 * np.tan(fov_y / 2.0)))
    return (focal, focal), (width / 2.0, height / 2.0)
