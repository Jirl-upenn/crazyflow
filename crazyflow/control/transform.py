"""Transformations between physical parameters of the quadrotors.

Bundles conversions between motor forces, rotor velocities, and PWM commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def quat_to_rot_mat(quat: Array) -> Array:
    """Body -> world rotation matrix from a scalar-last (x, y, z, w) quaternion.

    Batched over arbitrary leading dims: quat (..., 4) -> matrix (..., 3, 3).
    Equivalent to scipy.spatial.transform.Rotation.from_quat(quat).as_matrix(),
    but written purely in terms of array_api_compat's xp so it stays
    jit/vmap/scan-compatible on jax (scipy's Rotation materializes its input
    via __array__(), which fails on a jax tracer during tracing).
    """
    xp = array_namespace(quat)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    row0 = xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1)
    row1 = xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1)
    row2 = xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def euler_xyz_to_rot_mat(rpy: Array) -> Array:
    """Rotation matrix from extrinsic xyz Euler angles (roll, pitch, yaw), radians.

    Batched over arbitrary leading dims: rpy (..., 3) -> matrix (..., 3, 3).
    Equivalent to scipy.spatial.transform.Rotation.from_euler("xyz", rpy).
    as_matrix() (Rz(yaw) @ Ry(pitch) @ Rx(roll), verified numerically against
    scipy), but jit/vmap/scan-compatible -- see quat_to_rot_mat's docstring
    for why scipy's Rotation can't be used directly here.
    """
    xp = array_namespace(rpy)
    a, b, c = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    ca, sa = xp.cos(a), xp.sin(a)
    cb, sb = xp.cos(b), xp.sin(b)
    cc, sc = xp.cos(c), xp.sin(c)
    row0 = xp.stack([cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa], axis=-1)
    row1 = xp.stack([sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa], axis=-1)
    row2 = xp.stack([-sb, cb * sa, cb * ca], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def rot_mat_to_euler_xyz(rot_mat: Array) -> Array:
    """Extrinsic xyz Euler angles (roll, pitch, yaw), radians, from a rotation matrix.

    Inverse of ``euler_xyz_to_rot_mat``; equivalent to scipy's
    ``Rotation.from_matrix(m).as_euler("xyz")`` away from the pitch = +-90 deg singularity.
    Batched over arbitrary leading dims: matrix (..., 3, 3) -> rpy (..., 3).
    """
    xp = array_namespace(rot_mat)
    roll = xp.atan2(rot_mat[..., 2, 1], rot_mat[..., 2, 2])
    pitch = -xp.asin(xp.clip(rot_mat[..., 2, 0], -1.0, 1.0))
    yaw = xp.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
    return xp.stack([roll, pitch, yaw], axis=-1)


def motor_force2rotor_vel(motor_forces: Array, rpm2thrust: Array) -> Array:
    """Convert motor forces to rotor velocities, where f=a*rpm^2+b*rpm+c.

    Args:
        motor_forces: Motor forces in SI units with shape (..., N).
        rpm2thrust: RPM to thrust conversion factors.

    Returns:
        Array of rotor velocities in rad/s with shape (..., N).
    """
    xp = array_namespace(motor_forces)
    return (
        -rpm2thrust[1]
        + xp.sqrt(rpm2thrust[1] ** 2 - 4 * rpm2thrust[2] * (rpm2thrust[0] - motor_forces))
    ) / (2 * rpm2thrust[2])


def force2pwm(thrust: Array | float, thrust_max: Array | float, pwm_max: Array | float) -> Array:
    """Convert thrust in N to thrust in PWM.

    Args:
        thrust: Array or float of the thrust in [N]
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        Thrust converted in PWM.
    """
    return thrust / thrust_max * pwm_max


def pwm2force(
    pwm: Array | float, thrust_max: Array | float, pwm_max: Array | float
) -> Array | float:
    """Convert pwm thrust command to actual thrust.

    Args:
        pwm: Array or float of the pwm value
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        thrust: Array or float thrust in [N]
    """
    return pwm / pwm_max * thrust_max
