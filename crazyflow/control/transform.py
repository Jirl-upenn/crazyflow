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
        rpm2thrust: RPM to thrust conversion factors, indexed on the last axis ((3,) or per-world (N, 1, 1, 3)).

    Returns:
        Array of rotor velocities in rad/s with shape (..., N).
    """
    xp = array_namespace(motor_forces)
    k0, k1, k2 = rpm2thrust[..., 0], rpm2thrust[..., 1], rpm2thrust[..., 2]
    return (-k1 + xp.sqrt(k1**2 - 4 * k2 * (k0 - motor_forces))) / (2 * k2)


def vmotor2thrust_poly(v: Array, vmotor2thrust: Array) -> Array:
    """Per-motor thrust (N) at motor voltage v (V): the firmware's cubic c0 + c1 v + c2 v^2 + c3 v^3.

    Scalar coefficients give the shape of v; per-world ones ((N, 1, 1, 4), v (N, 1, 1)) give (N, 1, 1), which
    broadcasts against (N, M, 4) motor forces as crazyflow's other per-world coefficients do."""
    c0, c1, c2, c3 = (vmotor2thrust[..., i] for i in range(4))
    return c0 + (c1 + (c2 + c3 * v) * v) * v


def compensated_motor_forces(
    motor_forces: Array,
    *,
    thrust_min: Array | float,
    vmotor2thrust: Array,
    v_batt: Array | float,
    pwm_idle: Array | float,
    pwm_max: Array | float,
) -> Array:
    """The force each motor delivers for a requested force, through the firmware's battery compensation and cap.

    The Crazyflie firmware with ``CONFIG_ENABLE_THRUST_BAT_COMPENSATED`` reads every motor value as a force
    request and turns it into a motor-voltage ratio (``motorsCompensateBatteryVoltage``, motors.c), then caps the
    ratio per motor to ``[idle, 1]`` (``powerDistributionCap``, power_distribution_quadrotor.c):

    * a request below ``thrust_min`` (where the inverse of the voltage curve stops being unique) gets ratio 0,
      which the cap raises to the idle ratio ``pwm_idle / pwm_max``: the motor idles at
      ``vmotor2thrust(pwm_idle / pwm_max * v_batt)``;
    * any other request gets the voltage the inverted cubic gives, divided by the battery voltage; a ratio above 1
      is capped, so the motor delivers ``min(request, vmotor2thrust(v_batt))``.

    The firmware's ratio for a request ``f >= thrust_min`` always lies above the idle ratio (its voltage is at
    least the one ``thrust_min`` needs, which exceeds the idle voltage), so the idle cap only acts below
    ``thrust_min``. With the firmware's voltage estimate equal to ``v_batt`` the inversion and the motor's own
    voltage curve cancel exactly, and this closed form is the firmware's result without evaluating the Cardano
    inverse (whose cube roots have an infinite derivative at 0). Not modelled: battery sag under load and the
    firmware's 100 ms voltage filter (the estimate lagging the true voltage), and the < 2 V cut-off.

    A zero idle ratio stops the motor: it delivers 0 N, not the cubic's negative value at 0 V.

    Args:
        motor_forces: Requested per-motor forces in N with shape (..., 4); any sign or size (the firmware's
            legacy and force-torque mixers do not clip before compensation).
        thrust_min: The firmware's THRUST_MIN in N.
        vmotor2thrust: The firmware's VMOTOR2THRUST0..3 voltage-to-thrust cubic, indexed on the last axis.
        v_batt: Battery voltage in V.
        pwm_idle: The idle motor ratio in 16-bit counts (powerDist.idleThrust; 0 when the build needs no arming).
        pwm_max: Full-scale 16-bit count, 65535.

    Returns:
        The delivered per-motor forces in N with the shape of ``motor_forces``.
    """
    xp = array_namespace(motor_forces)
    # A stopped motor (ratio 0: no idle, as on the brushed CF2) delivers nothing; the cubic is a fit over the
    # flying range and goes negative near 0 V.
    f_idle = xp.maximum(vmotor2thrust_poly(pwm_idle / pwm_max * v_batt, vmotor2thrust), 0.0)
    f_ceil = vmotor2thrust_poly(xp.asarray(v_batt), vmotor2thrust)
    # Scalar coefficients give scalars; per-world ones ((N, 1, 1, 4), v_batt (N, 1, 1)) give (N, 1, 1), which
    # broadcasts against the (N, M, 4) forces as crazyflow's other per-world coefficients do.
    return xp.where(motor_forces < thrust_min, f_idle, xp.minimum(motor_forces, f_ceil))


def compensated_rotor_vel(
    motor_forces: Array,
    *,
    rpm2thrust: Array,
    thrust_min: Array | float,
    vmotor2thrust: Array,
    v_batt: Array | float,
    pwm_idle: Array | float,
    pwm_max: Array | float,
) -> Array:
    """Requested per-motor forces -> the rotor speeds (RPM) the firmware's compensated motor path drives.

    [compensated_motor_forces][crazyflow.control.transform.compensated_motor_forces], then the rotor speed at
    which ``rpm2thrust`` (the controller's, nominal, thrust curve: the motor's voltage-to-speed response) gives
    the delivered force. A motor that delivers nothing is stopped, at 0 rpm rather than the thrust curve's other
    zero; the inverse is evaluated on a non-negative force so that branch stays finite under differentiation.
    """
    xp = array_namespace(motor_forces)
    delivered = compensated_motor_forces(
        motor_forces, thrust_min=thrust_min, vmotor2thrust=vmotor2thrust, v_batt=v_batt, pwm_idle=pwm_idle,
        pwm_max=pwm_max,
    )
    rpm = motor_force2rotor_vel(xp.maximum(delivered, 0.0), rpm2thrust)
    return xp.where(delivered > 0.0, rpm, 0.0)


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
