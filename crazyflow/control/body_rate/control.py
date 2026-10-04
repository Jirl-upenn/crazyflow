"""Collective-thrust body-rate (CTBR) controller.

A single-stage PID rate controller that turns a body-rate setpoint plus a collective thrust into
the force and torque expected by
[force_torque2rotor_vel][crazyflow.control.mellinger.force_torque2rotor_vel]. It is a sibling of
the Mellinger attitude stage rather than a replacement: both consume a setpoint and emit
force/torque, so the mixer stage below them is shared.

``body_rate2force_torque`` → ``force_torque2rotor_vel``

Unlike the Mellinger controller, this controller is not a reimplementation of anything running on
the Crazyflie firmware. It works entirely in SI units, and its gains are expressed as angular
accelerations per unit rate error, so the commanded torque is ``J @ (kp e + ki ∫e - kd dω/dt)``.
Multiplying by the inertia at the end is what makes the gains roughly platform independent: they
set the closed-loop rate bandwidth in rad/s regardless of how heavy the drone is.

CTBR is the usual action interface for reinforcement learning on quadrotors, because a rate
setpoint is far easier to learn than a torque and far more transferable than a motor command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from flax.struct import dataclass, field

from crazyflow.control.core import controllable, load_params
from crazyflow.utils import leaf_replace, to_xp

if TYPE_CHECKING:
    from jax import Device

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.sim.data import SimData


def body_rate2force_torque(
    ang_vel: Array,
    cmd: Array,
    ang_vel_err_i: Array | None = None,
    prev_ang_vel: Array | None = None,
    ctrl_freq: int = 500,
    *,
    J: Array,
    kp: Array,
    ki: Array,
    kd: Array,
    int_err_max: Array,
) -> tuple[Array, Array, Array]:
    r"""Compute the force and torque commanded by a body-rate PID controller.

    All controllers are implemented as pure functions. Therefore, the integral error and the
    previous angular velocity have to be passed as arguments and returned as well.

    The derivative term acts on the measurement, not on the error
    (:math:`-k_d\,\dot{\omega}` rather than :math:`k_d\,\dot{e}`), so a step change of the setpoint
    does not produce a derivative kick. With a constant setpoint the two are identical.

    Note:
        The commanded collective thrust and the torque are passed on unclipped. Saturation happens
        one stage later, in
        [force_torque2rotor_vel][crazyflow.control.mellinger.force_torque2rotor_vel], which clips
        the individual motor forces -- as the Crazyflie firmware does, whose rate PIDs only
        saturate their outputs to the int16 range before power distribution clips the motors.
        The only windup bound is `int_err_max`, again as in the firmware.

    Warning:
        ``ang_vel`` is in the body frame, and so is the commanded rate. ``cmd`` puts the thrust
        last, matching the ``[roll, pitch, yaw, thrust]`` layout of the attitude command.

    Args:
        ang_vel: Angular velocity of the drone in the body frame in rad/s with shape (..., 3).
        cmd: Commanded body rates and collective thrust
            [roll_rate, pitch_rate, yaw_rate, thrust] with shape (..., 4) in [rad/s, rad/s, rad/s,
            N].
        ang_vel_err_i: Angular velocity integral error (..., 3) from the previous call. If None, it
            is initialised to zero.
        prev_ang_vel: Angular velocity (..., 3) from the previous call, used for the derivative
            term. If None, it is initialised to ``ang_vel``, which makes the derivative term zero
            on the first call instead of producing a spike.
        ctrl_freq: Control frequency in Hz. Scales the integral and derivative terms only, so it
            has to match the rate at which this function is actually called.
        J: Inertia matrix the controller assumes, with shape (3, 3) in kg m². This is the
            controller's model of the drone and does not have to match the true inertia used by the
            dynamics.
        kp: Proportional gain on the rate error with shape (3,), in 1/s.
        ki: Integral gain on the rate error with shape (3,), in 1/s².
        kd: Derivative gain on the measured angular acceleration with shape (3,), dimensionless.
        int_err_max: Range of the integral error with shape (3,) in rad.

    Returns:
        The collective force with shape (..., 1) in N, the body torque with shape (..., 3) in Nm,
        and the updated integral error with shape (..., 3).

    Example:
    ```python
    import numpy as np
    from crazyflow.control import parametrize
    from crazyflow.control.body_rate import body_rate2force_torque

    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    ang_vel = np.zeros(3)
    cmd = np.array([0.0, 0.0, 0.0, 0.4])  # Hold zero rates at 0.4 N collective thrust
    force, torque, ang_vel_err_i = ctrl(ang_vel, cmd)
    ```
    """
    xp = array_namespace(ang_vel)
    device = xp_device(ang_vel)
    J, kp, ki, kd = to_xp(J, kp, ki, kd, xp=xp, device=device)
    int_err_max = to_xp(int_err_max, xp=xp, device=device)

    ang_vel_des = cmd[..., :3]
    force_des = cmd[..., 3]
    dt = 1 / ctrl_freq

    ang_vel_err = ang_vel_des - ang_vel
    ang_vel_err_i = xp.zeros_like(ang_vel) if ang_vel_err_i is None else ang_vel_err_i
    ang_vel_err_i = xp.clip(ang_vel_err_i + ang_vel_err * dt, -int_err_max, int_err_max)
    # Derivative on the measurement rather than on the error, so that stepping the setpoint does
    # not kick the derivative term.
    prev_ang_vel = ang_vel if prev_ang_vel is None else prev_ang_vel
    ang_acc = (ang_vel - prev_ang_vel) / dt

    ang_acc_des = kp * ang_vel_err + ki * ang_vel_err_i - kd * ang_acc
    torque = (J @ ang_acc_des[..., None])[..., 0]
    # Do not torque the drone while it is not commanded to produce any thrust. The Mellinger
    # attitude controller gates on the same condition.
    torque = xp.where((force_des > 0)[..., None], torque, 0.0)
    return force_des[..., None], torque, ang_vel_err_i


@dataclass
class BodyRateData:
    cmd: Array  # (N, M, 4)
    """Body-rate control command for the drone.

    A command consists of [roll_rate, pitch_rate, yaw_rate, collective thrust].
    """
    staged_cmd: Array  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the body-rate control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the body-rate control command."""
    ang_vel_err_i: Array  # (N, M, 3)
    """Integral error of the body-rate controller."""
    last_ang_vel: Array  # (N, M, 3)
    """Last angular velocity of the drone."""
    # Parameters for the body-rate controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> BodyRateData:
        """Create a default set of body-rate data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        zeros_3d = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = load_params(body_rate2force_torque, drone, xp=jnp, device=device)
        return BodyRateData(
            cmd=cmd,
            staged_cmd=cmd,
            steps=steps,
            freq=freq,
            ang_vel_err_i=zeros_3d,
            last_ang_vel=zeros_3d,
            params=params,
        )


def control_body_rate2force_torque(data: SimData) -> SimData:
    """Compute the updated controls for the body-rate controller."""
    states = data.states
    body_rate_ctrl: BodyRateData = data.controls.body_rate
    assert body_rate_ctrl is not None, "Using body rate controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, body_rate_ctrl.steps, body_rate_ctrl.freq)
    body_rate_ctrl = leaf_replace(body_rate_ctrl, mask, cmd=body_rate_ctrl.staged_cmd)
    force, torque, ang_vel_err_i = body_rate2force_torque(
        states.ang_vel,
        body_rate_ctrl.cmd,
        ang_vel_err_i=body_rate_ctrl.ang_vel_err_i,
        prev_ang_vel=body_rate_ctrl.last_ang_vel,
        ctrl_freq=body_rate_ctrl.freq,
        **body_rate_ctrl.params,
    )
    body_rate_ctrl = leaf_replace(
        body_rate_ctrl,
        mask,
        ang_vel_err_i=ang_vel_err_i,
        last_ang_vel=states.ang_vel,
        steps=data.core.steps,
    )
    ft_ctrl = leaf_replace(
        data.controls.force_torque, mask, staged_cmd=jnp.concat([force, torque], axis=-1)
    )
    return data.replace(
        controls=data.controls.replace(body_rate=body_rate_ctrl, force_torque=ft_ctrl)
    )
