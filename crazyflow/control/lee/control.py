"""Lee geometric controller reimplementation based on the Crazyflie firmware.

Only the attitude stage is implemented for now: ``attitude2force_torque`` maps an
[roll, pitch, yaw, collective thrust] setpoint to a collective force and body torques in SI units,
which the shared ``force_torque2rotor_vel`` mixer (crazyflow/control/mellinger/control.py) turns
into rotor speeds. It is a drop-in alternative to the Mellinger attitude stage: same command layout,
same pipeline slot, selected with ``Sim(attitude_controller="lee")``.

We replicate the FIRMWARE (src/modules/src/controller/controller_lee.c, K. Wahba, W. Hoenig,
D. Schmidt, 2023-24), not the paper, so that gains transfer to the vehicle unchanged:
- rotation error eR = 0.5 * (R_des^T R - R^T R_des)^vee (the firmware keeps the 0.5, unlike its
  Mellinger),
- angular velocity error e_omega = omega - R^T R_des omega_des,
- torque u = -KR * eR - Komega * e_omega - KI * int(eR) dt + omega x (J omega); the paper's
  higher-order -J(omega^ R^T R_des omega_des - R^T R_des alpha_des) term is absent in the firmware
  and therefore here,
- the attitude integral is not clamped,
- below a thrust of 0.01 N the firmware zeroes its output and resets the integrators (manual branch:
  setpoint thrust < 1000/65535 of full scale); we do the same.

Deliberately NOT replicated: the firmware's manual branch negates the pitch setpoint for its
"legacy coordinate system". crazyflow's attitude command is interpreted the same way for every
attitude controller (extrinsic xyz Euler, see ``euler_xyz_to_rot_mat``), so the sign convention is
the setpoint writer's job, exactly as for the Mellinger stage, which omits the firmware's axis flip.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on
SE(3)", CDC 2010; the omega_des construction from the setpoint jerk (position stage, not yet
ported) follows D. Mellinger and V. Kumar, ICRA 2011.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from array_api_compat import array_namespace
from flax.struct import dataclass, field

from crazyflow.control.core import controllable, load_params
from crazyflow.control.transform import euler_xyz_to_rot_mat, quat_to_rot_mat
from crazyflow.utils import leaf_replace

if TYPE_CHECKING:
    from jax import Device

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.sim.data import SimData

#: Thrust in N below which the firmware zeroes its output and resets the integrators
#: (controller_lee.c: ``thrustSi < 0.01`` in the position branch; ``setpoint->thrust < 1000``,
#: i.e. 0.0097 N, in the manual branch).
THRUST_RESET_N = 0.01


def attitude2force_torque(
    quat: Array,
    ang_vel: Array,
    cmd: Array,
    i_error_att: Array | None = None,
    ctrl_freq: int = 500,
    ang_vel_des: Array | None = None,
    *,
    KR: Array,
    Komega: Array,
    KI: Array,
    J: Array,
) -> tuple[Array, Array, Array]:
    """Attitude stage of the Lee geometric controller (firmware ``controllerLee``, manual branch).

    All controllers are implemented as pure functions. Therefore, integral errors have to be passed
    as an argument and returned as well.

    Args:
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        ang_vel: Drone angular velocity in the body frame in rad/s with shape (..., 3).
        cmd: Commanded attitude (roll, pitch, yaw) and collective thrust [rad, rad, rad, N] with
            shape (..., 4).
        i_error_att: Integral of the rotation error (..., 3) from the previous call. If None, it is
            initialised to zero.
        ctrl_freq: Control frequency in Hz.
        ang_vel_des: Desired angular velocity expressed in the DESIRED body frame, shape (..., 3).
            None means zero, which is what the firmware's attitude-setpoint path produces (its
            feed-forward comes from the setpoint jerk and yaw rate, both absent here).
        KR: Rotation error gain with shape (3,), N m/rad.
        Komega: Angular velocity error gain with shape (3,), N m s/rad.
        KI: Rotation integral gain with shape (3,), N m/(rad s).
        J: Diagonal inertia with shape (3,), kg m^2 (the firmware stores J as a vector).

    Returns:
        Collective force in N with shape (..., 1), body torques in N m with shape (..., 3), and the
        updated integral error with shape (..., 3).
    """
    xp = array_namespace(quat)
    force_des = cmd[..., 3]
    rpy_des = cmd[..., :3]
    dt = 1 / ctrl_freq
    # l. 155 ff (manual branch): R_des from the rpy setpoint; l. 167 ff: rotation error
    rot_mat = quat_to_rot_mat(quat)
    rot_des_mat = euler_xyz_to_rot_mat(rpy_des)
    eRM = rot_des_mat.mT @ rot_mat - rot_mat.mT @ rot_des_mat
    eR = 0.5 * xp.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
    # l. 190 ff: omega_r = R^T R_des omega_des, omega_error = omega - omega_r
    ang_vel_des = xp.zeros_like(ang_vel) if ang_vel_des is None else ang_vel_des
    omega_r = ((rot_mat.mT @ rot_des_mat) @ ang_vel_des[..., None])[..., 0]
    e_omega = ang_vel - omega_r
    # l. 205: integral of the rotation error, no clamp in the firmware
    i_error_att = xp.zeros_like(ang_vel) if i_error_att is None else i_error_att
    i_error_att = i_error_att + dt * eR
    # l. 209 ff: u = -KR eR - Komega e_omega - KI int(eR) + omega x (J omega)
    gyro = xp.linalg.cross(ang_vel, J * ang_vel, axis=-1)
    torque = -KR * eR - Komega * e_omega - KI * i_error_att + gyro
    # l. 146 ff: below the thrust threshold the firmware outputs zero and resets its integrators
    active = force_des > THRUST_RESET_N
    torque = xp.where(active[..., None], torque, 0.0)
    i_error_att = xp.where(active[..., None], i_error_att, 0.0)
    force = xp.where(active, force_des, 0.0)[..., None]
    return force, torque, i_error_att


@dataclass
class LeeAttitudeData:
    cmd: Array  # (N, M, 4)
    """Attitude control command for the drone: [roll, pitch, yaw, collective thrust]."""
    staged_cmd: Array  # (N, M, 4)
    """Staging buffer to store the most recent command until the next controller tick."""
    steps: Array  # (N, 1)
    """Last simulation steps that the attitude control command was applied."""
    freq: int = field(pytree_node=False)
    """Frequency of the attitude control command."""
    i_error_att: Array  # (N, M, 3)
    """Integral of the rotation error."""
    # Parameters for the attitude controller
    params: dict[str, Array]

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, freq: int, drone: str, device: Device
    ) -> LeeAttitudeData:
        """Create a default set of attitude data for the simulation."""
        cmd = jnp.zeros((n_worlds, n_drones, 4), device=device)
        steps = -jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device)
        i_error_att = jnp.zeros((n_worlds, n_drones, 3), device=device)
        params = load_params(attitude2force_torque, drone, xp=jnp, device=device)
        return LeeAttitudeData(
            cmd=cmd, staged_cmd=cmd, steps=steps, freq=freq, i_error_att=i_error_att, params=params
        )


def control_attitude2force_torque(data: SimData) -> SimData:
    """Compute the updated controls for the Lee attitude controller."""
    states = data.states
    attitude_ctrl: LeeAttitudeData = data.controls.attitude
    assert attitude_ctrl is not None, "Using attitude controller without initialized data"
    mask = controllable(data.core.steps, data.core.freq, attitude_ctrl.steps, attitude_ctrl.freq)
    attitude_ctrl = leaf_replace(attitude_ctrl, mask, cmd=attitude_ctrl.staged_cmd)
    force, torque, i_error_att = attitude2force_torque(
        states.quat,
        states.ang_vel,
        attitude_ctrl.cmd,
        i_error_att=attitude_ctrl.i_error_att,
        ctrl_freq=attitude_ctrl.freq,
        **attitude_ctrl.params,
    )
    attitude_ctrl = leaf_replace(
        attitude_ctrl, mask, i_error_att=i_error_att, steps=data.core.steps
    )
    ft_ctrl = leaf_replace(
        data.controls.force_torque, mask, staged_cmd=jnp.concat([force, torque], axis=-1)
    )
    return data.replace(
        states=states, controls=data.controls.replace(attitude=attitude_ctrl, force_torque=ft_ctrl)
    )
