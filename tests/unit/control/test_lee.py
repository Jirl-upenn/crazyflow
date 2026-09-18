from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control, load_params, parametrize
from crazyflow.control.lee import THRUST_RESET_N, attitude2force_torque
from crazyflow.control.mellinger import attitude2force_torque as mellinger_attitude2force_torque
from crazyflow.control.transform import euler_xyz_to_rot_mat, quat_to_rot_mat
from crazyflow.drones import available_drones

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_euler("xyz", rpy).as_quat()


def create_rnd_states(shape: tuple[int, ...] = ()) -> tuple[Array, Array]:
    quat = _quat_from_rpy(np.random.uniform(-1.0, 1.0, (*shape, 3)))
    ang_vel = np.random.randn(*shape, 3)
    return quat, ang_vel


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_shapes(drone: str) -> None:
    controller = parametrize(attitude2force_torque, drone)
    quat, ang_vel = create_rnd_states()
    force, torque, i_error = controller(quat, ang_vel, np.array([0.1, 0.1, 0.1, 0.3]))
    assert force.shape == (1,)
    assert torque.shape == (3,)
    assert i_error.shape == (3,)
    quat, ang_vel = create_rnd_states((5, 4))
    cmd = np.random.randn(5, 4, 4)
    cmd[..., 3] = np.abs(cmd[..., 3]) + 0.1
    force, torque, i_error = controller(quat, ang_vel, cmd)
    assert force.shape == (5, 4, 1)
    assert torque.shape == (5, 4, 3)
    assert i_error.shape == (5, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_at_setpoint(drone: str) -> None:
    """At the commanded attitude with zero rates the torque is zero and the thrust passes through."""
    controller = parametrize(attitude2force_torque, drone)
    rpy = np.array([0.2, -0.1, 0.5])
    quat = _quat_from_rpy(rpy)
    force, torque, i_error = controller(quat, np.zeros(3), np.array([*rpy, 0.3]))
    assert np.allclose(torque, 0.0, atol=1e-7)
    assert np.allclose(i_error, 0.0, atol=1e-7)
    assert np.isclose(force[0], 0.3)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_restoring_sign(drone: str) -> None:
    """A positive roll error gives a negative roll torque; same sign convention as the Mellinger
    stage, which the shared mixer and the first-principles dynamics were validated against."""
    lee = parametrize(attitude2force_torque, drone)
    mel = parametrize(mellinger_attitude2force_torque, drone)
    cmd = np.array([0.0, 0.0, 0.0, 0.3])
    for axis in range(3):
        rpy = np.zeros(3)
        rpy[axis] = 0.1
        quat = _quat_from_rpy(rpy)
        _, torque_lee, _ = lee(quat, np.zeros(3), cmd)
        _, torque_mel, _ = mel(quat, np.zeros(3), cmd)
        assert torque_lee[axis] < 0, f"Lee axis {axis}: {torque_lee}"
        assert np.sign(torque_lee[axis]) == np.sign(torque_mel[axis]), (torque_lee, torque_mel)
    # rate damping: positive body rate about an axis at the setpoint gives negative torque
    for axis in range(3):
        ang_vel = np.zeros(3)
        ang_vel[axis] = 1.0
        _, torque_lee, _ = lee(_quat_from_rpy(np.zeros(3)), ang_vel, cmd)
        assert torque_lee[axis] < 0


@pytest.mark.unit
def test_attitude2force_torque_matches_firmware_law() -> None:
    """Independent numpy transcription of controllerLee (manual branch) on random states."""
    drone = "cf21B_500"
    params = load_params(attitude2force_torque, drone)
    controller = parametrize(attitude2force_torque, drone)
    quat, ang_vel = create_rnd_states((7,))
    cmd = np.random.randn(7, 4)
    cmd[:, 3] = np.abs(cmd[:, 3]) + 0.1
    i0 = np.random.randn(7, 3) * 0.01
    force, torque, i1 = controller(quat, ang_vel, cmd, i_error_att=i0, ctrl_freq=500)
    KR, Kw, KI, J = (np.asarray(params[k]) for k in ("KR", "Komega", "KI", "J"))
    for k in range(7):
        R = np.asarray(quat_to_rot_mat(quat[k]))
        Rd = np.asarray(euler_xyz_to_rot_mat(cmd[k, :3]))
        eRM = Rd.T @ R - R.T @ Rd
        eR = 0.5 * np.array([eRM[2, 1], eRM[0, 2], eRM[1, 0]])
        i_att = i0[k] + eR / 500
        w = ang_vel[k]
        u = -KR * eR - Kw * w - KI * i_att + np.cross(w, J * w)
        assert np.allclose(torque[k], u, atol=1e-6), (torque[k], u)
        assert np.allclose(i1[k], i_att, atol=1e-7)
        assert np.isclose(force[k, 0], cmd[k, 3])


@pytest.mark.unit
def test_attitude2force_torque_thrust_reset() -> None:
    """Below the firmware's thrust threshold the output is zero and the integrator resets."""
    controller = parametrize(attitude2force_torque, "cf21B_500")
    quat, ang_vel = create_rnd_states()
    cmd = np.array([0.3, 0.2, 0.1, THRUST_RESET_N / 2])
    force, torque, i_error = controller(quat, ang_vel, cmd, i_error_att=np.ones(3))
    assert np.all(force == 0) and np.all(torque == 0) and np.all(i_error == 0)


@pytest.mark.unit
def test_attitude2force_torque_batch_consistency() -> None:
    controller = parametrize(attitude2force_torque, "cf21B_500")
    quat, ang_vel = create_rnd_states((6,))
    cmd = np.random.randn(6, 4)
    cmd[:, 3] = np.abs(cmd[:, 3]) + 0.1
    fb, tb, ib = controller(quat, ang_vel, cmd)
    for k in range(6):
        f, t, i = controller(quat[k], ang_vel[k], cmd[k])
        assert np.allclose(fb[k], f) and np.allclose(tb[k], t) and np.allclose(ib[k], i)


@pytest.mark.unit
def test_attitude2force_torque_jit_and_grad() -> None:
    controller = parametrize(attitude2force_torque, "cf21B_500", xp=jnp)
    quat, ang_vel = (jnp.asarray(x) for x in create_rnd_states((3,)))
    cmd = jnp.asarray(np.array([[0.1, 0.0, 0.0, 0.3]] * 3))
    torque_fn = jax.jit(lambda q, w, c: controller(q, w, c)[1])
    torque = torque_fn(quat, ang_vel, cmd)
    assert torque.shape == (3, 3)
    grad = jax.grad(lambda c: jnp.sum(torque_fn(quat, ang_vel, c) ** 2))(cmd)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.unit
@pytest.mark.parametrize("attitude_controller", ["mellinger", "lee"])
def test_sim_attitude_hover(attitude_controller: str) -> None:
    """Both attitude stages hold a hover from rest for one second on the first-principles plant."""
    from crazyflow.sim import Sim
    from crazyflow.sim.functional import attitude_control
    from crazyflow.drones import load_params as load_drone_params
    from crazyflow.dynamics import Dynamics

    sim = Sim(
        n_worlds=2, n_drones=1, drone="cf21B_500", dynamics=Dynamics.first_principles,
        control=Control.attitude, attitude_controller=attitude_controller, freq=500,
    )
    assert sim.data.controls.attitude.__class__.__name__ == (
        "LeeAttitudeData" if attitude_controller == "lee" else "MellingerAttitudeData"
    )
    sim.reset()
    mass = float(load_drone_params("cf21B_500")["mass"])
    hover = jnp.tile(jnp.array([0.0, 0.0, 0.0, mass * 9.81]), (2, 1, 1))
    pos0 = np.asarray(sim.data.states.pos)
    for _ in range(100):
        sim.data = attitude_control(sim.data, hover)
        sim.step(5)
    pos = np.asarray(sim.data.states.pos)
    rpy_err = np.abs(np.asarray(sim.data.states.quat)[..., :3]).max()
    assert np.abs(pos - pos0).max() < 0.15, pos - pos0
    assert rpy_err < 0.05, rpy_err
