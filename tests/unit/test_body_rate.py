from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control, parametrize
from crazyflow.control.body_rate import body_rate2force_torque
from crazyflow.drones import available_drones
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim

RATE_CMD = np.array([1.745, -1.745, 3.49])  # 100 deg/s roll and pitch, 200 deg/s yaw


def settled_hover(sim: Sim, seconds: float = 0.8) -> np.ndarray:
    """Spin the rotors up to hover and return the hover command.

    The simulation resets with ``rotor_vel`` at zero, so a step applied straight after ``reset``
    measures the rotors spinning up from standstill rather than the rate loop.
    """
    mass = float(sim.data.params.mass[0, 0, 0])
    cmd = np.zeros((1, 1, 4), dtype=np.float32)
    cmd[0, 0, 3] = mass * 9.81
    for _ in range(int(seconds * sim.freq)):
        sim.body_rate_control(cmd)
        sim.step(1)
    return cmd


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_shapes_and_batching(drone: str):
    """Leading batch dims pass through, force is (..., 1) and torque (..., 3)."""
    ctrl = parametrize(body_rate2force_torque, drone)
    for batch in [(), (5,), (4, 3)]:
        force, torque, int_err = ctrl(np.zeros((*batch, 3)), np.zeros((*batch, 4)))
        assert force.shape == (*batch, 1), f"force shape {force.shape} for batch {batch}"
        assert torque.shape == (*batch, 3), f"torque shape {torque.shape} for batch {batch}"
        assert int_err.shape == (*batch, 3), f"int err shape {int_err.shape} for batch {batch}"


@pytest.mark.unit
def test_thrust_passthrough_and_zero_thrust_gate():
    """Thrust is forwarded untouched, and zero thrust gates the torque off."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    force, torque, _ = ctrl(np.zeros(3), np.array([1.0, 1.0, 1.0, 0.42]))
    assert np.isclose(force[0], 0.42), "Collective thrust must pass through unchanged"
    assert np.any(torque != 0), "A rate error at positive thrust must produce torque"
    # A drone that is not commanded to lift off must not be torqued, matching the Mellinger gate
    force, torque, _ = ctrl(np.zeros(3), np.array([1.0, 1.0, 1.0, 0.0]))
    assert np.all(torque == 0), "Torque must be gated off at zero thrust"


@pytest.mark.unit
def test_derivative_has_no_kick_on_first_call():
    """``prev_ang_vel=None`` must not manufacture an angular acceleration."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    ang_vel = np.array([2.0, -3.0, 1.0])  # Already rotating fast when first called
    _, torque_fresh, _ = ctrl(ang_vel, np.array([2.0, -3.0, 1.0, 0.4]))
    # Zero rate error and zero derivative => zero torque, no matter how fast the drone spins
    msg = f"Derivative kick on first call: {torque_fresh}"
    assert np.allclose(torque_fresh, 0.0, atol=1e-12), msg


@pytest.mark.unit
def test_torque_not_clipped():
    """No torque-space clip: as in the firmware, a saturating demand reaches the mixer unclipped."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    err = np.array([500.0, -500.0, 500.0])
    _, torque, _ = ctrl(np.zeros(3), np.array([*err, 0.4]))
    J, kp, ki = (np.asarray(ctrl.keywords[k]) for k in ("J", "kp", "ki"))
    expected = J @ (kp * err + ki * np.clip(err / 500, -1e9, 1e9))   # one step of integration at 500 Hz
    assert np.allclose(torque, expected), f"Torque was altered: {torque} vs {expected}"


@pytest.mark.unit
def test_integral_bounded_by_int_err_max():
    """A sustained saturating error charges the integrator up to int_err_max and no further."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    int_err_max = np.asarray(ctrl.keywords["int_err_max"])
    int_err = None
    for _ in range(int(2 * int_err_max[0] / (500.0 / 500)) + 10):
        _, _, int_err = ctrl(np.zeros(3), np.array([500.0, 0.0, 0.0, 0.4]), ang_vel_err_i=int_err)
    assert np.isclose(float(int_err[0]), int_err_max[0]), f"Integral not bounded at int_err_max: {int_err}"


@pytest.mark.unit
def test_integral_charges_when_not_saturated():
    """Ordinary integration: the integral accumulates err * dt."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    int_err = None
    # A small error, well inside the torque the mixer can deliver
    for _ in range(20):
        _, _, int_err = ctrl(np.zeros(3), np.array([0.0, 0.0, 0.02, 0.4]), ang_vel_err_i=int_err)
    assert float(int_err[2]) > 0.0, f"Integral never charged: {int_err}"
    assert np.isclose(float(int_err[2]), 0.02 * 20 / 500), "Integral did not accumulate err * dt"


@pytest.mark.unit
def test_integral_removes_steady_state_error():
    """With ki active the loop must reach the setpoint, not just get close to it."""
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.body_rate, freq=500)
    cmd = settled_hover(sim)
    cmd[0, 0, :3] = RATE_CMD
    for _ in range(sim.freq):  # One second, well past the measured settling time
        sim.body_rate_control(cmd)
        sim.step(1)
    ang_vel = np.asarray(sim.data.states.ang_vel[0, 0])
    assert np.allclose(ang_vel, RATE_CMD, atol=0.1), f"Steady state error too large: {ang_vel}"


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_step_response_tracks_each_axis(drone: str):
    """Each axis must track its own setpoint without driving the other two."""
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.body_rate, drone=drone, freq=500)
    for axis in range(3):
        sim.reset()
        cmd = settled_hover(sim)
        cmd[0, 0, axis] = RATE_CMD[axis]
        peak = 0.0
        for _ in range(sim.freq):
            sim.body_rate_control(cmd)
            sim.step(1)
            peak = max(peak, abs(float(sim.data.states.ang_vel[0, 0, axis])))
        ang_vel = np.asarray(sim.data.states.ang_vel[0, 0])
        target = RATE_CMD[axis]
        msg = f"{drone} axis {axis}: {ang_vel[axis]} != {target}"
        assert abs(ang_vel[axis] - target) < 0.1, msg
        # Overshoot stays bounded. The measured worst case across drones is ~10%.
        assert peak < abs(target) * 1.35, f"{drone} axis {axis} overshoot {peak / abs(target):.2f}"


@pytest.mark.unit
def test_sign_convention_matches_body_frame():
    """A positive roll-rate command must produce a positive body-frame roll rate, and so on."""
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.body_rate, freq=500)
    for axis in range(3):
        sim.reset()
        cmd = settled_hover(sim)
        cmd[0, 0, axis] = 1.0
        for _ in range(sim.freq // 4):
            sim.body_rate_control(cmd)
            sim.step(1)
        ang_vel = np.asarray(sim.data.states.ang_vel[0, 0])
        assert ang_vel[axis] > 0.5, f"Axis {axis} did not follow a positive command: {ang_vel}"
        others = [ang_vel[i] for i in range(3) if i != axis]
        assert all(abs(o) < 0.5 for o in others), f"Axis {axis} bled into the others: {ang_vel}"


@pytest.mark.unit
def test_jit_and_grad():
    """The controller must survive jit and produce finite gradients."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500", xp=jnp)

    def cost(cmd: jnp.ndarray) -> jnp.ndarray:
        _, torque, _ = ctrl(jnp.zeros(3), cmd)
        return jnp.sum(torque**2)

    grad = jax.jit(jax.grad(cost))(jnp.array([0.1, 0.1, 0.1, 0.4]))
    assert not jnp.any(jnp.isnan(grad)), "NaN gradient"
    assert jnp.any(grad != 0), "Gradient vanished away from any saturation bound"
