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
def test_firmware_torque_clip():
    """cf21B_500 runs the firmware rate loop: each axis saturates at the firmware's int16-equivalent torque."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    clip = np.asarray(ctrl.keywords["torque_clip"])
    _, torque, _ = ctrl(np.zeros(3), np.array([500.0, -500.0, 500.0, 0.4]))
    assert np.allclose(np.abs(torque), clip), f"Saturated command must sit at the clip: {torque} vs {clip}"


@pytest.mark.unit
def test_no_torque_clip_by_default():
    """Without torque_clip (drones with no firmware rate loop) a saturating demand reaches the mixer unclipped."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    err = np.array([500.0, -500.0, 500.0])
    _, torque, _ = ctrl(np.zeros(3), np.array([*err, 0.4]), torque_clip=None)
    J, kp, ki = (np.asarray(ctrl.keywords[k]) for k in ("J", "kp", "ki"))
    int_err_max = np.asarray(ctrl.keywords["int_err_max"])
    expected = J @ (kp * err + ki * np.clip(err / 500, -int_err_max, int_err_max))   # one integration step at 500 Hz
    assert np.allclose(torque, expected), f"Torque was altered: {torque} vs {expected}"


@pytest.mark.unit
def test_zero_thrust_resets_integral():
    """At zero thrust the firmware zeroes its outputs and resets its PIDs; cf21B_500 does the same."""
    ctrl = parametrize(body_rate2force_torque, "cf21B_500")
    _, torque, int_err = ctrl(np.zeros(3), np.array([1.0, 1.0, 1.0, 0.0]), ang_vel_err_i=np.array([0.3, -0.2, 0.1]))
    assert np.all(torque == 0) and np.all(int_err == 0), f"Not reset at zero thrust: {torque}, {int_err}"


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


def _lpf2p_firmware(samples: np.ndarray, sample_freq: float, cutoff_freq: float) -> np.ndarray:
    """crazyflie-firmware src/utils/src/filter.c lpf2pSetCutoffFreq + lpf2pApply, transcribed line by line."""
    fr = sample_freq / cutoff_freq
    ohm = np.tan(np.pi / fr)
    c = 1.0 + 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm
    b0 = ohm * ohm / c
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (ohm * ohm - 1.0) / c
    a2 = (1.0 - 2.0 * np.cos(np.pi / 4.0) * ohm + ohm * ohm) / c
    de1 = de2 = 0.0
    out = []
    for sample in samples:
        de0 = sample - de1 * a1 - de2 * a2
        if not np.isfinite(de0):
            de0 = sample
        out.append(de0 * b0 + de1 * b1 + de2 * b2)
        de2 = de1
        de1 = de0
    return np.array(out)


@pytest.mark.unit
@pytest.mark.parametrize("sample_freq", [500.0, 1000.0])
def test_gyro_lpf2p_matches_firmware(sample_freq: float):
    """gyro_lpf2p is the firmware's lpf2p, sample by sample, per axis."""
    from crazyflow.control.body_rate import gyro_lpf2p

    rng = np.random.default_rng(0)
    w = rng.normal(size=(400, 3)) + np.sin(np.arange(400) / 20.0)[:, None]
    d1 = d2 = np.zeros(3)
    out = []
    for x in w:
        y, d1, d2 = gyro_lpf2p(x, d1, d2, sample_freq, gyro_lpf_hz=np.asarray(80.0))
        out.append(y)
    ref = np.stack([_lpf2p_firmware(w[:, i], sample_freq, 80.0) for i in range(3)], -1)
    assert np.allclose(np.stack(out), ref, atol=1e-12), "gyro_lpf2p differs from filter.c"


@pytest.mark.unit
def test_gyro_lpf2p_off_and_params():
    """No cutoff passes the rate through; only the cf21B_500 (the firmware port) filters, at 80 Hz."""
    from crazyflow.control.body_rate import gyro_lpf2p
    from crazyflow.control.core import load_params

    w, d = np.array([1.0, -2.0, 3.0]), np.array([0.5, 0.5, 0.5])
    y, d1, d2 = gyro_lpf2p(w, d, 2 * d, 500.0)
    assert np.array_equal(y, w) and np.array_equal(d1, d) and np.array_equal(d2, 2 * d)
    for drone in available_drones:
        params = load_params(gyro_lpf2p, drone)
        if drone == "cf21B_500":
            assert float(params["gyro_lpf_hz"]) == 80.0
        else:
            assert "gyro_lpf_hz" not in params, f"{drone} is not a firmware port and reads the true rate"


@pytest.mark.unit
def test_rate_pid_reads_filtered_rate():
    """In the sim the PID's last rate is the filtered one: after a rate step it lags the true rate."""
    sim = Sim(n_worlds=1, n_drones=1, drone="cf21B_500", control=Control.body_rate, freq=500)
    sim.reset()
    cmd = settled_hover(sim)
    cmd[0, 0, 0] = RATE_CMD[0]
    for _ in range(10):
        sim.body_rate_control(cmd)
        sim.step(1)
    true_rate = float(sim.data.states.ang_vel[0, 0, 0])
    seen = float(sim.data.controls.body_rate.last_ang_vel[0, 0, 0])
    assert 0.0 < seen < true_rate, f"filtered rate {seen} should trail the rising true rate {true_rate}"
    sim.reset()
    assert np.all(np.asarray(sim.data.controls.body_rate.gyro_d1) == 0.0), "reset must clear the filter state"
