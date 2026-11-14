"""The body-rate controller's power distribution (crazyflow.control.body_rate.power_distribution)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control, parametrize
from crazyflow.control.body_rate import control_power_distribution, power_distribution
from crazyflow.control.core import load_params
from crazyflow.control.transform import compensated_motor_forces, vmotor2thrust_poly
from crazyflow.drones import available_drones
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.sim import build_control_fns

P = load_params(power_distribution, "cf21B_500")
C = np.asarray(P["vmotor2thrust"], dtype=np.float64)
UINT16_MAX = 65535.0


def firmware_motor(f_req: np.ndarray, v_batt: float, pwm_idle: float) -> np.ndarray:
    """The firmware's own path, transcribed, for requests (..., n_motors): motorsCompensateBatteryVoltage (motors.c,
    Cardano inverse of the voltage cubic, ratio = voltage / supply) on each request, upstream powerDistributionCap
    (the highest ratio's excess over UINT16_MAX subtracted from every motor, then capMinThrust to idle), then the
    motor's voltage curve at ratio * supply."""
    thrust_min = float(P["thrust_min"])
    p = -C[2] / (3 * C[3])
    r = C[1] / (3 * C[3])
    q = p**3 + (C[2] * C[1] - 3 * C[3] * (C[0] - f_req)) / (6 * C[3] ** 2)
    qrp = np.sqrt(q * q + (r - p * p) ** 3)
    v_motor = np.cbrt(q + qrp) + np.cbrt(q - qrp) + p
    ratio = np.where(f_req < thrust_min, 0.0, UINT16_MAX * v_motor / v_batt)
    highest = np.maximum(ratio.max(-1, keepdims=True), 0.0)
    ratio = np.maximum(ratio - np.maximum(highest - UINT16_MAX, 0.0), pwm_idle)
    v = ratio / UINT16_MAX * v_batt
    return C[0] + C[1] * v + C[2] * v**2 + C[3] * v**3


@pytest.mark.unit
@pytest.mark.parametrize("v_batt", [3.3, 3.7, 4.1])
def test_compensated_cap_matches_firmware(v_batt: float):
    """Delivered force = the firmware's compensate -> cap -> motor chain, one motor at a time, over requests from
    negative to beyond the battery ceiling (the legacy mixer does not clip before compensation)."""
    f_req = np.linspace(-0.05, 0.45, 5001)[:, None]
    ref = firmware_motor(f_req, v_batt, float(P["pwm_idle"]))
    out = compensated_motor_forces(
        f_req, thrust_min=P["thrust_min"], vmotor2thrust=P["vmotor2thrust"], v_batt=v_batt,
        pwm_idle=P["pwm_idle"], pwm_max=P["pwm_max"],
    )
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("v_batt", [3.3, 3.7, 4.1])
def test_uniform_reduction_matches_firmware(v_batt: float):
    """Four coupled motors: when one asks for more than the battery gives, every motor's ratio drops by the same
    excess (upstream powerDistributionCap), the rest of the floor and idle behaviour unchanged."""
    rng = np.random.default_rng(1)
    f_req = rng.uniform(-0.05, 0.45, (4096, 4))
    ref = firmware_motor(f_req, v_batt, float(P["pwm_idle"]))
    out = compensated_motor_forces(
        f_req, thrust_min=P["thrust_min"], vmotor2thrust=P["vmotor2thrust"], v_batt=v_batt,
        pwm_idle=P["pwm_idle"], pwm_max=P["pwm_max"],
    )
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-9)
    ceil = vmotor2thrust_poly(np.float64(v_batt), C)
    capped = f_req.max(-1) > ceil
    assert 0.1 < capped.mean() < 0.95
    np.testing.assert_allclose(out[capped].max(-1), ceil, rtol=1e-9)          # the top motor sits at the ceiling
    np.testing.assert_allclose(out[~capped], np.where(f_req[~capped] < P["thrust_min"], ref[~capped], f_req[~capped]))


@pytest.mark.unit
def test_idle_and_ceiling_values():
    """cf21B_500 at 3.7 V: idle (7000 counts of the supply) ~3.2 mN, ceiling (full ratio) ~0.273 N; a request
    inside [THRUST_MIN, ceiling] is delivered exactly."""
    f = compensated_motor_forces(
        np.array([-1.0, 0.0, 0.02, 0.0214, 0.1, 0.2, 0.25, 1.0])[:, None], thrust_min=P["thrust_min"],
        vmotor2thrust=P["vmotor2thrust"], v_batt=3.7, pwm_idle=P["pwm_idle"], pwm_max=P["pwm_max"],
    )
    idle = vmotor2thrust_poly(7000.0 / 65535.0 * 3.7, C)
    ceil = vmotor2thrust_poly(np.float64(3.7), C)
    f = f[:, 0]
    np.testing.assert_allclose(f[:3], idle)
    np.testing.assert_allclose(f[3:7], [0.0214, 0.1, 0.2, 0.25])
    np.testing.assert_allclose(f[7], ceil)
    assert 0.003 < idle < 0.0035 and 0.27 < ceil < 0.28


def _rpm(f: float, rpm2thrust) -> float:
    k0, k1, k2 = np.asarray(rpm2thrust, dtype=np.float64)
    return (-k1 + np.sqrt(k1**2 - 4 * k2 * (k0 - f))) / (2 * k2)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_hover_and_shapes(drone: str):
    """Hover thrust and zero torque -> every rotor at the thrust curve's hover speed; batch dims pass through."""
    pd = parametrize(power_distribution, drone)
    p = load_params(power_distribution, drone)
    for batch in [(), (5,), (4, 3)]:
        rpm = pd(np.full((*batch, 1), 0.1), np.zeros((*batch, 3)))
        assert rpm.shape == (*batch, 4)
        np.testing.assert_allclose(rpm, _rpm(0.1 / 4, p["rpm2thrust"]), rtol=1e-6)


@pytest.mark.unit
def test_gradients_finite():
    """d rotor speed / d (force, torque) finite over thrusts and torques reaching both motor limits."""
    p = load_params(power_distribution, "cf21B_500", xp=jnp)

    def total(ft):
        return jnp.sum(power_distribution(ft[:1], ft[1:], **p))

    ft = jnp.stack(
        [jnp.array([fc, tx, ty, tz]) for fc in (0.0, 0.05, 0.4, 0.8, 1.2) for tx in (-5e-3, 0.0, 5e-3)
         for ty in (-5e-3, 0.0, 5e-3) for tz in (-2e-3, 0.0, 2e-3)]
    )
    g = jax.vmap(jax.grad(total))(ft)
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.unit
def test_body_rate_pipeline_uses_power_distribution():
    """The body-rate pipeline's mixer stage is the power distribution, under the force-torque stage's name."""
    stages = dict(build_control_fns(Control.body_rate, Dynamics.first_principles))
    assert stages["force_torque_controller"] is control_power_distribution



@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_matches_shared_mixer_inside_the_motor_range(drone: str):
    """Where every motor asks for a force in [THRUST_MIN, THRUST_MAX] the shared mixer and the power distribution
    agree on roll and pitch."""
    from crazyflow.control.mellinger import force_torque2rotor_vel

    pd = parametrize(power_distribution, drone)
    ft = parametrize(force_torque2rotor_vel, drone)
    thrust_max = float(load_params(force_torque2rotor_vel, drone)["thrust_max"])
    thrust_min = float(load_params(force_torque2rotor_vel, drone)["thrust_min"])
    rng = np.random.default_rng(0)
    # every motor inside [THRUST_MIN, THRUST_MAX] (torques small enough)
    force = rng.uniform(4.0 * (thrust_min + 0.006), 2.0 * thrust_max, (512, 1))
    torque = rng.uniform(-5e-5, 5e-5, (512, 3))
    torque[:, 2] = 0.0      # the yaw split's constant differs on cf21B_500 (test_yaw_force_per_torque_is_firmware)
    np.testing.assert_allclose(pd(force, torque), ft(force, torque), rtol=1e-6)


@pytest.mark.unit
def test_ceiling_is_battery_compensated():
    """cf21B_500 at 3.7 V: a motor gets its request beyond THRUST_MAX = 0.2 N up to the thrust at full battery
    voltage, vmotor2thrust(3.7) = 0.273 N (the legacy mixer does not clip before compensation); past it, all four
    motors are reduced together and the top one sits at the ceiling."""
    p = load_params(power_distribution, "cf21B_500")
    c = np.asarray(p["vmotor2thrust"], dtype=np.float64)
    ceiling = c[0] + c[1] * 3.7 + c[2] * 3.7**2 + c[3] * 3.7**3
    assert 0.27 < ceiling < 0.28
    pd = parametrize(power_distribution, "cf21B_500")
    L = float(p["L"])
    for want in (0.25, 0.4):
        # equal roll and pitch torques move motor 2 (mixing column 2: +roll, +pitch) by +d and motor 0 by -d over the
        # 0.12 N base: each splits as tau / L across the four columns, then / 4
        d = want - 0.12
        rpm = pd(np.array([0.48]), np.array([2 * d * L, 2 * d * L, 0.0]))
        req = np.array([0.12 - d, 0.12, 0.12 + d, 0.12])
        f = firmware_motor(req, 3.7, float(p["pwm_idle"]))
        np.testing.assert_allclose(rpm, [_rpm(x, p["rpm2thrust"]) for x in f], rtol=1e-5)
        assert np.isclose(f[2], min(want, ceiling)) and (want < ceiling) == np.isclose(f[1], 0.12)


@pytest.mark.unit
def test_floor_idles():
    """cf21B_500: a motor asked for less than THRUST_MIN idles at powerDist.idleThrust = 7000 counts of the
    supply (~3.2 mN at 3.7 V), whatever the request, instead of being raised to THRUST_MIN."""
    pd = parametrize(power_distribution, "cf21B_500")
    L = float(P["L"])
    idle_rpm = _rpm(vmotor2thrust_poly(7000.0 / 65535.0 * 3.7, C), P["rpm2thrust"])
    for want in (0.015, 0.0, -0.05):
        # equal roll and pitch torques lower motor 0 (mixing column 0: -roll, -pitch) by d under the 0.1 N base
        d = 0.1 - want
        rpm = pd(np.array([0.4]), np.array([2 * d * L, 2 * d * L, 0.0]))
        np.testing.assert_allclose(rpm[0], idle_rpm, rtol=1e-5)
    rpm = pd(np.array([0.4]), np.array([2 * 0.07 * L, 2 * 0.07 * L, 0.0]))     # 0.03 N: delivered
    np.testing.assert_allclose(rpm[0], _rpm(0.03, P["rpm2thrust"]), rtol=1e-5)


@pytest.mark.unit
def test_floor_stops_a_motor_without_idle():
    """The brushed CF2 has no idle (no arming): a motor asked for less than THRUST_MIN stops -- 0 rpm, not the
    thrust curve's other zero nor the voltage cubic's negative value at 0 V."""
    p = load_params(power_distribution, "cf2x_L250")
    assert float(p["pwm_idle"]) == 0.0
    pd = parametrize(power_distribution, "cf2x_L250")
    L = float(p["L"])
    rpm = pd(np.array([0.3]), np.array([2 * 0.07 * L, 2 * 0.07 * L, 0.0]))     # motor 0: 0.075 - 0.07 < THRUST_MIN
    assert float(rpm[0]) == 0.0 and np.all(rpm[1:] > 0)


@pytest.mark.unit
def test_zero_thrust_idles():
    """Zero collective (the rate stage then zeroes the torque) idles every motor, as the armed firmware does,
    instead of the shared mixer's motor force 0."""
    pd = parametrize(power_distribution, "cf21B_500")
    rpm = pd(np.zeros(1), np.zeros(3))
    idle = vmotor2thrust_poly(7000.0 / 65535.0 * 3.7, C)
    np.testing.assert_allclose(rpm, _rpm(idle, P["rpm2thrust"]), rtol=1e-6)


@pytest.mark.unit
def test_sim_zero_thrust_idles():
    """Through the sim: a zero-thrust body-rate command holds the rotors at idle (~3.1k rpm on the cf21B_500)."""
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.body_rate, drone="cf21B_500", freq=500)
    sim.reset()
    sim.body_rate_control(np.zeros((1, 1, 4), dtype=np.float32))
    sim.step(1)
    assert float(jnp.min(sim.data.controls.rotor_vel)) > 3000.0


@pytest.mark.unit
def test_yaw_force_per_torque_is_firmware():
    """A yaw torque splits into +-tau / (4 THRUST2TORQUE_fw) per motor: the constant the yaw gains were converted
    with (0.00569), so the firmware's per-motor yaw force is reproduced."""
    tau = 1e-3
    pd = parametrize(power_distribution, "cf21B_500")
    rpm = pd(np.array([0.4]), np.array([0.0, 0.0, tau]))
    k = np.asarray(P["rpm2thrust"])
    f = k[0] + k[1] * rpm + k[2] * rpm**2
    np.testing.assert_allclose(np.abs(f - 0.1), tau / (4 * 0.00569278844371417), rtol=1e-5)
