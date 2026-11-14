"""The body-rate controller's power distribution (crazyflow.control.body_rate.power_distribution)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control, parametrize
from crazyflow.control.body_rate import control_power_distribution, power_distribution
from crazyflow.control.core import load_params
from crazyflow.drones import available_drones
from crazyflow.dynamics import Dynamics
from crazyflow.sim.sim import build_control_fns


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
def test_matches_shared_mixer_below_thrust_max(drone: str):
    """Where no motor asks for more than THRUST_MAX the shared mixer and the power distribution agree."""
    from crazyflow.control.mellinger import force_torque2rotor_vel

    pd = parametrize(power_distribution, drone)
    ft = parametrize(force_torque2rotor_vel, drone)
    thrust_max = float(load_params(force_torque2rotor_vel, drone)["thrust_max"])
    rng = np.random.default_rng(0)
    force = rng.uniform(0.0, 2.0 * thrust_max, (512, 1))
    force[:16] = 0.0
    torque = rng.uniform(-1e-3, 1e-3, (512, 3))
    np.testing.assert_allclose(pd(force, torque), ft(force, torque), rtol=1e-6)


@pytest.mark.unit
def test_ceiling_is_battery_compensated():
    """cf21B_500 at 3.7 V: a single motor gets its request beyond THRUST_MAX = 0.2 N up to the thrust at full
    battery voltage, vmotor2thrust(3.7) = 0.273 N (the legacy mixer does not clip before compensation)."""
    p = load_params(power_distribution, "cf21B_500")
    c = np.asarray(p["vmotor2thrust"], dtype=np.float64)
    ceiling = c[0] + c[1] * 3.7 + c[2] * 3.7**2 + c[3] * 3.7**3
    assert 0.27 < ceiling < 0.28
    pd = parametrize(power_distribution, "cf21B_500")
    L = float(p["L"])
    for want in (0.25, 0.4):
        # equal roll and pitch torques raise motor 2 (mixing column 2: +roll, +pitch) by d over the 0.12 N base:
        # each splits as tau / L across the four columns, then / 4
        d = want - 0.12
        rpm = pd(np.array([0.48]), np.array([2 * d * L, 2 * d * L, 0.0]))
        np.testing.assert_allclose(rpm[2], _rpm(min(want, ceiling), p["rpm2thrust"]), rtol=1e-5)
