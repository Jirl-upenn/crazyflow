"""Unit tests for the integration schemes."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator, euler, symplectic_euler
from crazyflow.sim.sim import select_dynamics_fn


def _moving_data(sim: Sim):
    """sim.data with a nonzero, non-hover state, so every integrated quantity has a derivative."""
    states = sim.data.states
    n = states.pos.shape[:-1]
    vel = jnp.broadcast_to(jnp.array([1.0, -0.5, 0.3]), n + (3,))
    ang_vel = jnp.broadcast_to(jnp.array([2.0, -1.0, 0.5]), n + (3,))
    quat = jnp.broadcast_to(R.from_euler("xyz", jnp.array([0.2, -0.1, 0.3])).as_quat(), n + (4,))
    rotor_vel = states.rotor_vel * jnp.array([1.1, 0.9, 1.05, 0.95])
    return sim.data.replace(
        states=states.replace(vel=vel, ang_vel=ang_vel, quat=quat, rotor_vel=rotor_vel)
    )


@pytest.mark.unit
def test_symplectic_euler_step():
    """One symplectic Euler step: velocities from the accelerations, then the pose from the NEW velocities."""
    sim = Sim(n_worlds=2, dynamics=Dynamics.first_principles, integrator=Integrator.symplectic_euler)
    deriv_fn = select_dynamics_fn(Dynamics.first_principles)
    data = _moving_data(sim)
    dt = 1 / data.core.freq
    d = deriv_fn(data).states_deriv
    s = data.states
    out = symplectic_euler(data, deriv_fn).states
    vel = s.vel + d.acc * dt
    ang_vel = s.ang_vel + d.ang_acc * dt
    np.testing.assert_allclose(out.vel, vel, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out.ang_vel, ang_vel, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out.rotor_vel, s.rotor_vel + d.rotor_acc * dt, rtol=1e-6)
    np.testing.assert_allclose(out.pos, s.pos + vel * dt, rtol=1e-6, atol=1e-7)
    quat = (R.from_quat(s.quat) * R.from_rotvec(ang_vel * dt)).as_quat()
    np.testing.assert_allclose(np.abs(np.sum(out.quat * quat, -1)), 1.0, atol=1e-6)


@pytest.mark.unit
def test_symplectic_euler_vs_euler():
    """Both schemes give the same velocities; the pose differs by exactly the velocity change times dt."""
    sim = Sim(n_worlds=2, dynamics=Dynamics.first_principles)
    deriv_fn = select_dynamics_fn(Dynamics.first_principles)
    data = _moving_data(sim)
    dt = 1 / data.core.freq
    a, b = euler(data, deriv_fn).states, symplectic_euler(data, deriv_fn).states
    np.testing.assert_allclose(a.vel, b.vel, rtol=1e-6)
    np.testing.assert_allclose(a.ang_vel, b.ang_vel, rtol=1e-6)
    np.testing.assert_allclose(b.pos - a.pos, (b.vel - data.states.vel) * dt, atol=1e-7)


def _fly(integrator: Integrator) -> np.ndarray:
    """One second of state-controlled flight towards a setpoint 0.5 m away; final positions."""
    sim = Sim(n_worlds=2, dynamics=Dynamics.first_principles, control=Control.state, integrator=integrator)
    sim.reset()
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = np.asarray(sim.data.states.pos) + np.array([0.3, -0.2, 0.3])
    for _ in range(sim.control_freq):
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
    return np.asarray(sim.data.states.pos)


@pytest.mark.unit
def test_sim_steps_with_every_integrator():
    """Every integrator runs through Sim.step, and at 500 Hz they fly the same closed-loop trajectory to within 1 cm."""
    ref = _fly(Integrator.euler)
    assert np.all(np.isfinite(ref))
    for integrator in (Integrator.rk4, Integrator.symplectic_euler):
        np.testing.assert_allclose(_fly(integrator), ref, atol=0.01, err_msg=str(integrator))
