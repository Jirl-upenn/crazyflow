"""Fly a roll-rate oscillation with the CTBR body-rate controller."""

import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim


def main():
    sim = Sim(control=Control.body_rate, body_rate_freq=500)
    sim.reset()
    duration = 5.0
    fps = 60

    mass, gravity = sim.data.params.mass, -sim.data.params.gravity_vec[-1]
    # [roll_rate, pitch_rate, yaw_rate, collective thrust] in [rad/s, rad/s, rad/s, N]
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
    cmd[..., 3] = (mass + 1e-4) * gravity  # Plus a small margin to accelerate slightly

    # The rotors start at rest, so hold zero rates while they spin up to hover. Commanding a rate
    # before that only measures the rotors accelerating from standstill.
    for _ in range(sim.control_freq):
        sim.body_rate_control(cmd)
        sim.step(sim.freq // sim.control_freq)

    for i in range(int(duration * sim.control_freq)):
        cmd[..., 0] = 2.0 * np.sin(2 * np.pi * i / sim.control_freq)  # 1 Hz roll-rate sweep
        sim.body_rate_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
