# Control Modes

Crazyflow provides five levels of control abstraction, from high-level position setpoints down to direct motor commands. Each level is a separate control mode selected at construction time.

## Control hierarchy

Commands flow down a hierarchy. A state command is converted to an attitude command by the Mellinger controller; an attitude command is converted to force/torque by the geometric controller; force/torque is converted to rotor velocities by the mixer.

```
State (13D)
  └─ Mellinger controller
       └─ Attitude (4D: roll, pitch, yaw, thrust)
            └─ Geometric controller
                 └─ Force/torque (4D: Fc, Tx, Ty, Tz)
                      └─ Mixer
                           └─ Rotor velocities (4D: ω₁…ω₄)

Body rate (4D: ωx, ωy, ωz, thrust)
  └─ CTBR rate controller
       └─ Force/torque (4D: Fc, Tx, Ty, Tz)  ← rejoins the chain above
```

When you select `Control.state`, the full chain runs on every control tick. When you select `Control.attitude`, only the lower two stages run.

`Control.body_rate` is a second entry point rather than a rung on the same ladder. It is a sibling of the attitude stage: both consume a setpoint and produce force/torque, so they share the mixer below them, but a rate setpoint never passes through the attitude controller.

## State control

```python
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(control=Control.state, state_freq=100, attitude_freq=500)
sim.reset()
```

Command shape: `(n_worlds, n_drones, 13)`

| Index | Variable | Units |
|---|---|---|
| 0–2 | Target position \(x, y, z\) | m |
| 3–5 | Target velocity \(\dot{x}, \dot{y}, \dot{z}\) | m/s |
| 6–8 | Target acceleration \(\ddot{x}, \ddot{y}, \ddot{z}\) | m/s² |
| 9 | Yaw | rad |
| 10 | Roll rate | rad/s |
| 11 | Pitch rate | rad/s |
| 12 | Yaw rate | rad/s |

Set unused elements to zero. A common hover command sets only the z position:

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(control=Control.state)
sim.reset()

cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 1.0  # hover at 1 m

sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

## Attitude control

```python
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.attitude, dynamics=Dynamics.so_rpy, attitude_freq=500)
sim.reset()
```

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Roll setpoint | rad |
| 1 | Pitch setpoint | rad |
| 2 | Yaw setpoint | rad |
| 3 | Collective thrust | N |

For a hover command, set thrust to `mass × g`:

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.attitude, dynamics=Dynamics.so_rpy)
sim.reset()

mass = float(sim.data.params.mass[0, 0, 0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = mass * 9.81

sim.attitude_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

## Body-rate control

Collective thrust and body rates (CTBR), the usual action interface for reinforcement learning: a rate setpoint is much easier to learn than a torque and transfers far better than a motor command. Requires `Dynamics.first_principles`.

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Body roll rate \(\omega_x\) | rad/s |
| 1 | Body pitch rate \(\omega_y\) | rad/s |
| 2 | Body yaw rate \(\omega_z\) | rad/s |
| 3 | Collective thrust | N |

Unlike the Mellinger stages, this controller is not a reimplementation of the onboard firmware. It is a PID loop on the body rates that works entirely in SI units, with the derivative taken on the measurement rather than on the error so that stepping the setpoint produces no derivative kick. Gains are angular accelerations per unit rate error, and the commanded torque is `J @ (kp e + ki ∫e - kd dω/dt)`. Multiplying by the inertia last is what keeps the gains roughly platform independent.

The torque is clipped to what the mixer can actually deliver, and the integrator freezes while an axis is saturated, so a stalled drone does not wind up a charge it has to pay back later.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(control=Control.body_rate, body_rate_freq=500)
sim.reset()

mass = float(sim.data.params.mass[0, 0, 0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = mass * 9.81
cmd[0, 0, 0] = 1.0  # Roll right at 1 rad/s

sim.body_rate_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

!!! warning "Spin the rotors up before commanding a rate"

    The simulation resets with `rotor_vel` at zero. A rate step applied straight after `reset` measures the rotors accelerating from standstill, not the rate loop, and takes several times longer to settle than it should. Hold zero rates at hover thrust for about a second first.

## Force-torque control

Direct force and torque input. Requires `Dynamics.first_principles`.

Command shape: `(n_worlds, n_drones, 4)`

| Index | Variable | Units |
|---|---|---|
| 0 | Collective force \(F_c\) | N |
| 1 | Body-frame torque \(\tau_x\) | Nm |
| 2 | Body-frame torque \(\tau_y\) | Nm |
| 3 | Body-frame torque \(\tau_z\) | Nm |

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.force_torque, dynamics=Dynamics.first_principles)
sim.reset()

mass = float(sim.data.params.mass[0, 0, 0])
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 0] = mass * 9.81

sim.force_torque_control(cmd)
sim.step(1)
```

## Rotor velocity control

Direct motor commands. Requires `Dynamics.first_principles`.

Command shape: `(n_worlds, n_drones, 4)`

| Index | Motor | Units |
|---|---|---|
| 0–3 | Motors 0–3 angular velocity | RPM |

The hover RPM for `cf2x_L250` is approximately 15 000 RPM, but the exact value depends on drone mass.

```python
import numpy as np
from crazyflow.sim import Sim, Dynamics
from crazyflow.control import Control

sim = Sim(control=Control.rotor_vel, dynamics=Dynamics.first_principles)
sim.reset()

cmd = np.full((1, 1, 4), 15_000.0, dtype=np.float32)

sim.rotor_vel_control(cmd)
sim.step(1)
```

## Control frequency

Each control mode has its own update rate. The dynamics tick (`freq`) is always the fastest.

| Mode | Rate argument | Default |
|---|---|---|
| `state` | `state_freq` | 100 Hz |
| `attitude` | `attitude_freq` | 500 Hz |
| `body_rate` | `body_rate_freq` | 500 Hz |
| `force_torque` | `force_torque_freq` | 500 Hz |
| `rotor_vel` | — | every dynamics step |

The simulator applies a new command only when the control tick fires. Between ticks, the previous command is held. The number of dynamics steps per control tick is `freq // control_freq`.

## Using the controllers standalone

The control modes above are how the simulator drives the onboard controllers. Those controllers also live in `crazyflow.control` as a self-contained library of pure functions, usable on their own for control design, learning-based policies, or as a reference implementation, independent of `Sim`. The following guides cover that standalone API:

- [Controllers](controllers.md): the controller interface and the Mellinger pipeline
- [Mellinger controller](mellinger.md): the three stages, their inputs and outputs
- [Parametrization](parametrize.md): binding a controller to a drone configuration
- [Integral errors](integral-errors.md): carrying controller state across calls
- [Batching](batching.md): evaluating many drones at once
- [JIT compilation](jit.md): compiling controllers with `jax.jit`

## Next steps

- [Functional API](../functional-api.md): running control inside JIT with `F.controllable`
- [Dynamics](../dynamics/index.md): compatibility between dynamics and control modes
