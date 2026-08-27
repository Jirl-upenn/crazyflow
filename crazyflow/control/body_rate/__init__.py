"""Collective-thrust body-rate (CTBR) controller.

A PID controller on the body rates, the usual action interface for reinforcement learning on
quadrotors. See [control][crazyflow.control.body_rate.control] for details.
"""

from crazyflow.control.body_rate.control import (
    BodyRateData,
    body_rate2force_torque,
    control_body_rate2force_torque,
)

__all__ = ["body_rate2force_torque", "BodyRateData", "control_body_rate2force_torque"]
