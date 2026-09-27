"""Sensors for the simulation.

:mod:`crazyflow.sim.sensors.depth` renders depth images with MuJoCo raycasting and is always
available. :mod:`crazyflow.sim.sensors.splat` renders photorealistic RGB(-D) images from gaussian
splats and requires the optional ``splats`` extra. :mod:`crazyflow.sim.sensors.warp` ray traces
RGB(-D) images of the MuJoCo scene with MuJoCo Warp and requires the optional ``warp`` extra.
"""
