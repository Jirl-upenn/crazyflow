# MuJoCo Warp Rendering

Crazyflow renders the MuJoCo scene with the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) ray tracer through `mujoco.mjx.render`. A camera sensor renders batched RGB(-D) images of every drone's camera across all worlds in a single GPU call. Unlike [gaussian splats](splats.md), it needs no scene capture: it draws the same meshes, materials, and textures as the MuJoCo viewer.

## Installation

Install crazyflow with the `warp` extra. It pulls in `warp-lang` at the version MJX was built against.

```bash
pip install "crazyflow[warp]"
```

The sensor ray traces with CUDA kernels and requires an NVIDIA GPU and a simulation created with `device="gpu"`.

## Camera sensor

`render_warp_rgb` from `crazyflow.sim.sensors.warp` renders RGB images from any model camera, batched over all worlds and drones. Cameras are resolved by name as `{camera_prefix}:{drone}`, so `fpv_cam` (the default) is each drone's first-person camera and `track_cam` its chase camera.

<!-- notest: requires the warp extra and a CUDA GPU -->
```{ .python notest }
from crazyflow.sim.sensors.warp import build_render_warp_fn, render_warp_rgb

sim = Sim(n_worlds=4, n_drones=2, device="gpu")
imgs = render_warp_rgb(sim, resolution=(320, 240))  # (4, 2, 240, 320, 3) in [0, 1]
imgs = render_warp_rgb(sim, drones=0, resolution=(320, 240))  # single drone: (4, 1, 240, 320, 3)

# Build the renderer once and call it with sim.data, e.g. inside a jitted training loop
render_fn = build_render_warp_fn(sim, resolution=(320, 240))
imgs = render_fn(sim.data)
```

Rays that hit nothing show the model's skybox. Pass `background=(r, g, b)` for a solid color instead.

Geom colors can change from call to call without rebuilding the renderer. Pass `geom_rgba` with shape `(ngeom, 4)` to recolor every world, or `(n_worlds, ngeom, 4)` to color each world differently, e.g. to highlight each world's current target. As in MuJoCo, only geoms without a material take their color from `geom_rgba`.

<!-- notest: requires the warp extra and a CUDA GPU -->
```{ .python notest }
rgba = np.broadcast_to(sim.mj_model.geom_rgba, (sim.n_worlds, sim.mj_model.ngeom, 4)).copy()
rgba[0, target_geom] = [0.0, 1.0, 0.0, 1.0]  # Only world 0 sees the target in green
imgs = render_fn(sim.data, rgba)
```

Each renderer owns a MuJoCo Warp render context sized to the simulation's model and number of worlds. Building one compiles the kernels and renders a warm-up frame, so it takes about a second. `render_warp_rgb` builds a renderer on its first call for a given set of arguments and reuses it afterwards. `build_render_warp_fn` returns the renderer directly. Rebuilding the MuJoCo model, e.g. with `Sim.build_mjx`, requires building a new renderer.

The renderer is not differentiable and cannot be `vmap`ed, since the render context has a fixed number of worlds. Use the [splat sensor](splats.md) when you need gradients through the image.

See `examples/rendering/warp_camera.py` for a matplotlib-based camera sensor demo.

## Aliasing and supersampling

The ray tracer shades each pixel from a single ray through its center and samples textures without mipmaps. Texture detail finer than a pixel, such as the distant checkerboard floor and its thin grid lines, therefore aliases, and the aliasing pattern changes from frame to frame as the camera moves, which shows up as flicker. `supersample=k` renders at `k` times the resolution and averages each `k x k` block, which removes most of it at a cost that grows with `k**2`:

<!-- notest: requires the warp extra and a CUDA GPU -->
```{ .python notest }
imgs = render_warp_rgb(sim, resolution=(320, 240), supersample=3)  # 9 rays per pixel
```

For RGB-D, depth reports the nearest of the `k x k` samples, so that pixels on an object's silhouette do not get depths in between the object and the background. `benchmark/render_compare.py` renders a flight with the OpenGL renderer and the Warp sensor side by side for a visual comparison.

## Depth sensor

`render_warp_rgbd` adds depth as a fourth channel.

<!-- notest: requires the warp extra and a CUDA GPU -->
```{ .python notest }
from crazyflow.sim.sensors.warp import build_render_warp_rgbd_fn, render_warp_rgbd

rgbd = render_warp_rgbd(sim, resolution=(320, 240), max_range=8.0)  # (4, 2, 240, 320, 4)
rgb, depth = rgbd[..., :3], rgbd[..., 3]

render_fn = build_render_warp_rgbd_fn(sim, resolution=(320, 240), max_range=8.0)
rgbd = render_fn(sim.data)
```

The values are depth along the camera's optical axis in meters, the same convention as `render_splat_rgbd`. Pixels whose ray hits nothing report `max_range`, which is also the value depth is clipped to.

## Poses

The sensor computes body, camera, and light poses with MuJoCo Warp from the drone states in `sim.data`. MJX's JAX kinematics, which `sync_sim2mjx` uses, leaves bodies attached below a drone (such as its propellers) at their initial pose and does not compute the centers of mass that `trackcom` cameras like `track_cam` follow. Cameras and lights with `targetbody` modes are not supported.

## CUDA graphs and driver versions

MuJoCo Warp captures each render into a CUDA graph. Loading a kernel module while capturing requires CUDA driver 12.3 or newer, so building a renderer first renders one frame without graph capture to load all kernels. Renders afterwards replay the graph, which is much faster than rendering without it.
