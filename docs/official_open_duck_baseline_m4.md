# M4.4 Official Open Duck Baseline Reproduction

M4.4 changes the debugging strategy: before tuning Soridormi's ONNX wrapper any further, run the official Open Duck Mini v2 MuJoCo inference path inside the same Docker stack.

The purpose is to answer one question:

```text
Can the official BEST_WALK_ONNX_2.onnx policy produce forward locomotion in our container, MuJoCo XML, and GPU/CPU runtime environment?
```

If the official baseline walks or clearly steps, then Soridormi still has an observation/action/default-pose/contact mismatch. If the official baseline also only wiggles, then the issue is likely outside Soridormi's wrapper: model asset version, Open Duck submodule content, XML, dependency/runtime behavior, or command usage.

## Why this matters

The official Open Duck path is not just ONNX inference. It also depends on exact details:

- `scene_flat_terrain.xml`
- `model.keyframe("home").qpos`
- `model.keyframe("home").ctrl` as `default_actuator`
- `sim_dt = 0.002`
- `decimation = 10`, so policy runs at 50 Hz
- polynomial reference-motion period for imitation phase
- foot contact bodies `foot_assembly` and `foot_assembly_2` against `floor`
- accelerometer x-bias of `+1.3`
- `action_scale = 0.25`
- `max_motor_velocity = 5.24`

M4.4 runs the official `playground.open_duck_mini_v2.mujoco_infer.MjInfer` class and only wraps it with a fixed non-interactive command and finite runtime.

## One-time setup

Make sure the Open Duck submodules and policy file exist:

```bash
git submodule update --init --recursive workspace/Open_Duck_Playground workspace/Open_Duck_Mini
```

This update adds `onnxruntime` and `etils` to the simulator optional dependencies. Rebuild the simulator image once if the official baseline reports missing imports:

```bash
./scripts/build_sim.sh
```

## Run official forward baseline

```bash
./scripts/run_official_forward_baseline.sh
```

Defaults:

```text
SORIDORMI_OFFICIAL_COMMAND_X=0.15
SORIDORMI_OFFICIAL_COMMAND_Y=0.0
SORIDORMI_OFFICIAL_COMMAND_YAW=0.0
SORIDORMI_OFFICIAL_MAX_SECONDS=20
SORIDORMI_OFFICIAL_VIEWER=1
```

Run headless:

```bash
SORIDORMI_OFFICIAL_VIEWER=0 ./scripts/run_official_forward_baseline.sh
```

Try a negative command to check sign/frame convention:

```bash
SORIDORMI_OFFICIAL_COMMAND_X=-0.15 ./scripts/run_official_forward_baseline.sh
```


## Docker/X11 exit segfault workaround

Some Docker + X11 + MuJoCo viewer combinations can segfault during Python/C-extension teardown after the official baseline has already finished, printed the summary, and written `latest_official_baseline.json`.

The runner defaults to a fast successful exit after the summary is flushed:

```bash
SORIDORMI_OFFICIAL_FAST_EXIT=1 ./scripts/run_official_forward_baseline.sh
```

To debug normal interpreter shutdown, disable it:

```bash
SORIDORMI_OFFICIAL_FAST_EXIT=0 ./scripts/run_official_forward_baseline.sh
```

A successful baseline is determined by the JSON summary, not by viewer teardown.

## Run official keyboard baseline

```bash
./scripts/run_official_keyboard_baseline.sh
```

In the MuJoCo viewer:

```text
arrow up     forward
arrow down   backward
left/right   lateral
A/E          yaw
```

## Inspect official summary

The wrapper writes:

```text
data/official_baseline/latest_official_baseline.json
```

Show it:

```bash
./scripts/show_official_baseline_summary.sh
```

Important fields:

```text
base_displacement_xyz
policy_steps
action_stats
motor_target_stats
contact_stats
```

## Interpret result

### Official baseline steps / moves forward

Then Soridormi must be changed to match the official path more exactly. Focus on:

```text
default_actuator from home ctrl
phase period from reference motion
contact logic
joint/action order
motor target initialization
accelerometer bias
```

### Official baseline also wiggles with feet planted

Then do not keep tuning Soridormi. First verify:

```text
correct policy file
correct Open_Duck_Playground version
correct XML/assets
whether keyboard/fixed command is actually setting nonzero command
whether official contact sensors/body names behave in this model
```

## Why M4.4 uses a tiny wrapper

The official script is interactive and infinite. M4.4 reuses the official class but adds:

- fixed command from env/CLI
- finite `max_seconds`
- JSON summary
- a lightweight stub for `playground.common.utils.LowPassActionFilter`, so the simulator image does not need JAX just to import an unused filter path

It does not rewrite the official observation/action loop in Soridormi.


## JAX-free official baseline import shim

The official Open Duck `mujoco_infer.py` imports `mujoco_infer_base.py`, which imports
`playground.open_duck_mini_v2.base`. That official `base.py` is mostly for the MJX/JAX
training environment and imports JAX-related packages. Soridormi's M4.4 baseline runner
is only trying to reproduce the pure MuJoCo + ONNX inference loop, so it installs a
small in-process shim for `playground.open_duck_mini_v2.base.get_assets()` before
importing the official inference class. This avoids adding JAX to the sim image just to
run the official baseline.

If you intentionally want the full official training environment later, add the official
Open Duck training dependencies in the M6 training image instead of the lightweight sim
baseline image.

## JAX-free asset loader note

The official `base.py` imports JAX/MJX training dependencies, but the M4.4
baseline only needs `base.get_assets()` to load MuJoCo XML and mesh assets.
Soridormi installs a lightweight `get_assets()` stub before importing the
official inference module. The stub stores one in-memory asset per basename
only, because MuJoCo rejects assets dictionaries containing duplicate basenames
such as both `head.stl` and `assets/head.stl`.

