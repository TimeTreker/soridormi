# Reset and default-pose tuning

This step makes MuJoCo startup repeatable by moving reset/default pose data into the robot config.

## Export the current MuJoCo XML pose

Run inside the sim container:

```bash
sim
python -m soridormi_sim.export_pose_from_model | tee /data/open_duck_mini_v2_pose_snippet.yaml
```

Then inspect it on the host:

```bash
cat data/open_duck_mini_v2_pose_snippet.yaml
```

Copy the generated `reset_pose` and `default_pose` sections into:

```text
configs/robots/open_duck_mini_v2.yaml
```

For the first baseline, keep `default_pose.positions` equal to `reset_pose.joints`.

## Test reset pose only

Terminal 1:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
./scripts/run_sim_server.sh
```

Do not start the runtime yet. The robot should appear in a repeatable pose. A slow fall is acceptable at this stage; a violent snap or instant flip is not.

## Test standing mode with the same pose

Terminal 1:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
./scripts/run_sim_server.sh
```

Terminal 2:

```bash
SORIDORMI_RUNTIME_MODE=stand \
SORIDORMI_STAND_RAMP_SECONDS=5.0 \
./scripts/run_runtime_loop.sh
```

If `reset_pose == default_pose`, the runtime should not command a big motion. The robot may still slowly fall because there is no balance controller yet.

## Tune carefully

Only change small values at a time:

```text
0.05 rad
0.10 rad
0.15 rad
```

Suggested log table:

| Run | Change | Result | Notes |
|---|---|---|---|
| 001 | reset_pose == default_pose | | |
| 002 | knees ±0.05 rad | | |
| 003 | knees ±0.10 rad | | |
| 004 | hips ±0.05 rad | | |
| 005 | ankles ±0.05 rad | | |

## Success criteria

- `reset_pose` exists in the robot config.
- `MujocoBackend` applies `reset_pose` before the first `mj_forward()`.
- `data.ctrl` is initialized from the configured qpos.
- `default_pose` exists and is used by `SORIDORMI_RUNTIME_MODE=stand`.
- `reset_pose == default_pose` causes no violent snap.
