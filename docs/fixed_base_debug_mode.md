# Fixed-base debug mode

Fixed-base mode is a MuJoCo-only debug mode for inspecting joint poses while gravity remains enabled.

It is not physically realistic and should not be used for final walking or policy evaluation.

## Why it exists

Normal gravity with a free floating base makes the robot fall unless a balance controller or walking policy is active.
Zero-gravity mode is useful for checking joint directions, but it removes gravity entirely.
Fixed-base mode is the middle ground: gravity stays enabled, but the floating base is held at its reset pose after every simulation step.

This helps you inspect:

- default pose signs
- hip/knee/ankle symmetry
- feet placement relative to the body
- whether one joint moves at a time in joint sweep mode

## Run

Terminal 1:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_MUJOCO_FIXED_BASE=1 \
./scripts/run_sim_server.sh
```

Terminal 2:

```bash
SORIDORMI_RUNTIME_MODE=stand \
SORIDORMI_STAND_RAMP_SECONDS=5.0 \
./scripts/run_runtime_loop.sh
```

Or for joint-direction validation:

```bash
SORIDORMI_RUNTIME_MODE=joint_sweep \
SORIDORMI_JOINT_SWEEP_AMPLITUDE=0.15 \
SORIDORMI_JOINT_SWEEP_PERIOD_SECONDS=4.0 \
SORIDORMI_JOINT_SWEEP_HOLD_SECONDS=1.0 \
./scripts/run_runtime_loop.sh
```

## Expected result

The robot body should stay in place while the legs/head joints move. If `SORIDORMI_MUJOCO_VIEWER=1` is enabled, the MuJoCo viewer should remain stable.

## Disable

Do not set `SORIDORMI_MUJOCO_FIXED_BASE`, or set it to `0`:

```bash
SORIDORMI_MUJOCO_FIXED_BASE=0
```

## Implementation notes

The backend captures the initial/reset floating-base pose after applying `reset_pose`. After each MuJoCo substep, it restores the configured floating-base qpos slices and zeros the configured floating-base qvel slices.

The slices come from the robot config:

```yaml
base:
  qpos_xyz_slice: [0, 3]
  qpos_quat_wxyz_slice: [3, 7]
  qvel_linear_slice: [0, 3]
  qvel_angular_slice: [3, 6]
```
