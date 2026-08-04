# MuJoCo Auto Reset Debug Mode

simulator fall detection and reset adds an optional simulation safety loop that resets the MuJoCo robot after a fall.

This is a development/debug tool. It is useful while tuning poses and policies because the robot can fall many times without requiring the simulator server to be restarted manually.

## Enable auto reset

Start the simulator:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Then start a runtime mode, for example:

```bash
./scripts/run_stand_runtime.sh
```

or:

```bash
SORIDORMI_RUNTIME_MODE=stand \
SORIDORMI_STAND_RAMP_SECONDS=5.0 \
./scripts/run_runtime_loop.sh
```

## Helper script

```bash
./scripts/run_auto_reset_stand_server.sh
```

In another terminal:

```bash
./scripts/run_stand_runtime.sh
```

## Config

Add or merge this into `configs/robots/open_duck_mini_v2.yaml`:

```yaml
safety:
  auto_reset:
    enabled_env: SORIDORMI_AUTO_RESET
    min_base_height: 0.05
    max_tilt_rad: 1.2
    cooldown_seconds: 1.0
```

## Trigger conditions

The simulator resets when either condition is true:

- base height is below `min_base_height`
- absolute roll or pitch exceeds `max_tilt_rad`

Yaw is ignored because yaw rotation alone does not mean the robot has fallen.

## Expected log

```text
MuJoCo auto-reset enabled via SORIDORMI_AUTO_RESET=1 (...)
Auto reset triggered: height=..., tilt=... rad, reset_count=...
```

## Notes

Auto reset is skipped while fixed-base mode is enabled because fixed-base mode already prevents falls and is used for pose inspection.

A reset keeps the MuJoCo model and viewer open. It resets `MjData`, reapplies `reset_pose`, initializes actuator controls from the current qpos, and continues serving the runtime API.
