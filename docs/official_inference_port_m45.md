# M4.5 Official Inference Port

M4.4 proved that the official Open Duck Mini v2 runner can move forward in the same Docker/MuJoCo environment. M4.5 ports the most important startup/default-actuator behavior into Soridormi's normal API-based policy runtime.

## What changed

- MuJoCo simulator can start/reset from `model.keyframe("home").qpos`.
- MuJoCo simulator can keep `model.keyframe("home").ctrl` as the initial actuator control vector.
- When `SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1`, configured `reset_pose` is skipped so the official home keyframe remains authoritative.
- `RobotState` can carry optional `actuator_ctrl`, filled by the MuJoCo backend from `data.ctrl`.
- ONNX runtime bootstrap now prefers `state.actuator_ctrl` over joint qpos for policy defaults and motor-target history.
- Policy profiles can describe simulator-side requirements such as `use_home_keyframe: true`.
- `scripts/run_official_compatible_policy_server.sh` starts the sim server with the profile-compatible home-keyframe settings.

## Why this matters

The official Open Duck runner uses:

```text
data.qpos[:] = model.keyframe("home").qpos
data.ctrl[:] = model.keyframe("home").ctrl
default_actuator = model.keyframe("home").ctrl
```

Before M4.5, Soridormi could start from a different MuJoCo initial qpos and guess the policy default actuator from the first observed joint qpos. That is enough to make the body wiggle, but it can prevent useful stepping.

## Recommended run

Terminal 1:

```bash
./scripts/run_official_compatible_policy_server.sh open_duck_forward
```

Terminal 2:

```bash
./scripts/run_policy_experiment.sh open_duck_forward
```

Then inspect:

```bash
./scripts/inspect_latest_log.sh
./scripts/analyze_latest_policy_log.sh
```

The debug payload should report:

```text
bootstrap_source: actuator_ctrl
```

If it reports `joint_positions`, the sim backend is not exposing `actuator_ctrl` or the server was not started with the M4.5 code.

## Relationship to M4.4

M4.4 remains the reference baseline. M4.5 does not replace the official runner; it makes Soridormi's runtime path closer to it while keeping the runtime/backend API separation needed for sim-to-real.
