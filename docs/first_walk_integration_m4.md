# M4.0 First Walk Integration

M4.0 changes Soridormi from a connected-but-falling policy loop into a closer Open Duck Mini v2 inference compatibility loop.

## What changed

- `RobotState` can carry `feet_contacts: [left, right]`.
- MuJoCo backend reads real foot contacts using Open Duck Mini v2 body names first:
  - left: `foot_assembly`
  - right: `foot_assembly_2`
  - ground: `floor`
- MuJoCo backend reads real `gyro` and `accelerometer` sensors when they exist in the XML, with safe fallback to old values.
- Observation builder can add the Open Duck accelerometer x-bias: `[1.3, 0.0, 0.0]`.
- Observation builder uses simulator-provided foot contacts when available.
- ONNX controller resets policy/action/motor-target history when MuJoCo auto-reset rewinds robot time.
- First-walk launch scripts are added.

## First run

Terminal 1:

```bash
./scripts/run_first_walk_server.sh
```

Terminal 2:

```bash
./scripts/run_first_walk_experiment.sh
```

Conservative default:

```bash
SORIDORMI_COMMAND_X=0.01
SORIDORMI_PHASE_FREQUENCY=1.0
SORIDORMI_ACTION_SCALE=0.10
SORIDORMI_MAX_MOTOR_VELOCITY=5.24
SORIDORMI_POLICY_ACCEL_BIAS_X=1.3
SORIDORMI_USE_STATE_FEET_CONTACTS=1
```

If motion is not violent, try the original Open Duck action scale:

```bash
SORIDORMI_ACTION_SCALE=0.25 ./scripts/run_first_walk_experiment.sh
```

## Export model-derived compatibility defaults

Run this in the sim environment:

```bash
./scripts/export_policy_compat_config.sh
```

It writes:

```text
./data/open_duck_mini_v2.policy_compat_generated.yaml
```

Inspect it before merging into `configs/robots/open_duck_mini_v2.yaml`. The most important part is `default_pose.positions`, which is exported from the MuJoCo `home` keyframe ctrl if the model has it.

## Why this matters

Open Duck's original MuJoCo inference uses real MuJoCo sensors, real foot contacts, `default_actuator + action * 0.25`, motor speed limiting, and an accelerometer x-bias. If any of those are missing or mismatched, a pretrained walking policy may move but fall quickly.

## Pass condition

M4.0 passes if:

- first-walk scripts start the sim/runtime pair;
- logs show nonzero command and changing phase;
- `/soridormi/policy_debug` includes `feet_contacts`;
- after auto-reset, policy/action/motor-target history resets instead of continuing from the fallen cycle;
- the robot attempts repeated stepping using real contact observations.

Stable walking is the target, but not guaranteed by this patch alone. If it still falls, compare exported `default_pose.positions` against your active YAML first.
