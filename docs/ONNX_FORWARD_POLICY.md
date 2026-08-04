# ONNX forward-policy compatibility

ONNX forward-policy compatibility keeps the pretrained ONNX policy path. It does not use open-loop gait or scripted joint motion.

The goal is to make Soridormi's runtime loop closer to Open Duck's original MuJoCo inference path so that a nonzero forward command can produce a real forward walking attempt.

## What changed

- `GaitPhaseGenerator` now supports Open Duck-style step phase mode.
- `OnnxPolicyController` can bootstrap the policy default actuator pose from the first MuJoCo `RobotState`.
- The first-walk script now calls `run_forward_policy_experiment.sh`.
- The forward experiment uses ONNX policy mode only.

## Why this matters

Open Duck's original inference loop uses a MuJoCo home keyframe control vector as `default_actuator`, then builds the policy observation as:

```text
gyro
accelerometer + [1.3, 0, 0]
command
joint_angles - default_actuator
joint_velocities * 0.05
last action history
motor targets
feet contacts
imitation phase
```

If Soridormi's YAML `default_pose` differs from that home control vector, the policy can move in place or fall without producing useful forward locomotion. Bootstrapping from the first MuJoCo state is a practical compatibility step while the generated policy config is being finalized.

## Run

Terminal 1:

```bash
./scripts/run_first_walk_server.sh
```

Terminal 2:

```bash
./scripts/run_forward_policy_experiment.sh
```

Default command:

```text
SORIDORMI_COMMAND_X=0.06
SORIDORMI_ACTION_SCALE=0.25
SORIDORMI_MAX_MOTOR_VELOCITY=5.24
SORIDORMI_PHASE_MODE=step
SORIDORMI_PHASE_PERIOD_STEPS=50
SORIDORMI_BOOTSTRAP_POLICY_DEFAULTS_FROM_STATE=1
SORIDORMI_COMMAND_RAMP_SECONDS=1.0
```

## Tuning order

Try this order, one variable at a time:

```bash
SORIDORMI_COMMAND_X=0.03 ./scripts/run_forward_policy_experiment.sh
SORIDORMI_COMMAND_X=0.06 ./scripts/run_forward_policy_experiment.sh
SORIDORMI_COMMAND_X=0.10 ./scripts/run_forward_policy_experiment.sh
SORIDORMI_COMMAND_X=0.15 ./scripts/run_forward_policy_experiment.sh
```

Then try phase periods:

```bash
SORIDORMI_PHASE_PERIOD_STEPS=40 ./scripts/run_forward_policy_experiment.sh
SORIDORMI_PHASE_PERIOD_STEPS=50 ./scripts/run_forward_policy_experiment.sh
SORIDORMI_PHASE_PERIOD_STEPS=60 ./scripts/run_forward_policy_experiment.sh
```

## Expected status

This is the first ONNX-forward integration pass. It should produce a stronger forward walking attempt than the previous neutral/slow profile. It may still fall. If it falls, inspect:

```bash
./scripts/inspect_latest_log.sh
./scripts/analyze_latest_policy_log.sh
```

The policy debug log should show:

```text
policy_defaults_bootstrapped: true
bootstrapped_default_count: 14
command: [nonzero x, ...]
phase: changing each control step
feet_contacts: nonzero when feet touch floor
```
