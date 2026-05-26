# M4.2 Runnable policy engine

M4.2 turns the ONNX runtime from a set of scripts into a policy-engine entrypoint.
The goal is engineering stability before model training: the simulator, runtime,
policy contract, logging, and model replacement path should stay stable while the
ONNX model changes.

## One-command policy run

Start MuJoCo in one terminal:

```bash
./scripts/run_first_walk_server.sh
```

Run the default ONNX policy profile in another terminal:

```bash
./scripts/run_policy_experiment.sh open_duck_forward
```

A safer profile is also included:

```bash
./scripts/run_policy_experiment.sh open_duck_crawl_safe
```

The older forward entrypoint now routes through the policy profile system:

```bash
./scripts/run_forward_policy_experiment.sh
```

## Policy profiles

Profiles live in:

```text
configs/policies/*.yaml
```

A profile externalizes the model contract and runtime parameters:

```yaml
model:
  path: /workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: [1, 101]
  output_shape: [1, 14]

command:
  x: 0.06

phase:
  mode: step
  period_steps: 50

action_mapping:
  action_scale: 0.25
  max_motor_velocity: 5.24
```

For a newly trained model, copy a profile, change `model.path`, and update the
contract if your exported ONNX names or shapes differ.

## Model compatibility check

Before running a policy, validate the ONNX model:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward
```

Or check an override model against the same profile contract:

```bash
./scripts/check_policy_model.sh /models/my_policy.onnx --profile open_duck_forward
```

The checker validates model file existence, input/output names, shapes, and dtypes.

## Displacement analysis

M4.2 adds optional `base_position_xyz` and `base_quat_wxyz` fields to `RobotState`.
The MuJoCo backend fills them from the floating base qpos. The log analyzer now
reports forward displacement:

```bash
./scripts/analyze_latest_policy_log.sh
```

This separates cases where the policy moves but falls from cases where the policy
never produces forward displacement.

## Engineering contract

The model replacement target is:

```text
same runtime
same RobotState
same MotorCommand
same observation/action contract
replace only the profile/model path
```

This supports the later M5/M6/M7 flow: model replacement, training, and transfer
to real hardware.
