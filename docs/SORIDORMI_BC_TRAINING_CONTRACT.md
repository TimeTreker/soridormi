# Soridormi BC training contract

behavior-cloning training contract adds a versioned behavior-cloning contract before changing the learner
architecture. The goal is to keep the low-level Soridormi policy structured and
bounded:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

The contract lives at:

```text
configs/training/open_duck_mini_v2_context_bc_contract_v1.json
```

This patch does **not** train a new policy and does **not** change runtime
control. It defines the dataset/model interface that later BC and residual/RL
work must satisfy.

## Why this exists

Earlier datasets can still use `soridormi.policy_supervision.v1`, where the
learner consumes an observation vector and 14D action. That is useful for the command-conditioned baseline
walking BC, but it hides important context in scripts and filenames. New
context-conditioned rows should use:

```json
{
  "sample_type": "soridormi.policy_supervision.context_v1",
  "scenario_id": "flat_walk_varied_speed_v1",
  "skill_id": "walk_velocity",
  "robot_state": {"observation": [101]},
  "desired_command": {"vx_mps": 0.12, "vy_mps": 0.0, "yaw_radps": 0.05},
  "task_context": {"skill_id": "walk_velocity", "gait_style": "default_walk"},
  "environment_context": {"terrain_type": "flat"},
  "teacher_action": [14]
}
```

Natural language is not a policy input. Chromie is the robot brain and may
choose a skill/context from user intent, but Soridormi is the robot cerebellum:
it validates bounded structured fields before training or execution and owns the
low-level `action_14d` body policy.

## Validate the contract

```bash
./scripts/validate_bc_training_contract.sh \
  --json | python -m json.tool
```

Write a Markdown report:

```bash
./scripts/validate_bc_training_contract.sh \
  --output artifacts/training_contract/contract_report.md
```

Validate a context JSONL dataset:

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/context_walk.jsonl \
  --output artifacts/training_contract/context_walk_report.md \
  --json | python -m json.tool
```

Legacy command-conditioned validation is explicit:

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/legacy_walk.jsonl \
  --allow-legacy
```

`--allow-legacy` is only a compatibility bridge for the command-conditioned input. Richer context work should
require context rows with `task_context`, `environment_context`, and scenario
metadata.

## Adoption stages

1. `velocity_commanded_bc`: robot state plus continuous `vx/vy/yaw` command.
2. `skill_task_context_bc`: add `skill_id`, `gait_style`, and priority labels.
3. `terrain_context_bc`: add terrain/friction/obstacle context after scenario
   eval and dataset gates pass.
4. `history_conditioned_bc`: add bounded short history for command transitions,
   contacts, and residual/RL compatibility.

Before a stage is used for training, run:

```bash
./scripts/gate_dataset_scenario_coverage.sh ...
./scripts/report_dataset_coverage.sh ...
./scripts/validate_bc_training_contract.sh --sample-jsonl ...
```

This keeps BC data variation and scenario coverage central before policy
architecture changes.

## Command-conditioned offline training mode

The linear and neural BC trainers keep the legacy 101D observation input by
default. For command-conditioned context experiments, use the explicit input
mode:

```bash
./scripts/train_behavior_clone.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --input-mode context_command_v1
```

This trains on a 104D feature vector:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
```

The default normalization artifact for this mode is
`normalization.context_command_v1.json`, so it does not overwrite the
legacy 101D `normalization.json`.

For neural smoke runs without runtime artifacts:

```bash
./scripts/train_neural_behavior_clone.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --input-mode context_command_v1 \
  --skip-onnx \
  --no-profile
```

## Runtime command-conditioned context input

clearance qualification adds the matching runtime/profile input mode:

```text
context_command_v1
```

This runtime mode appends the first three policy command values to the 101D
robot observation:

```text
policy_input[104] = robot_state.observation[101] + vx_mps + vy_mps + yaw_radps
```

A runnable context policy profile must declare both:

```yaml
contract:
  input_mode: context_command_v1
  policy_input_size: 104
model:
  input_shape: [1, 104]
  input_mode: context_command_v1
```

After clearance qualification profile plumbing, neural context-mode training can export an ONNX
model and generate this profile metadata:

```bash
./scripts/train_neural_behavior_clone.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --input-mode context_command_v1 \
  --profile-name context_command_candidate \
  --force-profile
```

A context-trained ONNX policy still needs profile/model validation and MuJoCo
rollout evidence before promotion.
