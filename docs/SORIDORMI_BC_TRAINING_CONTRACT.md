# Soridormi BC training contract

M9E adds a versioned behavior-cloning contract before changing the learner
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
learner consumes an observation vector and 14D action. That is useful for Stage 1
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

Natural language is not a policy input. Chromie or another planner may choose a
skill and context, but Soridormi validates bounded structured fields before
training or execution.

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

Legacy Stage 1 validation is explicit:

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/legacy_walk.jsonl \
  --allow-legacy
```

`--allow-legacy` is only a preflight bridge for Stage 1. Stage 2+ work should
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
