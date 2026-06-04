# Soridormi context BC dataset export

M9F adds the first adapter from collected teacher rows into the context-conditioned
BC contract introduced by M9E.

The target row type is:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

The adapter does not train a model and does not change runtime control. It only
converts existing JSONL data into `soridormi.policy_supervision.context_v1` rows
and validates each converted row against
`configs/training/open_duck_mini_v2_context_bc_contract_v1.json`.

## Prerequisite: collect a non-empty teacher dataset

The exporter does not connect to MuJoCo. First collect a non-empty raw teacher
JSONL. The random teacher collector owns its MuJoCo collection lifecycle: it
starts its own temporary sim container, waits for it to listen, runs the runtime
collector, and stops the sim when done. Do not start a separate
`run_sim_server.sh` for the same run. Use `--viewer` on the collector command
when visual inspection is needed.

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --viewer \
  --follow-camera \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 2 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --json | python -m json.tool
```

Continue only when the collection JSON reports `ok: true` and a positive
`sample_count`.

## Convert a scenario-aware teacher dataset

```bash
./scripts/export_context_bc_dataset.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --report artifacts/training/context_bc/flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

Then validate the exported rows against the M9E contract:

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --json | python -m json.tool
```

## Context resolution

The adapter reads each source row and resolves:

- `robot_state.observation` from `robot_state.observation` or legacy `observation`
- `teacher_action` from `teacher_action` or legacy `action`
- `desired_command` from `desired_command`, `policy_command_target`, `applied_command`, or `policy_command`
- `applied_command` from `applied_command`, `policy_command`, or `desired_command`
- `task_context` and `environment_context` from row metadata plus the scenario manifest
- `short_history.previous_action` and `short_history.previous_command` from the previous row in the same rollout

The command fields are normalized to the contract names:

```json
{"vx_mps": 0.12, "vy_mps": 0.0, "yaw_radps": 0.05}
```

Legacy `x_velocity`, `y_velocity`, and `yaw_velocity` keys are accepted as input
aliases but are not written as the primary context command field names.

## Strict context mode

Use `--strict-context` before BC training gates when every row should resolve to
a scenario manifest entry:

```bash
./scripts/export_context_bc_dataset.sh \
  /data/training_datasets/pre_bc/raw \
  --strict-context \
  --output /data/training_datasets/context_bc/pre_bc.context.jsonl
```

Without strict mode, rows with unknown scenario IDs can still be converted if
they contain enough structured context. Missing terrain defaults to `unknown`
and is reported as a warning.

## Prepared manifest input

Inputs may be raw JSONL files, directories with JSONL files, or a prepared
manifest with `train`, `val`, and `test` split paths:

```bash
./scripts/export_context_bc_dataset.sh \
  /data/training_datasets/prepared/pre_bc/prepared_manifest.json \
  --output /data/training_datasets/context_bc/pre_bc.context.jsonl
```

## No hardware

This is an offline dataset adapter. It does not connect to the simulator or
hardware. It is safe to run before MuJoCo or hardware is available.

## Empty export guard

The exporter writes through a temporary file and only replaces the requested
output when at least one valid context row is converted and validated. If the
input path is wrong, empty, or every row is invalid, the command fails and the
previous output file is left untouched. Check `converted_count`, `sample_count`,
`errors`, and `output_written` in the JSON result before running the BC contract
validator. If the JSON reports `input JSONL not found`, `no samples read from
input paths`, or `converted_count: 0`, go back to collection; do not validate or
prepare the empty context file.

## Next step: prepare train/val/test splits

After exporting context rows, prepare split files with rollout-preserving grouping:

```bash
./scripts/prepare_context_bc_dataset.sh \
  /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
  --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

The prepare step writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and
`prepared_manifest.json`. By default it groups by `rollout_id` and stratifies by
`scenario_id` to avoid adjacent-timestep leakage across splits.
