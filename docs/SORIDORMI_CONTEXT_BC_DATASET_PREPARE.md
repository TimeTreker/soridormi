# Soridormi context BC dataset prepare

M9G prepares exported context-conditioned BC rows into train/val/test JSONL files.
It is an offline dataset step only: it does not change training code, simulator
control, or hardware behavior.

The input rows should already satisfy the M9E context contract:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

## Why grouped splits are required

Walking datasets contain highly correlated adjacent timesteps. A sample-level
random split can place timestep `N` in train and timestep `N+1` in validation,
which makes validation look better than true closed-loop generalization. M9G
therefore defaults to grouping by `rollout_id`. All rows from one rollout stay in
the same split.

The default also stratifies by `scenario_id`, so each scenario is split using its
own rollout groups where enough groups are available.

## Prerequisite: non-empty context JSONL

Prepare only runs after a successful context export. The input context JSONL must
contain rows and the export report should show `ok: true`, `converted_count > 0`,
and `output_written: true`. If prepare reports `no samples read from input
paths`, the context file is empty or missing; return to collection/export instead
of trying to train.

## Basic usage

```bash
./scripts/prepare_context_bc_dataset.sh \
  /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
  --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

Outputs:

```text
/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/train.jsonl
/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/val.jsonl
/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/test.jsonl
/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json
```

Validate a split against the context contract:

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/train.jsonl \
  --json | python -m json.tool
```

## Multi-scenario pre-BC corpus

```bash
./scripts/prepare_context_bc_dataset.sh \
  /data/training_datasets/context_bc/*.context.jsonl \
  --output-dir /data/training_datasets/context_bc/prepared/pre_bc \
  --seed 7 \
  --report artifacts/training/context_bc/prepared_pre_bc.md
```

Then run the scenario gate against the prepared manifest:

```bash
./scripts/gate_dataset_scenario_coverage.sh \
  /data/training_datasets/context_bc/prepared/pre_bc/prepared_manifest.json \
  --require-ready-locomotion \
  --min-samples-per-scenario 1000 \
  --min-command-range-fraction 0.35
```

## Important options

- `--split-group-field rollout_id` keeps complete rollouts together. This is the
  default and should usually stay enabled.
- `--split-group-field episode_id` can be useful if the exporter lacks rollout
  ids but preserved episode metadata.
- `--no-stratify-by-scenario` disables per-scenario group splitting.
- `--skip-invalid` intentionally skips invalid rows without failing the prepare
  result. Avoid this for final training corpora unless the invalid-row report has
  been reviewed.
- `--no-shuffle` preserves first-seen group order. The default deterministic
  hash shuffle is preferred for normal training preparation.

## Validation template

```bash
git apply --check ~/Downloads/soridormi_m9g_context_bc_dataset_prepare.patch
git apply ~/Downloads/soridormi_m9g_context_bc_dataset_prepare.patch

PYTHONPATH=src pytest -q \
  tests/test_context_bc_dataset_prepare_m9g.py \
  tests/test_context_bc_dataset_export_m9f.py \
  tests/test_bc_training_contract_m9e.py

python -m compileall -q src tests
bash -n scripts/prepare_context_bc_dataset.sh scripts/export_context_bc_dataset.sh
```
