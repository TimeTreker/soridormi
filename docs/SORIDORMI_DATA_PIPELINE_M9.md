# Soridormi M9 data pipeline runbook

This runbook is the source-of-truth command order for the M9 locomotion data
pipeline.  It separates commands by simulator ownership so dataset collection does
not fight a second MuJoCo server.

## Simulator ownership rules

There are two live-MuJoCo patterns:

1. **External-sim tools** expect a sim server that was started in another
   terminal.  Use this pattern for scenario rollout evaluation, scenario suite
   evaluation, free-walk policy checks, scripted social skills, and look-target
   commands.
2. **Collector-owned-sim tools** own their MuJoCo collection lifecycle.  Use this
   pattern for `collect_random_teacher_dataset.sh`.  The wrapper starts a
   temporary simulator container, waits for the API port, runs the runtime
   collector, and then stops the simulator.  Do not start a second
   `run_sim_server.sh` for the same collection run; that can create reset
   contention and leave the JSONL empty.

When a collector-owned tool supports `--viewer`, treat it as a request for the
collector's own MuJoCo server to use the viewer.  It is not a reminder to
manually start a separate viewer server.  Use `--external-sim` only for advanced
debugging when you intentionally want to connect to an already-running server.

## End-to-end flat-walk context BC data flow

### 1. Collect raw scenario-aware teacher rows

Run the collector directly.  Do **not** start `run_sim_server.sh` in another
terminal for this command.

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 2 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --reset-attempts 10 \
  --reset-retry-sleep 0.5 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --json | python -m json.tool
```

For visual inspection during collection, request the viewer on the same command:

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

A successful collection must report `ok: true` and a positive `sample_count`.
If the output reports `sample_count: 0`, stop here and fix the collection issue
before export or prepare.  A transient reset error such as
`Again('Resource temporarily unavailable')` should be retried by the collector;
raise `--reset-attempts`/`--reset-retry-sleep` only if the sim is still warming
up slowly.

Quick check inside the runtime data volume:

```bash
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  wc -l /data/training_datasets/flat_walk_varied_speed_v1.jsonl
  head -n 1 /data/training_datasets/flat_walk_varied_speed_v1.jsonl
'
```

### 2. Run descriptive dataset coverage

```bash
./scripts/report_dataset_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

### 3. Run the scenario coverage gate

```bash
./scripts/gate_dataset_scenario_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --require-scenario flat_walk_varied_speed_v1 \
  --min-samples-per-scenario 300 \
  --min-command-range-fraction 0.25 \
  --output-dir artifacts/dataset_coverage/flat_walk_varied_speed_v1_gate \
  --json | python -m json.tool
```

### 4. Export context BC rows

```bash
./scripts/export_context_bc_dataset.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --report artifacts/training/context_bc/flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

A successful export must report `ok: true`, `converted_count > 0`, and
`output_written: true`.  The exporter intentionally refuses to overwrite the
output with an empty file.

### 5. Validate context rows against the BC contract

```bash
./scripts/validate_bc_training_contract.sh \
  --sample-jsonl /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --json | python -m json.tool
```

A valid sample report must have `sample_validation.ok: true` and a positive
`context_sample_count`.

### 6. Prepare train/val/test splits

```bash
./scripts/prepare_context_bc_dataset.sh \
  /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
  --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

The prepare step should report positive split sample counts.  If it reports
`no samples read from input paths`, the context JSONL is empty or missing; go
back to collection/export instead of trying to train.

### 7. Gate the prepared train/val/test artifact

```bash
./scripts/gate_context_bc_prepared_dataset.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --require-scenario flat_walk_varied_speed_v1 \
  --output-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

The gate must report `ok: true`.  A split with zero samples or zero rollout
groups is not training-ready.

### 8. Build the training-ready bundle

After both the scenario coverage gate and prepared dataset gate pass, bundle
their outputs with the prepared manifest, BC contract, file hashes, and
recommended training commands:

```bash
./scripts/build_context_bc_training_ready_report.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --scenario-gate artifacts/dataset_coverage/flat_walk_varied_speed_v1_gate/dataset_scenario_gate_summary.json \
  --prepared-gate artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1/prepared_context_gate_report.json \
  --profile-name context_stage1_candidate \
  --output-dir artifacts/training/context_bc/training_ready/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

This report is the final offline readiness checkpoint before running
`train_behavior_clone.sh` or `train_neural_behavior_clone.sh`.

### Full prepare → gate → training-ready pipeline

To run prepare, prepared-gate, and training-ready report generation in one step,
use:

```bash
./scripts/run_context_bc_training_ready_pipeline.sh \
  /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
  --scenario-gate artifacts/dataset_coverage/flat_walk_varied_speed_v1_gate/dataset_scenario_gate_summary.json \
  --require-scenario flat_walk_varied_speed_v1 \
  --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
  --prepared-gate-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1 \
  --training-ready-dir artifacts/training/context_bc/training_ready/flat_walk_varied_speed_v1 \
  --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
  --json | python -m json.tool
```

This wrapper script is useful when you want the M9 data pipeline to proceed from
prepared dataset creation through readiness reporting without manual staging.

## External-sim eval example

Scenario evaluation is not the same as teacher dataset collection.  It uses the
external-sim pattern because it runs policy/skill rollouts against a live server.

Terminal 1:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Terminal 2:

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --backend mujoco \
  --profile open_duck_forward \
  --output-dir artifacts/scenario_eval/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

## Troubleshooting empty files

- Empty-file SHA256 is
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- If collection reports `sample_count: 0`, the raw teacher dataset is not usable.
- If export reports `converted_count: 0`, validate the raw input path and row
  schema before running prepare.
- If prepare reports all split counts as zero, the context JSONL input is empty.
- Do not continue the BC pipeline after an `ok: false` JSON result; fix the
  earliest failing stage first.
