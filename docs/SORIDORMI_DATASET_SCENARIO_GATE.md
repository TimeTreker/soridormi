# Soridormi dataset scenario gate

M9D adds a strict pre-training gate for policy-supervision datasets.  It is
separate from the descriptive dataset coverage report: the coverage report tells
what is present, while this gate fails when required scenario evidence is
missing or too narrow for behavior-cloning use.

The gate supports raw JSONL teacher datasets, prepared dataset directories, and
`prepared_manifest.json` files.  It checks each scenario against the scenario
curriculum manifest and reports machine-readable JSON plus Markdown.

## What the gate checks

For each required or present scenario, the gate checks:

- scenario id exists in `configs/scenarios/open_duck_mini_v2_scenarios.json`
- minimum valid sample count
- command coverage for `vx_mps`, `vy_mps`, and `yaw_radps`
- `command_ramp_alpha` presence
- structured `task_context` and `environment_context`
- fall/stuck/termination/failure metadata presence
- maximum failure ratio

The default command source is `applied_command`, because it represents the
command actually sent to the policy/runtime after ramps are applied.  Use
`--command-source desired_command` when validating target-command coverage
instead.

## Basic use

```bash
./scripts/gate_dataset_scenario_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --require-scenario flat_walk_varied_speed_v1 \
  --min-samples-per-scenario 300 \
  --output-dir artifacts/dataset_coverage/pre_bc \
  --json | python -m json.tool
```

The wrapper runs inside `compose.sim.yaml` service `runtime`, so host Python
dependencies are not required.  Repository-relative output paths are mounted via
`/host_repo`, and `data/...` paths are translated to `/data/...`.

## Gate all current registry-ready locomotion scenarios

```bash
./scripts/gate_dataset_scenario_coverage.sh \
  /data/training_datasets/prepared/pre_bc/prepared_manifest.json \
  --require-ready-locomotion \
  --min-samples-per-scenario 1000 \
  --min-command-range-fraction 0.35 \
  --output-dir artifacts/dataset_coverage/pre_bc_gate
```

This is the recommended pre-BC gate once the first multi-scenario teacher corpus
exists.

## Recommended MuJoCo-first flow

Collect scenario-aware teacher data directly. The random teacher collector owns
the MuJoCo collection lifecycle; do not start a second `run_sim_server.sh` for
this command. Pass `--viewer` to the collector when visual inspection is needed.


```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --viewer \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 4 \
  --steps-per-episode 500 \
  --command-ramp-steps 30 \
  --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --json | python -m json.tool
```

Continue only if the collection JSON reports `ok: true` and a positive
`sample_count`.

Run descriptive coverage first:

```bash
./scripts/report_dataset_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1
```

Then run the gate:

```bash
./scripts/gate_dataset_scenario_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --require-scenario flat_walk_varied_speed_v1 \
  --min-samples-per-scenario 300 \
  --min-command-range-fraction 0.25 \
  --output-dir artifacts/dataset_coverage/flat_walk_varied_speed_v1_gate \
  --json | python -m json.tool
```

## Outputs

The output directory contains:

- `dataset_scenario_gate_summary.json`
- `dataset_scenario_gate_report.md`

The JSON schema marker is:

```json
{
  "gate_type": "soridormi.policy_supervision.scenario_gate.v1",
  "schema_version": 1
}
```

## Notes

The gate is intentionally strict by default.  During exploratory data collection,
relax one check at a time rather than disabling the gate entirely.  Useful debug
flags include:

- `--min-command-range-fraction 0.0`
- `--allow-any-failure-ratio`
- `--no-require-failure-flags`

Do not use this gate as evidence of hardware readiness.  It validates dataset
metadata and command coverage only; runtime behavior still needs MuJoCo scenario
acceptance reports from M9A/M9C.
