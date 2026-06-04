# Soridormi dataset coverage reports

Soridormi behavior-cloning data should be accepted only after the dataset shows
useful coverage across scenario, skill, command, ramp, terrain, and failure
metadata. The M8C coverage reporter is an offline gate for collected JSONL data
and prepared train/val/test directories.

The reporter summarizes:

- `scenario_id`, `skill_id`, split, terrain type, and dataset tag counts
- applied, desired, and policy command distributions for `vx_mps`, `vy_mps`, and `yaw_radps`
- `command_ramp_alpha` distribution
- fall, stuck, terminated, and combined failure flag ratios
- input file SHA256 digests

It accepts a raw dataset JSONL, a `prepared_manifest.json`, or a prepared dataset
directory containing train/val/test JSONL files.

## Validate a raw MuJoCo teacher dataset

Collect scenario-aware teacher data directly. The random teacher collector owns
its MuJoCo collection lifecycle, so do not start a separate `run_sim_server.sh`
for the same run. Use `--viewer` on the collector command when visual inspection
is needed.


```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --viewer \
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
`sample_count`. If the JSONL is empty, fix collection before running coverage,
export, or prepare.

Report coverage:

```bash
./scripts/report_dataset_coverage.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

The generated artifacts are:

- `dataset_coverage_summary.json`
- `dataset_coverage_report.md`

## Validate a prepared dataset before training

```bash
./scripts/prepare_training_dataset.sh \
  /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --output-dir /data/training_datasets/prepared/flat_walk_varied_speed_v1 \
  --split-group-field rollout

./scripts/report_dataset_coverage.sh \
  /data/training_datasets/prepared/flat_walk_varied_speed_v1 \
  --output-dir /data/training_datasets/coverage/flat_walk_varied_speed_v1_prepared
```

Use this report before BC training to catch accidental single-command datasets,
missing scenario metadata, missing ramp variation, overrepresented failure rows,
or train/val/test splits that lost important scenario coverage.
