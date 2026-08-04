# Soridormi prepared context BC dataset gate

prepared context dataset gate adds a final offline gate between context BC dataset preparation and BC
training. It validates the prepared train/validation/test artifact instead of
raw collection rows.

The gate checks:

- `prepared_manifest.json` exists and has dataset type `soridormi.policy_supervision.context_prepared.v1`.
- The prepare step reported `ok: true` unless explicitly overridden.
- `train.jsonl`, `val.jsonl`, and `test.jsonl` exist and contain at least the requested minimum samples.
- Each split row validates against `open_duck_mini_v2_context_bc_contract_v1.json`.
- Manifest sample counts, scenario counts, group counts, and SHA256 digests match the current split files.
- Required scenarios are present with enough samples.
- Rollout groups do not leak across splits.

This is intentionally separate from `gate_dataset_scenario_coverage.sh`:

- `gate_dataset_scenario_coverage.sh` gates raw/prepared scenario coverage before export or after collection.
- `gate_context_bc_prepared_dataset.sh` gates the final context BC train/val/test artifact before training.

## Example

```bash
./scripts/gate_context_bc_prepared_dataset.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json \
  --require-scenario flat_walk_varied_speed_v1 \
  --min-train-samples 1 \
  --min-val-samples 1 \
  --min-test-samples 1 \
  --output-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

For the future full pre-BC corpus:

```bash
./scripts/gate_context_bc_prepared_dataset.sh \
  /data/training_datasets/context_bc/prepared/pre_bc/prepared_manifest.json \
  --require-ready-locomotion \
  --min-samples-per-required-scenario 1000 \
  --output-dir artifacts/training/context_bc/prepared_gate/pre_bc
```

## Empty file troubleshooting

If this gate reports that a split is empty, do not train. Trace backward:

1. `collect_random_teacher_dataset.sh` should have produced a non-empty raw JSONL.
2. `export_context_bc_dataset.sh` should report `converted_count > 0` and `output_written: true`.
3. `prepare_context_bc_dataset.sh` should report `ok: true` and non-empty split counts.
4. This gate should then pass.

The empty SHA256 value
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` means a file
has zero bytes.
