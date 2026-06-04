# Soridormi Context BC Dataset Prepare Report

Result: **PASS**

Output dir: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1`
Manifest: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json`
Samples read: `600`
Valid samples: `600`
Invalid samples: `0`
Skipped invalid samples: `0`
Split group field: `rollout_id`
Scenario-stratified: `True`
Seed: `0`
Shuffle: `True`

## Splits

### train

Path: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/train.jsonl`
Samples: `300`
Groups: `1`
SHA256: `0f3ebbf90e10215a8ad5798973c59bec5160b68d825a5d9e79f882649d01abfe`

Scenario counts:
- `flat_walk_varied_speed_v1`: 300

### val

Path: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/val.jsonl`
Samples: `0`
Groups: `0`
SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

### test

Path: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/test.jsonl`
Samples: `300`
Groups: `1`
SHA256: `c13f1d907c1f62388eda550b89c0fe05df7bf9f411df4b4009997d50d4af8628`

Scenario counts:
- `flat_walk_varied_speed_v1`: 300

## Warnings

- Scenario 'flat_walk_varied_speed_v1' has only 2 split group(s); validation/test splits may be empty for this scenario. Collect more independent rollouts.
