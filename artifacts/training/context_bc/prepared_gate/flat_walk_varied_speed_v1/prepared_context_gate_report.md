# Soridormi Prepared Context BC Dataset Gate

Result: **FAIL**

Manifest: `/data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1/prepared_manifest.json`
Dataset type: `soridormi.policy_supervision.context_prepared.v1`
Manifest ok: `True`
Total samples: `600`
Split group field: `rollout_id`
Require no group leakage: `True`

## Splits

| Split | Samples | Groups | Exists | SHA256 |
|---|---:|---:|---|---|
| train | 300 | 1 | True | `0f3ebbf90e10215a8ad5798973c59bec5160b68d825a5d9e79f882649d01abfe` |
| val | 0 | 0 | True | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| test | 300 | 1 | True | `c13f1d907c1f62388eda550b89c0fe05df7bf9f411df4b4009997d50d4af8628` |

## Scenario counts

- `flat_walk_varied_speed_v1`: 600

## Required scenarios

- `flat_walk_varied_speed_v1`

## Errors

- split 'val' has 0 sample(s); minimum required is 1
