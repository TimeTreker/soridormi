# Soridormi dataset scenario gate

Result: **FAILED**
Samples: **0 valid** / 0 total
Command source: `applied_command`
Required scenarios: `flat_walk_varied_speed_v1`

## Scenario results

| Scenario | Required | Samples | Failure ratio | Result |
| --- | --- | ---: | ---: | --- |
| `flat_walk_varied_speed_v1` | yes | 0 | 0.000 | FAILED |

## Command range coverage

### `flat_walk_varied_speed_v1`

| Command | Count | Min | Max | Covered fraction |
| --- | ---: | ---: | ---: | ---: |
| `vx_mps` | 0 | n/a | n/a | n/a |
| `vy_mps` | 0 | n/a | n/a | n/a |
| `yaw_radps` | 0 | n/a | n/a | n/a |

Errors:
- min_samples_per_scenario: 0 samples >= required 300


## Errors

- dataset input not found: /data/training_datasets/flat_walk_varied_speed_v1.jsonl
- flat_walk_varied_speed_v1: min_samples_per_scenario: 0 samples >= required 300
