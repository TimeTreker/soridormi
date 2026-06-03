# Soridormi dataset scenario gate

Result: **FAILED**
Samples: **0 valid** / 0 total
Command source: `applied_command`
Required scenarios: `flat_walk_varied_speed_v1`, `start_stop_velocity_ramp_v1`, `curve_turn_walk_v1`

## Scenario results

| Scenario | Required | Samples | Failure ratio | Result |
| --- | --- | ---: | ---: | --- |
| `curve_turn_walk_v1` | yes | 0 | 0.000 | FAILED |
| `flat_walk_varied_speed_v1` | yes | 0 | 0.000 | FAILED |
| `start_stop_velocity_ramp_v1` | yes | 0 | 0.000 | FAILED |

## Command range coverage

### `curve_turn_walk_v1`

| Command | Count | Min | Max | Covered fraction |
| --- | ---: | ---: | ---: | ---: |
| `vx_mps` | 0 | n/a | n/a | n/a |
| `vy_mps` | 0 | n/a | n/a | n/a |
| `yaw_radps` | 0 | n/a | n/a | n/a |

Errors:
- min_samples_per_scenario: 0 samples >= required 1000

### `flat_walk_varied_speed_v1`

| Command | Count | Min | Max | Covered fraction |
| --- | ---: | ---: | ---: | ---: |
| `vx_mps` | 0 | n/a | n/a | n/a |
| `vy_mps` | 0 | n/a | n/a | n/a |
| `yaw_radps` | 0 | n/a | n/a | n/a |

Errors:
- min_samples_per_scenario: 0 samples >= required 1000

### `start_stop_velocity_ramp_v1`

| Command | Count | Min | Max | Covered fraction |
| --- | ---: | ---: | ---: | ---: |
| `vx_mps` | 0 | n/a | n/a | n/a |
| `vy_mps` | 0 | n/a | n/a | n/a |
| `yaw_radps` | 0 | n/a | n/a | n/a |

Errors:
- min_samples_per_scenario: 0 samples >= required 1000


## Errors

- prepared manifest not found: /data/training_datasets/prepared/pre_bc/prepared_manifest.json
- curve_turn_walk_v1: min_samples_per_scenario: 0 samples >= required 1000
- flat_walk_varied_speed_v1: min_samples_per_scenario: 0 samples >= required 1000
- start_stop_velocity_ramp_v1: min_samples_per_scenario: 0 samples >= required 1000
