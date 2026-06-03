# Soridormi scenario rollout report

Result: PASS
Scenario: `curve_turn_walk_v1` — Curved walking and in-place turning envelope
Status: mujoco_registry_ready
Family: locomotion_turning
Expected skill: curve_walk
Threshold source: scenario_manifest
Log: data/logs/scenario_suite_curve_turn_walk_v1_20260603_095842.jsonl
Samples: 275
Duration: 5.48000 s

## Key metrics

| Metric | Value |
| --- | ---: |
| forward_distance_m | 0.12450 |
| horizontal_distance_m | 0.12829 |
| mean_forward_speed_mps | 0.02272 |
| stuck_ratio | 0.25182 |
| fallen | no |
| min_base_z_m | 0.15687 |
| touchdown_count | 22 |
| cadence_steps_per_s | 4.01460 |
| step_length_mean_m | 0.02711 |
| swing_clearance_p50_m | 0.00556 |
| low_clearance_swing_ratio | 1.00000 |

## Acceptance checks

| Check | Result | Value | Threshold | Severity |
| --- | --- | ---: | ---: | --- |
| forward_distance_m | PASS | 0.12450 | 0.06000 | error |
| mean_forward_speed_mps | PASS | 0.02272 | 0.01500 | error |
| stuck_ratio | PASS | 0.25182 | 0.30000 | error |
| not_fallen | PASS | no | no | error |
| touchdown_count | PASS | 22 | 4 | warning |
| low_clearance_swing_ratio | FAIL | 1.00000 | 0.45000 | warning |
| swing_clearance_p50_m | FAIL | 0.00556 | 0.01200 | warning |

## Acceptance thresholds

```json
{
  "contact_threshold": 0.5,
  "max_abs_roll_pitch_rad": 0.95,
  "max_low_clearance_ratio": 0.45,
  "max_stuck_sample_ratio": 0.3,
  "min_base_z_m": 0.12,
  "min_distance_m": 0.06,
  "min_mean_forward_speed_mps": 0.015,
  "min_swing_clearance_m": 0.012,
  "min_touchdown_count": 4,
  "require_foot_metrics": false,
  "require_not_fallen": true
}
```

## Scenario context

```json
{
  "command_space": {
    "duration_s": [
      3.0,
      8.0
    ],
    "ramps": [
      "straight_curve_straight",
      "curve_reverse_curve",
      "turn_stop"
    ],
    "vx_mps": [
      -0.02,
      0.18
    ],
    "vy_mps": [
      -0.02,
      0.02
    ],
    "yaw_radps": [
      -0.2,
      0.2
    ]
  },
  "environment_context": {
    "friction_range": [
      0.8,
      1.2
    ],
    "obstacles": [],
    "terrain_height_m_range": [
      0.0,
      0.0
    ],
    "terrain_type": "flat"
  },
  "task_context": {
    "gait_style": "default_walk",
    "requires_obstacle_crossing": false,
    "requires_progress": true,
    "skill_family": "locomotion"
  }
}
```

## Warnings

- low swing-clearance ratio is high: 1.000 > 0.450
- log does not carry scenario_id metadata; evaluated against requested scenario only
- low_clearance_swing_ratio: 1.0 > 0.45
- swing_clearance_p50_m: 0.005556822984959235 < 0.012

## Errors

- none
