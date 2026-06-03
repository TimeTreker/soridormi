# Soridormi scenario rollout report

Result: PASS
Scenario: `start_stop_velocity_ramp_v1` — Start, transition, and stop command ramps
Status: mujoco_registry_ready
Family: locomotion_flat
Expected skill: walk_velocity
Threshold source: scenario_manifest
Log: data/logs/scenario_suite_start_stop_velocity_ramp_v1_20260603_095755.jsonl
Samples: 325
Duration: 6.48000 s

## Key metrics

| Metric | Value |
| --- | ---: |
| forward_distance_m | 0.28396 |
| horizontal_distance_m | 0.28786 |
| mean_forward_speed_mps | 0.04382 |
| stuck_ratio | 0.05247 |
| fallen | no |
| min_base_z_m | 0.15624 |
| touchdown_count | 25 |
| cadence_steps_per_s | 3.85802 |
| step_length_mean_m | 0.01524 |
| swing_clearance_p50_m | 0.00792 |
| low_clearance_swing_ratio | 1.00000 |

## Acceptance checks

| Check | Result | Value | Threshold | Severity |
| --- | --- | ---: | ---: | --- |
| forward_distance_m | PASS | 0.28396 | 0.10000 | error |
| mean_forward_speed_mps | PASS | 0.04382 | 0.02000 | error |
| stuck_ratio | PASS | 0.05247 | 0.25000 | error |
| not_fallen | PASS | no | no | error |
| touchdown_count | PASS | 25 | 4 | warning |
| low_clearance_swing_ratio | FAIL | 1.00000 | 0.40000 | warning |
| swing_clearance_p50_m | FAIL | 0.00792 | 0.01200 | warning |

## Acceptance thresholds

```json
{
  "contact_threshold": 0.5,
  "max_abs_roll_pitch_rad": 0.9,
  "max_low_clearance_ratio": 0.4,
  "max_stuck_sample_ratio": 0.25,
  "min_base_z_m": 0.12,
  "min_distance_m": 0.1,
  "min_mean_forward_speed_mps": 0.02,
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
      10.0
    ],
    "ramps": [
      "stand_slow_fast_stop",
      "stand_fast_slow_stop",
      "pulse_walk_stop"
    ],
    "vx_mps": [
      0.0,
      0.22
    ],
    "vy_mps": [
      -0.02,
      0.02
    ],
    "yaw_radps": [
      -0.05,
      0.05
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

- low swing-clearance ratio is high: 1.000 > 0.400
- log does not carry scenario_id metadata; evaluated against requested scenario only
- low_clearance_swing_ratio: 1.0 > 0.4
- swing_clearance_p50_m: 0.007919188662089231 < 0.012

## Errors

- none
