# Soridormi scenario rollout report

Result: PASS
Scenario: `flat_walk_varied_speed_v1` — Flat walk with varied continuous speed
Status: mujoco_registry_ready
Family: locomotion_flat
Expected skill: walk_velocity
Log: data/logs/scenario_flat_walk_varied_speed_v1_20260603_093311.jsonl
Samples: 250
Duration: 4.98000 s

## Key metrics

| Metric | Value |
| --- | ---: |
| forward_distance_m | 0.31918 |
| horizontal_distance_m | 0.32380 |
| mean_forward_speed_mps | 0.06409 |
| stuck_ratio | 0.00402 |
| fallen | no |
| min_base_z_m | 0.15761 |
| touchdown_count | 18 |
| cadence_steps_per_s | 3.61446 |
| step_length_mean_m | 0.01826 |
| swing_clearance_p50_m | 0.01002 |
| low_clearance_swing_ratio | 1.00000 |

## Acceptance checks

| Check | Result | Value | Threshold | Severity |
| --- | --- | ---: | ---: | --- |
| forward_distance_m | PASS | 0.31918 | 0.05000 | error |
| mean_forward_speed_mps | PASS | 0.06409 | 0.02000 | error |
| stuck_ratio | PASS | 0.00402 | 0.40000 | error |
| not_fallen | PASS | no | no | error |
| touchdown_count | PASS | 18 | 4 | warning |
| low_clearance_swing_ratio | FAIL | 1.00000 | 0.35000 | warning |
| swing_clearance_p50_m | FAIL | 0.01002 | 0.01500 | warning |

## Scenario context

```json
{
  "command_space": {
    "duration_s": [
      2.0,
      8.0
    ],
    "ramps": [
      "hold",
      "smooth_start",
      "smooth_stop",
      "slow_fast_slow"
    ],
    "vx_mps": [
      -0.03,
      0.25
    ],
    "vy_mps": [
      -0.03,
      0.03
    ],
    "yaw_radps": [
      -0.08,
      0.08
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

- low swing-clearance ratio is high: 1.000 > 0.350
- log does not carry scenario_id metadata; evaluated against requested scenario only
- low_clearance_swing_ratio: 1.0 > 0.35
- swing_clearance_p50_m: 0.010023828465555135 < 0.015

## Errors

- none
