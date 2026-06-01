# Soridormi scenario curriculum

This document defines the first MuJoCo-first scenario curriculum for Soridormi, the Open Duck Mini v2 robot-body capability provider. The registry lives at `configs/scenarios/open_duck_mini_v2_scenarios.json`.

The curriculum is intentionally broader than the skills that are currently executable. The registry can include future scenarios, but only scenarios that pass static checks, simulation checks, and dataset coverage checks should be promoted into collector or evaluator defaults.

## Policy contract

Soridormi's low-level policy should remain structured and bounded:

```text
robot_state + desired_command/desired_state + task_context + environment_context + short_history -> action_14d
```

Raw natural language should not be passed to the low-level policy. Chromie or another planner can translate language into a structured skill invocation and scenario context, but the policy should consume numeric and categorical fields.

## Stage plan

### Stage 1: flat command-conditioned locomotion

Initial behavior cloning should cover continuous command variation on flat ground:

- `flat_walk_varied_speed_v1`
- `start_stop_velocity_ramp_v1`
- `curve_turn_walk_v1`

These scenarios are registry-ready first because they do not require new obstacle objects or perception features. They focus on continuous `vx`, `vy`, and `yaw` ranges plus smooth command ramps.

### Stage 2: transition and tracking robustness

Use the same flat-ground scenarios to evaluate command tracking, stop quality, stuck ratio, and terminal velocity. Do not rely on a single clean walking rollout.

### Stage 3: terrain context

Terrain scenarios add bounded environment context:

- `rough_ground_walk_v1`
- `small_stones_walk_v1`

Surviving rough ground is not enough. Evaluation should measure progress, fall status, stuck ratio, and foot clearance.

### Stage 4: obstacle context

Obstacle scenarios introduce explicit obstacle metadata before any full navigation stack:

- `corridor_keep_heading_v1`
- `stop_before_obstacle_v1`
- `step_over_low_obstacle_v1`

The stop-before-obstacle scenario should precede step-over behavior. Step-over should remain conservative until stride and clearance metrics are trustworthy.

### Stage 5: recovery and safe social behaviors

Recovery and social behaviors should be staged after baseline locomotion is stable:

- `recovery_after_push_v1`
- `look_direction_stationary_v1`

Safe social skills should use head, neck, or body-safe motions only. Arm and hand skills remain unsupported until there is a current actuator contract.

## Required metadata

Every scenario entry must define:

- `id`, `title`, `status`, `priority`, `family`, and `skills`
- `task_context`
- `environment_context`
- `command_space`
- `success_metrics.required`
- `dataset_tags`

The scenario `status` must be one of the manifest-level `status_values`.

## Promotion gates

A scenario should be promoted in stages:

1. `planned`: documented intent only.
2. `mujoco_registry_ready`: static metadata is valid and tests pass.
3. `mujoco_eval_ready`: a deterministic MuJoCo world/evaluator exists and reports metrics.
4. `training_ready`: collector integration records scenario IDs, command coverage, context fields, and failure/stuck metadata.

## Dataset coverage expectations

Teacher data rows should eventually include:

- `scenario_id`
- `skill_id`
- robot state
- desired command or desired state
- applied command
- command ramp name and ramp alpha
- task context
- environment context
- teacher action
- fall, stuck, progress, clearance, and obstacle metadata when available

Coverage reports should group by `scenario_id`, `skill_id`, velocity/yaw buckets, terrain type, obstacle type, ramp name, and failure/stuck flags.

## Validation

Use the patch-specific test first:

```bash
PYTHONPATH=src pytest -q tests/test_scenario_manifest_m8.py
```

Then run the broader static checks:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```
