# Soridormi scenario curriculum

This document defines the first MuJoCo-first scenario curriculum for Soridormi, the Open Duck Mini v2 robot-body capability provider. The registry lives at `configs/scenarios/open_duck_mini_v2_scenarios.json`.

The curriculum is intentionally broader than the skills that are currently executable. The registry can include future scenarios, but only scenarios that pass static checks, simulation checks, and dataset coverage checks should be promoted into collector or evaluator defaults.

## Training and acceptance case library

Soridormi also keeps a structured training/evaluation case library in `training_cases/`.
These YAML suites map natural-language interaction requests to bounded Soridormi
skills, parameters, expected outcomes, safety checks, timeouts, and pass/fail
metrics:

- `training_cases/locomotion_basic.yaml`
- `training_cases/head_gestures.yaml`
- `training_cases/compound_skills.yaml`
- `training_cases/safety_recovery.yaml`
- `training_cases/chromie_interaction_commands.yaml`
- `training_cases/navigation_goals.yaml`

The library is intentionally an acceptance contract, not a license to pass raw
language into the low-level policy. Chromie or another planner may translate a
request such as "turn left then nod twice" into structured skill invocations,
but Soridormi still consumes bounded skill parameters and scenario context.
Cases with `planned` status document the desired curriculum without making them
eligible for training promotion.

Task-level MCP acceptance cases live separately in
`task_acceptance_cases/mcp_task_acceptance.yaml`. Those cases replay against
`soridormi.task.preview` and `soridormi.task.submit` and validate the
brain/body boundary for examples such as "walk forward for 10 seconds", "turn
left then nod twice", "bring me water", and "walk forward to the house". They
are no-motion contract tests, not low-level locomotion training data. The task
outputs may include `plan_steps` and `blocked_subsystems` to document the
embodied layers that are executable, held, or missing.
They also include `task_graph`, a derived body-DAG view for monitoring the
Soridormi-owned task skeleton without exposing raw robot control.
They also assert `recommended_next_actions` so the examples preserve the safe
Chromie routing behavior: report blocked goals, call stop tools for immediate
stop, and never lower rich missing-capability goals into velocity recipes.

The readiness surface behind those cases is `soridormi.task.get_capabilities`.
It is Soridormi-owned and backed by
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`. Update
that config when new sensing, localization, routing, manipulation, recovery, or
execution subsystems become real.

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


## Scenario-aware teacher collection

The random teacher collector can now attach a curriculum scenario to every JSONL row. When `--scenario` is supplied, the collector reads `configs/scenarios/open_duck_mini_v2_scenarios.json`, uses the scenario `vx_mps`, `vy_mps`, and `yaw_radps` ranges as defaults, and records structured context fields for future context-conditioned BC:

- `scenario_id`, `scenario_status`, `scenario_family`, and `scenario_dataset_tags`
- `skill_id` and `scenario_skills`
- `task_context` and `environment_context`
- `command_space`
- `desired_command`, `applied_command`, `command_ramp_alpha`, and the actual collector `command_ramp_name`

List configured scenarios without starting MuJoCo:

```bash
PYTHONPATH=src python -m soridormi_runtime.random_teacher_dataset_collect \
  --list-scenarios \
  --json | python -m json.tool
```

For a live MuJoCo collection run, start the simulator first:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then collect scenario-aware teacher data from another terminal:

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 4 \
  --steps-per-episode 600 \
  --command-ramp-steps 20 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1.jsonl \
  --json | python -m json.tool
```

When `--json` is set, the wrapper keeps stdout machine-readable and sends Docker/status text to stderr. If the runtime fails before the collector can emit JSON, the wrapper returns a JSON failure payload with `ok=false` and a short stderr preview instead of leaving callers with an empty pipe.

`planned` scenarios are rejected by default so they cannot silently become accepted training sources. Use `--allow-planned-scenario` only when intentionally collecting metadata-only exploratory rows before MuJoCo evaluator promotion.

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
