# Soridormi scenario rollout evaluation

scenario rollout evaluation added a MuJoCo-first scenario rollout evaluator.  scenario acceptance thresholds adds scenario-specific
acceptance thresholds in `configs/scenarios/open_duck_mini_v2_scenarios.json`.
This is a measurement gate, not a training patch and not a hardware bridge.

The evaluator answers a small set of practical questions for one scenario:

- Did the robot make forward progress when the scenario requires progress?
- Did it fall or trip according to base-height / roll-pitch telemetry?
- Was it stuck for too much of the rollout?
- Did the log contain touchdown, cadence, step-length, and clearance telemetry?
- Which scenario and skill metadata were present in the JSONL log?
- Which scenario-specific acceptance thresholds were applied?

This follows the Soridormi direction that rough-ground or obstacle work should
not be judged by survival alone.  Progress, stuck ratio, clearance, and fall
status must be measured before collecting larger BC datasets or promoting a
scenario.

## Analyze an existing JSONL log

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --log data/logs/scenario_flat_walk_varied_speed_v1.jsonl \
  --output-dir artifacts/scenario_eval/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

Outputs:

- `scenario_run_plan.json`
- `scenario_rollout_report.json`
- `scenario_rollout_report.md`

## Run and evaluate a live MuJoCo rollout

Start MuJoCo first:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

In a second terminal:

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --backend mujoco \
  --profile open_duck_forward \
  --output-dir artifacts/scenario_eval/flat_walk_varied_speed_v1
```

When `--log` is omitted, the wrapper derives a deterministic primary-skill
command from `configs/scenarios/open_duck_mini_v2_scenarios.json`, executes it
through `run_skill_in_sim.sh`, finds the newest JSONL log with the scenario log
prefix, and evaluates that log.

## Dry-run the generated scenario command

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --backend mujoco \
  --profile open_duck_forward \
  --dry-run-only
```

This writes `scenario_run_plan.json` and prints the derived skill and arguments
without launching the runtime container.

## Scenario-specific thresholds

scenario acceptance thresholds stores normalized rollout thresholds under each scenario's
`acceptance_thresholds` object.  The evaluator uses those manifest thresholds by
default.  CLI threshold flags now mean **override the manifest**, not global
defaults.

Example manifest fragment:

```json
{
  "id": "flat_walk_varied_speed_v1",
  "acceptance_thresholds": {
    "schema_version": "soridormi.scenario_rollout_acceptance.v1",
    "min_distance_m": 0.15,
    "min_mean_forward_speed_mps": 0.03,
    "max_stuck_sample_ratio": 0.20,
    "require_not_fallen": true,
    "min_touchdown_count": 4,
    "min_swing_clearance_m": 0.015,
    "max_low_clearance_ratio": 0.25,
    "require_foot_metrics": false,
    "min_base_z_m": 0.12,
    "max_abs_roll_pitch_rad": 0.90,
    "swing_boundary_exclusion_samples": 1
  }
}
```

For clearance qualification clearance-gate scenarios, `low_clearance_swing_ratio` is measured on
stable swing samples.  `swing_boundary_exclusion_samples: 1` excludes one
sample immediately after toe-off and one sample immediately before touchdown
from each contiguous swing segment.  The 15 mm swing-clearance threshold and the
25% max low-clearance ratio are unchanged; the exclusion avoids counting normal
contact-transition samples as mid-swing foot-drag risk.  Reports include
`raw_swing_count`, `stable_swing_count`, and
`swing_boundary_exclusion_samples`.

Use overrides only for local experiments or temporary debugging:

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --log data/logs/scenario_flat_walk_varied_speed_v1.jsonl \
  --min-distance-m 0.05 \
  --json | python -m json.tool
```

The report includes:

- `acceptance_thresholds`
- `threshold_source`, either `scenario_manifest`, `explicit`, or
  `default_fallback`

Foot metrics remain optional for the first registry-ready flat scenarios, but
terrain and obstacle scenarios can set `require_foot_metrics: true` before they
are promoted to MuJoCo eval readiness.

## Relationship to existing tools

The evaluator reuses `soridormi_runtime.stride_step_metrics_eval` for low-level
stride, clearance, stuck, and fall metrics.  The scenario layer adds scenario
context, scenario/skill metadata checks, report files, and pass/fail acceptance
for a single scenario.

The next logical patch is scenario suite evaluation: a batch scenario evaluation suite that runs all
registry-ready scenarios and summarizes pass/fail status in one place.
