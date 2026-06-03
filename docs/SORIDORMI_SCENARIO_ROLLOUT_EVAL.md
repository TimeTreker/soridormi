# Soridormi scenario rollout evaluation

M9A adds a MuJoCo-first scenario rollout evaluator.  It is a measurement gate,
not a training patch and not a hardware bridge.

The evaluator answers a small set of practical questions for one scenario:

- Did the robot make forward progress when the scenario requires progress?
- Did it fall or trip according to base-height / roll-pitch telemetry?
- Was it stuck for too much of the rollout?
- Did the log contain touchdown, cadence, step-length, and clearance telemetry?
- Which scenario and skill metadata were present in the JSONL log?

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

## Current acceptance thresholds

M9A keeps thresholds as CLI defaults because we do not yet have enough measured
baselines to store them in the scenario manifest.  M9B should move scenario-
specific thresholds into the manifest after we inspect early reports.

Defaults:

- `--min-distance-m 0.05`
- `--min-mean-forward-speed-mps 0.02`
- `--max-stuck-sample-ratio 0.40`
- `--min-touchdown-count 4`
- `--min-swing-clearance-m 0.015`
- `--max-low-clearance-ratio 0.35`
- `--min-base-z-m 0.12`
- `--max-abs-roll-pitch-rad 0.90`

Foot metrics are warning-level by default for compatibility with older logs.
Use `--require-foot-metrics` when evaluating logs that must include foot pose
and contact telemetry.

## Relationship to existing tools

M9A reuses `soridormi_runtime.stride_step_metrics_eval` for low-level stride,
clearance, stuck, and fall metrics.  The new layer adds scenario context,
scenario/skill metadata checks, report files, and pass/fail acceptance for a
single scenario.

The next logical patch is M9B: scenario-specific acceptance thresholds in the
scenario manifest or a companion config.
