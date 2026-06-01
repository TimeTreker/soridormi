# Soridormi stride/step metrics evaluation

M8D adds an offline JSONL analyzer for MuJoCo-first walking validation. It is meant to complement dataset coverage reports and foot-clearance checks before any hardware work.

The analyzer reads runtime JSONL rows with `state.base_position_xyz`, optional `state.base_quat_wxyz`, `state.feet_position_xyz`, and `state.feet_contacts`. It also preserves scenario-aware metadata when rows include `scenario_id` and `skill_id`.

## Metrics

The report includes:

- forward/lateral/base displacement
- mean and instantaneous forward speed
- speed-based stuck sample ratio
- explicit stuck/failure/fall flags when present
- low-base-height and roll/pitch fall detection
- touchdown counts for left/right feet
- cadence in touchdown events per second
- alternating touchdown ratio
- touchdown step length and same-foot stride progress
- swing foot-clearance distribution and low-clearance ratio

These metrics are intentionally acceptance-gate friendly. A rollout that merely survives rough ground is not enough; it must also show forward progress, reasonable touchdown structure, adequate swing clearance, and no fall/stuck indicators.

## Static validation

```bash
PYTHONPATH=src pytest -q tests/test_stride_step_metrics_eval_m8d.py
python -m compileall -q src tests
bash -n scripts/run_stride_step_metrics_eval.sh
```

## MuJoCo validation

Start the simulator in one terminal:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Run the stride/step smoke evaluation in a second terminal:

```bash
./scripts/run_stride_step_metrics_eval.sh open_duck_forward \
  --steps 1200 \
  --output-dir data/stride_step/open_duck_forward_flat
```

The wrapper writes:

- `stride_step_metrics_report.md`
- `stride_step_metrics_report.json`

You can also analyze an existing JSONL log directly:

```bash
PYTHONPATH=src python -m soridormi_runtime.stride_step_metrics_eval \
  data/logs/some_rollout.jsonl \
  --output data/stride_step/some_rollout/stride_step_metrics_report.md \
  --json-output data/stride_step/some_rollout/stride_step_metrics_report.json
```

## Threshold tuning

Defaults are conservative smoke-test thresholds:

- `--min-forward-speed 0.02`
- `--max-stuck-sample-ratio 0.40`
- `--min-base-z 0.12`
- `--max-abs-roll-pitch 0.90`
- `--min-touchdown-count 4`
- `--min-step-length 0.01`
- `--min-swing-clearance 0.015`
- `--max-low-clearance-ratio 0.35`

Use stricter thresholds for acceptance suites after scenario-specific baselines are established.
