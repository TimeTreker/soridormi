# Soridormi scenario suite evaluation

scenario suite evaluation adds a batch runner around the scenario rollout evaluation single-scenario rollout evaluator.  The
suite is still MuJoCo-first and does not touch hardware.

The default suite includes only scenario-registry entries that are both:

- in a ready status (`mujoco_registry_ready`, `mujoco_eval_ready`, or
  `training_ready`), and
- backed by a locomotion policy skill that scenario rollout evaluation can evaluate.

This intentionally excludes scripted social skills, because they are governed by
`evaluate_scripted_social_skills.sh` and the social readiness report.

Generated scenario run plans use the same minimum useful forward walking speed
as skill execution: `walk_velocity` and `curve_walk` commands are not planned
below `0.12 m/s` when the scenario command range can support it. This avoids
evaluating "walking" scenarios with near-zero commands that mostly wiggle in
place.

The clearance qualification core clearance gate evaluates stable swing clearance.  The scenario
manifest keeps `min_swing_clearance_m: 0.015` and
`max_low_clearance_ratio: 0.25`, while `swing_boundary_exclusion_samples: 1`
excludes the first and last sample of each contiguous swing segment so toe-off
and touchdown transitions do not dominate the low-clearance ratio.

The current default ready locomotion suite contains the three-scenario clearance qualification core
plus the pre-WBC clearance enrichment:

- `flat_walk_varied_speed_v1`
- `start_stop_velocity_ramp_v1`
- `curve_turn_walk_v1`
- `startup_tail_clearance_v1`
- `s_turn_reversal_v1`
- `turn_stop_settle_v1`

Before WBC tuning, run the dry/offline surface gate:

```bash
./scripts/validate_pre_wbc_scenario_surface.sh
```

That command checks that the default suite, the clearance qualification core scenarios, and the WBC
clearance contract all agree on this six-scenario surface.

## Dry-run plan

```bash
./scripts/evaluate_scenario_suite.sh \
  --dry-run-only \
  --json | python -m json.tool
```

The dry-run writes:

```text
artifacts/scenario_eval/suite/suite_plan.json
```

## Live MuJoCo suite

Start MuJoCo first:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then run the suite from another terminal:

```bash
./scripts/evaluate_scenario_suite.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --output-dir artifacts/scenario_eval/suite
```

Artifacts:

```text
artifacts/scenario_eval/suite/suite_plan.json
artifacts/scenario_eval/suite/suite_summary.json
artifacts/scenario_eval/suite/suite_summary.md
artifacts/scenario_eval/suite/<scenario_id>/scenario_rollout_report.json
artifacts/scenario_eval/suite/<scenario_id>/scenario_rollout_report.md
```

The runner continues after individual scenario failures so the summary captures
all pass/fail outcomes.  The final exit code is nonzero if any expected scenario
failed or did not produce a report.

## Planned scenarios

Terrain and obstacle scenarios are still planned by default.  Include them only
when intentionally testing exploratory support:

```bash
./scripts/evaluate_scenario_suite.sh \
  --include-planned \
  --family locomotion_terrain \
  --backend mujoco \
  --profile open_duck_forward \
  --output-dir artifacts/scenario_eval/terrain_exploratory
```
