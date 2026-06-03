# Soridormi scenario suite evaluation

M9C adds a batch runner around the M9A single-scenario rollout evaluator.  The
suite is still MuJoCo-first and does not touch hardware.

The default suite includes only scenario-registry entries that are both:

- in a ready status (`mujoco_registry_ready`, `mujoco_eval_ready`, or
  `training_ready`), and
- backed by a locomotion policy skill that M9A can evaluate.

This intentionally excludes scripted social skills, because they are governed by
`evaluate_scripted_social_skills.sh` and the social readiness report.

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
