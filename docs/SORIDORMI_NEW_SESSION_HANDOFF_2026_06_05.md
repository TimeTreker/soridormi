# Soridormi new-session handoff — 2026-06-05

This document is the handoff for continuing Soridormi in a fresh ChatGPT session. It summarizes the current project state after the M8 social-skill slice and the M9 scenario/data-pipeline slice.

## 1. Copy this prompt into the new chat

We are continuing Soridormi, my Open Duck Mini v2 humanoid robot stack. Focus on Soridormi only unless I explicitly switch to Chromie. Use normal `.patch` files, assume they are downloaded to `~/Downloads`, and every patch must include `git apply --check` plus functional validation commands. Keep hardware disabled unless explicitly requested. The current direction is MuJoCo-first validation, scenario-aware locomotion data, and a context-conditioned BC pipeline:

```text
robot_state + desired_command/desired_state + task_context + environment_context + short_history -> action_14d
```

Do not feed raw natural language or raw perception directly into the low-level 14D action policy. Higher layers should translate those inputs into structured skill/context fields.

## 2. Current operating rules

- Soridormi is the robot-body capability provider.
- Chromie is the higher-level LLM/MCP/planner/voice layer and should stay out of scope unless explicitly requested.
- Use MuJoCo first. Do not resume hardware bridge work by default.
- Deliver small incremental patches with tests.
- Use Docker/container wrappers consistently.
- For user-facing commands, prefer machine-readable JSON when `--json` is used; status/noise should go to stderr.
- If a command fails with an empty JSONL, inspect the previous pipeline step before continuing.

## 3. Patch/application rules

Every future patch response should include at least:

```bash
cd /home/chromie/github/soridormi

git apply --check ~/Downloads/<patch-name>.patch
git apply ~/Downloads/<patch-name>.patch
```

And targeted validation, for example:

```bash
PYTHONPATH=src pytest -q <relevant tests>
python -m compileall -q src tests
bash -n <changed shell scripts>
```

For MuJoCo runtime validation, include explicit `--backend mujoco`. Avoid vague "run the sim" instructions.

## 4. Important Docker/runtime convention update

There are now two different runtime ownership patterns.

### 4.1 Random teacher dataset collection owns its temporary sim

For `collect_random_teacher_dataset.sh`, do **not** start `run_sim_server.sh` separately. The collector wrapper should own the MuJoCo collection lifecycle by default. Use `--viewer --follow-camera` directly on the collector command when a viewer is needed.

Example:

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --viewer \
  --follow-camera \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 10 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --reset-attempts 10 \
  --reset-retry-sleep 0.5 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1_10ep.jsonl \
  --json | python -m json.tool
```

Expected success shape:

```json
{
  "ok": true,
  "sample_count": 3000
}
```

If collection fails during reset with `Again('Resource temporarily unavailable')`, verify that no separate sim server or previous collector/evaluator is competing for the same API port. The collector also has reset retries.

### 4.2 Eval/runtime tools still use external sim server

Scenario rollout eval, scenario suite eval, scripted social skills, and policy runtime skill execution use the two-terminal pattern unless their wrapper says otherwise.

Terminal 1:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Terminal 2 example:

```bash
./scripts/evaluate_scenario_rollout.sh \
  --scenario flat_walk_varied_speed_v1 \
  --backend mujoco \
  --profile open_duck_forward \
  --output-dir artifacts/scenario_eval/flat_walk_varied_speed_v1 \
  --json | python -m json.tool
```

## 5. Completed M8 social-skill slice

The safe scripted social-skill stack was built and debugged.

Implemented/validated skills:

- `look_direction`
- `nod_yes`
- `shake_no`
- `neutral_head`
- `bow`
- `express_attention`
- `look_at_person`

Key design decisions:

- Social skills are MuJoCo-only experimental unless explicitly promoted.
- They are head/neck-only and preserve non-head actuator controls.
- The final working model is: plan a bounded head-pose trajectory, then stream pose commands step by step.
- `shake_no` starts neutral, yaw-left/yaw-right for at least two visible cycles, then returns neutral.
- `nod_yes` starts neutral, pitch down/up for at least two visible cycles, then returns neutral.
- `look_at_person` should hold gaze by default and only return neutral when explicitly requested.
- `look_at_person` does **not** know where a person is by itself. It consumes a structured target such as `target_yaw_rad` and `target_pitch_rad` from a target provider or future perception module.

Useful commands:

```bash
./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"medium","duration_s":2.0}' \
  --backend mujoco \
  --control-hz 50
```

```bash
./scripts/run_look_at_person_target.sh \
  --image-x-norm 0.75 \
  --image-y-norm 0.45 \
  --duration-s 4.0 \
  --backend mujoco
```

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --execute \
  --backend mujoco \
  --require-observed
```

## 6. Completed M9 scenario/data-pipeline slice so far

The project moved back to locomotion/scenario training readiness.

Implemented components include:

- Scenario rollout evaluator.
- Scenario acceptance thresholds.
- Batch scenario eval suite.
- Dataset scenario coverage gate.
- BC training contract.
- Context BC dataset exporter.
- Empty-export guard.
- Context BC dataset prepare/split.
- Prepared context dataset gate.
- Collector-owned sim lifecycle and reset retry for random teacher collection.
- Documentation updates distinguishing collector-owned sim vs external sim tools.

## 7. Current BC data pipeline

The intended smoke pipeline is:

```text
collect raw teacher JSONL
  -> export context BC JSONL
  -> prepare train/val/test splits
  -> gate prepared dataset
```

### 7.1 Collect raw teacher data

Use at least 10 episodes for a useful train/val/test split because splitting preserves whole `rollout_id` groups to avoid leakage.

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --viewer \
  --follow-camera \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 10 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --reset-attempts 10 \
  --reset-retry-sleep 0.5 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1_10ep.jsonl \
  --json | python -m json.tool
```

### 7.2 Export context BC rows

```bash
./scripts/export_context_bc_dataset.sh \
  /data/training_datasets/flat_walk_varied_speed_v1_10ep.jsonl \
  --output /data/training_datasets/context_bc/flat_walk_varied_speed_v1_10ep.context.jsonl \
  --report artifacts/training/context_bc/flat_walk_varied_speed_v1_10ep.md \
  --json | python -m json.tool
```

Expected success shape:

```json
{
  "ok": true,
  "converted_count": 3000,
  "output_written": true
}
```

### 7.3 Prepare train/val/test splits

```bash
./scripts/prepare_context_bc_dataset.sh \
  /data/training_datasets/context_bc/flat_walk_varied_speed_v1_10ep.context.jsonl \
  --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1_10ep \
  --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1_10ep.md \
  --json | python -m json.tool
```

### 7.4 Gate prepared dataset

```bash
./scripts/gate_context_bc_prepared_dataset.sh \
  /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1_10ep/prepared_manifest.json \
  --require-scenario flat_walk_varied_speed_v1 \
  --min-train-samples 1 \
  --min-val-samples 1 \
  --min-test-samples 1 \
  --output-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1_10ep \
  --json | python -m json.tool
```

## 8. Current known result from the latest run

A 2-episode collection/export/prepare succeeded with 600 valid samples, but the prepared gate failed because the validation split was empty. This is expected: with only 2 rollout groups and group-preserving splitting, train/test can be filled but val cannot.

Interpretation:

```text
600 samples is enough to prove the pipeline works.
2 rollout groups is not enough for a real train/val/test split.
Collect more independent episodes, preferably 10+.
```

For a temporary smoke gate on 2 episodes, `--min-val-samples 0` can be used, but that should not be used for real BC training.

## 9. Troubleshooting quick reference

### Empty context JSONL

If validation says:

```text
Sample JSONL is empty
```

or the SHA256 is:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

then the file is empty. Check the upstream collection/export result before running prepare.

### Export converted 0 rows

If export reports:

```text
input JSONL not found
no samples read from input paths
```

then the raw teacher JSONL does not exist inside the runtime container. Run the collector first and verify line count with:

```bash
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  wc -l /data/training_datasets/<file>.jsonl
  head -n 1 /data/training_datasets/<file>.jsonl
'
```

### Prepared gate has empty val split

If prepare is ok but gate fails with:

```text
split 'val' has 0 sample(s)
```

then there are too few rollout groups. Collect more episodes.

### JSON piping fails with `Expecting value`

This usually means a wrapper printed non-JSON to stdout or produced no stdout. For `--json`, stdout should be JSON and human status should go to stderr. Recent wrappers were updated to follow this convention; if a new wrapper regresses, patch the wrapper rather than removing `python -m json.tool`.

## 10. Recommended next section

Start the next session with one of these two paths.

### Recommended path A: finish data pipeline smoke with 10 episodes

1. Collect `flat_walk_varied_speed_v1_10ep.jsonl`.
2. Export context rows.
3. Prepare train/val/test.
4. Gate prepared dataset.
5. Fix any operational issues found.

This path validates that the whole BC data pipeline actually works end to end.

### Recommended path B: implement M9I training-ready manifest/report

After the prepared gate passes, add a training-ready manifest/report that bundles:

- prepared manifest path
- contract path
- scenario coverage gate result
- prepared dataset gate result
- file hashes
- recommended train command placeholder

Do not implement full BC architecture changes until the training-ready report is reliable.

## 11. Do / do not for the next assistant

Do:

- Continue Soridormi only.
- Prefer small incremental patches.
- Keep MuJoCo-first validation.
- Keep dataset commands Docker-aware.
- Treat empty JSONL files as upstream failure signals.
- Use 10+ independent rollout groups for real train/val/test split validation.

Do not:

- Start a separate sim server for `collect_random_teacher_dataset.sh` by default.
- Resume hardware bridge work.
- Feed raw natural language into low-level policy inputs.
- Train BC before validating the prepared dataset gate.
- Count a 2-episode split with empty val as training-ready.
