# LLM_CONTEXT.md

Compact handoff for starting a new LLM session on Soridormi.

## Project Summary

Soridormi is a sim-to-real humanoid robot stack for Open Duck Mini v2. It
separates runtime, simulator, and shared API so one policy runtime can talk to
MuJoCo now and hardware later. The current project direction is
scenario-aware, context-conditioned locomotion data and behavior cloning in
MuJoCo.

Soridormi is the robot cerebellum/body runtime. Chromie is the robot brain in
`TimeTreker/chromie.git` on `main`. Chromie talks with people, understands
intent, plans high-level behavior, and chooses skills. Soridormi validates and
executes body skills safely in MuJoCo or hardware.

## Current Focus

Active direction: M9 context BC data pipeline.

Low-level policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Do not pass raw natural language into the low-level policy. A planner or skill
router may choose structured fields such as `skill_id`, velocity command,
terrain type, or obstacle context.

Current offline Stage 1 trainer input mode:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
```

Use:

```bash
./scripts/train_behavior_clone.sh PREPARED_CONTEXT_DATASET --input-mode context_stage1_command
./scripts/train_neural_behavior_clone.sh PREPARED_CONTEXT_DATASET --input-mode context_stage1_command --skip-onnx --no-profile
```

Context-mode neural training is checkpoint-only until runtime context features
are wired into policy execution.

## Read First

```text
README.md
docs/README.md
docs/PROJECT_SOP.md
docs/PATCH_DELIVERY_AND_VALIDATION.md
docs/SORIDORMI_TARGET_AND_ROADMAP.md
docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md
docs/SORIDORMI_BC_TRAINING_CONTRACT.md
docs/SORIDORMI_DATA_PIPELINE_M9.md
docs/SORIDORMI_SCENARIO_CURRICULUM.md
```

## Simulator Ownership

External-sim tools need a separately running MuJoCo server:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Visual inspection:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

`collect_random_teacher_dataset.sh` is different: it owns its temporary MuJoCo
server. Do not start a second sim server for it. Pass `--viewer` and usually
`--follow-camera` to the collector itself when needed.

## Current M9 Pipeline

1. Collect scenario-aware teacher rows:

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 10 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --reset-attempts 10 \
  --reset-retry-sleep 0.5 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1_10ep.jsonl \
  --json
```

2. Gate/report/export/prepare:

```bash
./scripts/report_dataset_coverage.sh RAW.jsonl --json
./scripts/gate_dataset_scenario_coverage.sh RAW.jsonl --require-scenario flat_walk_varied_speed_v1 --json
./scripts/export_context_bc_dataset.sh RAW.jsonl --output CONTEXT.jsonl --json
./scripts/validate_bc_training_contract.sh --sample-jsonl CONTEXT.jsonl --json
./scripts/prepare_context_bc_dataset.sh CONTEXT.jsonl --output-dir PREPARED_DIR --json
./scripts/gate_context_bc_prepared_dataset.sh PREPARED_DIR/prepared_manifest.json --require-scenario flat_walk_varied_speed_v1 --json
```

3. Train offline BC smoke models:

```bash
./scripts/train_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_stage1_command --json
./scripts/train_neural_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_stage1_command --skip-onnx --no-profile --json
```

## Validation Policy

Preferred validation:

```bash
pytest -q
python -m compileall -q src
```

For JSON-producing shell wrappers, stdout must stay machine-readable when
`--json`; Docker/Compose/CUDA status belongs on stderr.

## Boundaries

- MuJoCo before hardware.
- Chromie is brain; Soridormi is cerebellum/body runtime.
- Chromie calls structured skills/context, never raw joint actions or low-level
  `action_14d` policy outputs.
- Hardware commands require explicit user intent; otherwise dry-run/read-only.
- BC copies the teacher distribution. Do not claim it improves stride,
  clearance, obstacles, or recovery beyond teacher behavior without rollout
  evidence.
- Generated reports belong under `artifacts/` and should not be committed.
