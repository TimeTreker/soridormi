# AGENTS.md

Repository-local guidance for coding agents working on Soridormi.

## Primary objective

Build a reusable sim-to-real engineering runtime, not a one-off demo. Official
Open Duck code is the reference; Soridormi should reproduce it through clean
runtime/API/backend contracts and make model replacement, training, and
hardware transfer auditable.

Soridormi is the robot cerebellum/body runtime. Chromie, in
`TimeTreker/chromie.git` on `main`, is the robot brain that handles conversation,
memory, high-level planning, and skill choice.

## Current work authority

Read `docs/STATUS.md`. Do not encode temporary project sequence labels in this
file, runtime payloads, tests, filenames, or schemas.

Durable policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Current command-conditioned trainable contract:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps) -> action_14d
```

## Guardrails

- Do not replace the current task with open-loop gait.
- Do not jump to hardware before MuJoCo validation.
- Do not hide failures behind tuning.
- Do not remove the official baseline, replay, or parity scripts.
- Do not feed raw natural language or raw perception directly into the low-level
  14D action policy.
- Do not let Chromie or any planner send raw joint actions, motor commands,
  physical coordinates, or low-level `action_14d` outputs.
- Runtime body state is authoritative for `safe_idle`, active motion, and safety.
- Never invent a target, pose, capability, or completion result.
- Preserve Docker host wrapper behavior; host scripts should enter the correct
  Docker service internally when package imports are required.
- If official compatibility needs reference files, fail fast when they are
  missing.
- Use semantic issue names, not numbered project sequences.

## Patch style

The user prefers plain git patch files. Assume downloaded patches live in
`~/Downloads` unless stated otherwise.

```bash
cd /path/to/soridormi
git apply --check ~/Downloads/<patch_name>.patch
git apply ~/Downloads/<patch_name>.patch
```

Every patch response includes functional validation, not only patch integrity:

- docs-only: governance check and Markdown checks;
- code: focused `pytest`, compile checks, and CLI smoke tests;
- sim/training: local checks plus explicit MuJoCo commands;
- hardware: read-only/dry-run by default and an explicit statement about whether
  actuator commands were sent.

Include tests when behavior changes. Include docs when a capability contract
changes. Mention whether a Docker rebuild is needed.

See `docs/PATCH_DELIVERY_AND_VALIDATION.md`.

## Validation expectations

```bash
python scripts/validate_repository_governance.py
pytest -q
python -m compileall -q src
```

For policy behavior:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/evaluate_scenario_rollout.sh --scenario flat_walk_varied_speed_v1 --backend mujoco --profile open_duck_forward
```

For context data and training:

```bash
./scripts/collect_random_teacher_dataset.sh --backend mujoco --scenario flat_walk_varied_speed_v1 --profile open_duck_forward --json
./scripts/report_dataset_coverage.sh RAW.jsonl --json
./scripts/gate_dataset_scenario_coverage.sh RAW.jsonl --require-scenario flat_walk_varied_speed_v1 --json
./scripts/export_context_bc_dataset.sh RAW.jsonl --output CONTEXT.jsonl --json
./scripts/validate_bc_training_contract.sh --sample-jsonl CONTEXT.jsonl --json
./scripts/prepare_context_bc_dataset.sh CONTEXT.jsonl --output-dir PREPARED_DIR --json
./scripts/train_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_command_v1 --json
./scripts/train_neural_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_command_v1 --profile-name context_command_candidate --force-profile --json
```
