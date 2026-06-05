# CLAUDE.md

Project-local instructions for Claude Code or any coding assistant working on
Soridormi.

## Project Identity

Soridormi is a reusable sim-to-real humanoid robotics stack for Open Duck Mini
v2. The goal is not a one-off walking demo. Soridormi should preserve clean
runtime/API/backend contracts so the same policy runtime can run in MuJoCo,
support model replacement/training, and eventually transfer to real hardware.

Official Open Duck code is the behavioral reference. Soridormi should reproduce
official behavior through its own runtime contracts, logging, profiles, and
Docker host workflows.

## Current Direction

Current active direction: M9 context-aware locomotion data and behavior cloning.

Near-term policy contract:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Near-term trainable stage:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps) -> action_14d
```

The Stage 1 context input mode is offline training only until runtime context
plumbing is implemented. Do not package a context-mode policy as runtime ONNX
unless the runtime can provide the same context features.

## Core Rules

- Focus on Soridormi unless the user explicitly asks for Chromie or another
  project.
- Keep MuJoCo-first validation before hardware.
- Do not replace locomotion work with open-loop gait.
- Do not hide failures behind tuning.
- Preserve official baseline, replay, comparison, and parity scripts.
- Preserve Docker host wrapper behavior; users usually run scripts from the
  host, and wrappers should enter the correct Docker service internally.
- If official compatibility needs reference files, fail fast when they are
  missing.
- Do not feed raw natural language or raw perception directly into the low-level
  14D action policy. Convert it to bounded structured context first.
- Hardware work must default to read-only or dry-run validation unless the user
  explicitly asks to send actuator commands.

## Important Docs

Read these before changing direction:

```text
README.md
docs/README.md
docs/PROJECT_SOP.md
docs/PATCH_DELIVERY_AND_VALIDATION.md
docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md
docs/SORIDORMI_BC_TRAINING_CONTRACT.md
docs/SORIDORMI_DATA_PIPELINE_M9.md
```

## Validation

Preferred local validation:

```bash
pytest -q
python -m compileall -q src
```

For live simulator tests, use MuJoCo explicitly:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Optional visual inspection:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Exception: `collect_random_teacher_dataset.sh` owns its temporary MuJoCo
simulator lifecycle. Do not pair it with a second `run_sim_server.sh`; use the
collector's own `--viewer` and usually `--follow-camera` flags.

## Patch Style

The user prefers plain git patch files, not zip archives. Assume downloaded
patches live in `~/Downloads` unless the user says otherwise.

Every patch response must include both patch integrity and functional validation
commands. For docs-only patches, still validate expected sections and Markdown
fences. For code, run relevant tests and compile checks. For sim/training, give
local checks plus live MuJoCo validation commands.
