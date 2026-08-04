# CLAUDE.md

Project-local instructions for coding assistants working on Soridormi.

## Project identity

Soridormi is a reusable sim-to-real humanoid robotics stack for Open Duck Mini
v2. It preserves clean runtime/API/backend contracts so the same body runtime
can run in MuJoCo, support model replacement and training, and later transfer to
qualified hardware.

Soridormi is the robot cerebellum. Chromie is the separate cognitive/social
brain in `TimeTreker/chromie.git` on `main`.

## Current direction

Read `docs/STATUS.md`. This file contains durable rules only.

Policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Command-conditioned policy input:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps) -> action_14d
```

Do not package a richer context model unless runtime produces the exact declared
features and ordering.

## Core rules

- Focus on Soridormi unless the user explicitly asks for another project.
- Keep MuJoCo-first validation before hardware.
- Preserve official baseline, replay, comparison, and parity scripts.
- Preserve Docker host wrappers.
- Do not feed raw language or raw perception into the low-level action policy.
- Do not invent targets, body state, capability, or completion results.
- Project `safe_idle` and active motion from the live body runtime.
- Preserve one Cognitive Core and three Chromie coordination lanes: Social-
  Attention Proposal, Speaking Execution, and Activity Execution.
- Keep speech outside Soridormi; use `soridormi.activity.*` for compatible
  physical concurrency with declared resources and one final motor command.
- Use semantic issue names rather than numbered project sequences.
- Hardware work defaults to read-only or dry-run unless actuator commands are
  explicitly requested and qualified.

## Important docs

```text
README.md
docs/STATUS.md
docs/DOCUMENTATION_GOVERNANCE.md
docs/README.md
docs/PROJECT_SOP.md
docs/architecture.md
docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md
docs/SORIDORMI_BODY_CONCURRENCY.md
docs/PATCH_DELIVERY_AND_VALIDATION.md
docs/SORIDORMI_TARGET_AND_ROADMAP.md
docs/SORIDORMI_EXECUTION_ROADMAP.md
docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md
docs/SORIDORMI_BC_TRAINING_CONTRACT.md
docs/SORIDORMI_CONTEXT_DATA_PIPELINE.md
```

## Validation

```bash
python scripts/validate_repository_governance.py
./scripts/validate_body_concurrency.sh
pytest -q
python -m compileall -q src
```

Live simulator tests use an explicit MuJoCo backend and profile. Random teacher
collection owns its temporary simulator lifecycle and must not be paired with a
second simulator server.

## Patch style

Deliver plain git patches with both integrity and functional validation. Rebuild
Docker only when dependencies, Dockerfiles, system packages, or base images
change.
