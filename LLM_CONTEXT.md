# Soridormi LLM context

This is a compact entry point, not a status diary.

## Identity

Soridormi is the platform execution runtime paired with Chromie. Open Duck Mini
v2 body control and MuJoCo remain its verified foundation, but the approved
target also places platform-facing vocal, media, audio/sensor, and device
execution behind Soridormi.

Chromie is the separate cognitive and social brain. Chromie owns conversation,
memory, Goal meaning, response authorship, vocal mode, clarification,
confirmation, user-level cancellation scope, and interaction orchestration.

```text
Chromie authorizes one immutable platform-neutral execution envelope.
Soridormi Execution Runtime validates, prepares, schedules, executes, monitors,
recovers, and may refuse exact capability members.
Soridormi Platform Provider adapts MuJoCo, physical robot, desktop audio,
sensors, controllers, and drivers.
```

Chromie never supplies raw joints, motors, torques, physical coordinates, audio-
device indexes, SDK objects, or low-level `action_14d` outputs as execution
authority. Soridormi never rewrites Goal meaning, authored content, vocal mode,
or confirmation.

The target is not the current runtime claim. Today Soridormi primarily executes
body capabilities; Chromie still owns TTS/playback and cross-provider
coordination. Check `docs/STATUS.md` before describing anything as implemented.

## Read order

1. `docs/STATUS.md` — the only current-state authority.
2. `docs/README.md` — documentation map and authority hierarchy.
3. `docs/PROJECT_SOP.md` — durable engineering loop.
4. `docs/architecture.md` — execution-runtime and platform-provider boundary.
5. `docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md` — one core, three semantic lanes, and two coordinators.
6. `docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md` — brain/execution/platform ownership split.
7. `docs/SORIDORMI_EXECUTION_ROADMAP.md` — gate-driven execution migration.
8. `docs/SORIDORMI_TARGET_AND_ROADMAP.md` — durable platform-execution target.
9. `docs/SORIDORMI_BODY_CONCURRENCY.md` — body resources and composition.
10. `docs/SORIDORMI_MCP_SERVER.md` — current MCP behavior and deployment.
11. `docs/DOCUMENTATION_GOVERNANCE.md` — rules preventing status drift.
12. `docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md` — future adaptive-control design.

## Non-negotiable rules

- MuJoCo first; hardware remains fail-closed until explicitly qualified.
- Model output is never execution authorization.
- Task preview/submit is a no-motion contract surface.
- Current physical motion uses validated Soridormi skill, motion, or body-
  activity paths.
- Chromie has one Cognitive Core; Social Attention proposes, Speaking describes
  generated vocal outcomes, and Activity describes exact capability work.
- Speaking lane is semantic ownership, not a process boundary. Chromie authors
  content and vocal mode; the approved target executes TTS, singing, and other
  vocal modes through Soridormi.
- Singing and humming are vocal modes. Playing existing music is media Activity.
  Neither may be substituted for the other, and expressive TTS is not proof of
  singing capability.
- Soridormi has two logical containers: an execution runtime and one active
  platform provider. Platform adaptation must not leak into Chromie.
- Chromie Interaction Orchestrator owns user interaction and cancellation
  meaning. Soridormi Execution Coordinator owns provider-local resources,
  preparation, timing, execution, stop, recovery, and evidence.
- Soridormi allows one primary locomotion member plus compatible subtle
  expressions and retains one final motor-command authority.
- Runtime/platform state is authoritative for `safe_idle`, active execution,
  provider health, and safety.
- Never invent a target, pose, coordinate, vocal mode, media item, capability,
  or completion state.
- Offline loss, prepared state, and dry run are not promotion or completion
  evidence; closed-loop evidence is required.
- Generated data and reports stay under ignored `data/` and `artifacts/`.
- Use semantic issue names. Do not introduce numbered project sequence labels,
  numbered implementation stages, or numbered implementation-step headings.
- Update `docs/STATUS.md` with behavior-changing work; do not add another top-
  level handoff or project-status document.

## Validation

```bash
python scripts/validate_repository_governance.py
SORIDORMI_TASK_AGENT_USE_DOCKER=0 ./scripts/validate_task_agent_contract.sh
```
