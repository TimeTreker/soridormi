# Soridormi LLM context

This is a compact entry point, not a status diary.

## Identity

Soridormi is the robot body/cerebellum runtime for Open Duck Mini v2. It owns
robot state, embodied capability availability, body-skill planning, locomotion,
safety, MuJoCo execution, and future hardware execution.

Chromie is the separate cognitive and social brain. Chromie owns conversation,
memory, clarification, confirmation, high-level goals, and cross-provider
orchestration.

```text
Chromie proposes structured intent.
Soridormi validates, plans, executes, monitors, and may refuse body behavior.
Chromie never supplies raw joints, motors, torques, physical coordinates, or
low-level action_14d outputs as execution authority.
```

## Read order

1. `docs/STATUS.md` — the only current-state authority.
2. `docs/README.md` — documentation map and authority hierarchy.
3. `docs/PROJECT_SOP.md` — durable engineering loop.
4. `docs/architecture.md` — process, package, backend, and MCP boundaries.
5. `docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md` — ownership split.
6. `docs/SORIDORMI_MCP_SERVER.md` — current MCP behavior and deployment.
7. `docs/DOCUMENTATION_GOVERNANCE.md` — rules preventing status drift.
8. `docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md` — future adaptive-control design.

## Non-negotiable rules

- MuJoCo first; hardware remains fail-closed until explicitly qualified.
- Model output is never body authorization.
- Task preview/submit is a no-motion contract surface.
- Physical motion uses validated Soridormi skill/motion execution paths.
- Runtime body state is authoritative for `safe_idle`, active motion, and safety.
- Never invent a target, pose, coordinate, capability, or completion state.
- Offline loss is not a promotion gate; closed-loop evidence is required.
- Generated data and reports stay under ignored `data/` and `artifacts/`.
- Use semantic issue names. Do not introduce numbered project sequence labels,
  numbered implementation stages, or numbered implementation-step headings.
- Update `docs/STATUS.md` with behavior-changing work; do not add another
  top-level handoff or project-status document.

## Validation

```bash
python scripts/validate_repository_governance.py
SORIDORMI_TASK_AGENT_USE_DOCKER=0 ./scripts/validate_task_agent_contract.sh
```
