# Soridormi docs index

This directory should hold project contracts, active runbooks, and durable
architecture notes. Generated reports belong under `artifacts/` and should not
be committed.

## Core project docs

- `README.md`: project overview, quick start, common host commands.
- `AGENTS.md`: repository-local rules for coding agents.
- `CLAUDE.md`: compact assistant guidance for current project direction.
- `LLM_CONTEXT.md`: new-session handoff for current Soridormi work.
- `docs/PROJECT_SOP.md`: project backbone and validation philosophy.
- `docs/architecture.md`: runtime/API/simulator architecture.
- `docs/PATCH_DELIVERY_AND_VALIDATION.md`: patch and validation expectations.
- `docs/SORIDORMI_TARGET_AND_ROADMAP.md`: cerebellum target, Chromie brain
  boundary, milestone direction, and current policy evidence.
- `docs/SORIDORMI_EXECUTION_ROADMAP.md`: gated milestone sequence, acceptance
  criteria, major risks, and immediate execution plan.
- `docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md`: agreed Chromie brain
  and Soridormi body-agent boundary.
- `docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md`: staged plan for
  adding Soridormi task-level MCP APIs, embodied task schemas, and the internal
  task state machine.
- `task_acceptance_cases/mcp_task_acceptance.yaml`: no-motion acceptance cases
  for Chromie-facing Soridormi task submissions and fail-closed embodied goals.
- `configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`:
  Soridormi-owned readiness table for task-level MCP capabilities.
- `scripts/validate_m11_task_agent_contract.sh`: one-command M11A gate for the
  no-motion Chromie/Soridormi task-agent contract, task graph surface, task
  capability manifest, and acceptance cases.
- `scripts/demo_task_mcp_contract.sh`: local no-motion demo for the
  Chromie-to-Soridormi task MCP boundary.
- `docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md`: future
  neural parameter-adaptive MPC/WBC controller design; NN estimates model
  error while MPC/WBC enforces physics.
- `scripts/build_m10_evidence_package.sh`: packages M10 clearance readiness,
  follow-camera planning, and visual review evidence.
- `scripts/compare_m10_teacher_suite.sh`: compares the candidate and official
  teacher scenario-suite summaries after visual review.

## Active locomotion/data docs

- `docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md`
- `docs/SORIDORMI_BC_TRAINING_CONTRACT.md`
- `docs/SORIDORMI_DATA_PIPELINE_M9.md`: source-of-truth M9 command order,
  including the M9I training-ready bundle.
- `docs/SORIDORMI_SCENARIO_CURRICULUM.md`
- `docs/SORIDORMI_DATASET_COVERAGE.md`
- `docs/SORIDORMI_DATASET_SCENARIO_GATE.md`
- `docs/SORIDORMI_CONTEXT_BC_DATASET_EXPORT.md`
- `docs/SORIDORMI_CONTEXT_BC_DATASET_PREPARE.md`
- `docs/SORIDORMI_CONTEXT_BC_PREPARED_GATE.md`

## Skill and interaction docs

- `docs/SORIDORMI_SKILL_TAXONOMY.md`
- `docs/SORIDORMI_SKILL_EXECUTION.md`
- `docs/SORIDORMI_SCRIPTED_SOCIAL_SKILLS.md`
- `docs/SORIDORMI_LOOK_TARGET_PROVIDER.md`

## Historical docs

Milestone-specific M2-M6, ONNX parity, tuning, and old next-session prompt docs
are historical references. Prefer updating the core docs above instead of
adding another one-off status file.
