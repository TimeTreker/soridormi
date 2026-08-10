# Soridormi documentation

This index defines which documents are authoritative. A document must not
become a second status database.

## Authority order

1. Runtime contracts and tests are authoritative for executable behavior.
2. `docs/STATUS.md` is the only current-state and current-blocker summary.
3. `docs/PROJECT_SOP.md` defines the durable engineering and promotion loop.
4. `docs/architecture.md` and boundary documents define ownership.
5. Feature-specific contracts and runbooks define their own surfaces.
6. Git history contains replaced status reports and old planning diaries.

When documents disagree, do not add another summary. Correct the highest-
authority durable document and `docs/STATUS.md`, then add or update a guard.

## Start here

- `README.md` — repository overview and operator quick start.
- `LLM_CONTEXT.md` — compact assistant entry point.
- `docs/STATUS.md` — verified current capabilities, blockers, and work order.
- `docs/PROJECT_SOP.md` — baseline-to-simulation-to-hardware engineering loop.
- `docs/architecture.md` — runtime, simulator, API, MCP, and tooling boundaries.
- `docs/DOCUMENTATION_GOVERNANCE.md` — source-of-truth and naming policy.
- `docs/PATCH_DELIVERY_AND_VALIDATION.md` — patch delivery expectations.

## Brain/body and MCP contracts

- `docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md`
- `docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md`
- `docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md`
- `docs/SORIDORMI_BODY_CONCURRENCY.md`
- `docs/SORIDORMI_MCP_SERVER.md`
- `docs/mcp_capability_manifest.md`
- `docs/mcp_dag_integration.md`
- `docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md`
- `docs/SORIDORMI_RESOURCE_ACQUISITION_DELIVERY.md`
- `docs/SORIDORMI_TEXT_INPUT_ACCEPTANCE.md`
- `configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`
- `task_acceptance_cases/mcp_task_acceptance.yaml`

The task API is contract-first and no-motion. Named skill, bounded motion, and
exact concurrent body-activity execution are separate runtime paths and remain
Soridormi-owned. Speech and singing remain Chromie-owned peer execution.

## Locomotion, policy, and data contracts

- `docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md`
- `docs/SORIDORMI_BC_TRAINING_CONTRACT.md`
- `docs/SORIDORMI_CONTEXT_DATA_PIPELINE.md`
- `docs/SORIDORMI_SCENARIO_CURRICULUM.md`
- `docs/SORIDORMI_DATASET_COVERAGE.md`
- `docs/SORIDORMI_DATASET_SCENARIO_GATE.md`
- `docs/SORIDORMI_CONTEXT_BC_DATASET_EXPORT.md`
- `docs/SORIDORMI_CONTEXT_BC_DATASET_PREPARE.md`
- `docs/SORIDORMI_CONTEXT_BC_PREPARED_GATE.md`
- `docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md`
- `docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md`
- `docs/MODEL_REPLACEMENT_INTERFACE.md`

## Skill and interaction contracts

- `docs/SORIDORMI_SKILL_TAXONOMY.md`
- `docs/SORIDORMI_SKILL_EXECUTION.md`
- `docs/SORIDORMI_SCRIPTED_SOCIAL_SKILLS.md`
- `docs/SORIDORMI_LOOK_TARGET_PROVIDER.md`
- `docs/SORIDORMI_SCRIPTED_SOCIAL_ACCEPTANCE.md`

## Deployment and operator runbooks

- `docs/SORIDORMI_DEPLOYMENT.md`
- `docs/CHROMIE_SORIDORMI_EFFECT_CHECK.md`
- `docs/host_setup.md`
- `docs/troubleshooting.md`

## Evidence and history

Generated datasets, reports, traces, screenshots, and candidate outputs belong
under ignored `data/` and `artifacts/`. Durable conclusions may be summarized
once in `docs/STATUS.md`; raw metrics must not be copied into multiple roadmaps.

Replaced handoffs and obsolete sequence-labelled documents are removed from the
working tree. Git history remains the historical archive.

## Clearance qualification tools

```bash
./scripts/report_clearance_candidate_history.sh
./scripts/validate_clearance_engineering_process.sh
```

These commands summarize retained evidence and validate the offline
clearance process. They do not train, launch MuJoCo, or authorize hardware.
