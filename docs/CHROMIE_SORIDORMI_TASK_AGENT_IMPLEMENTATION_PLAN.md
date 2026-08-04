# Chromie and Soridormi task-agent implementation

This document describes the implemented contract and remaining semantic work.
Current status lives in `docs/STATUS.md`.

## Boundary contract

Chromie owns user language, clarification, confirmation, memory, global task
state, provider selection, and user-facing reporting.

Soridormi owns robot state, embodied readiness, body-task interpretation,
bounded skill lowering, physical-resource arbitration, body-command
composition, safety, execution authority, monitoring, recovery, and refusal.

Chromie has one Cognitive Core and three coordination lanes. Goal meaning and
capability selection remain in the Cognitive Core/planner; the Activity lane
only executes and monitors selected provider work.

## Task-level MCP surface

```text
soridormi.task.get_capabilities
soridormi.task.preview
soridormi.task.submit
soridormi.task.status
soridormi.task.events
soridormi.task.cancel
```

It provides:

- versioned readiness projected from the robot capability manifest;
- recursive rejection of raw low-level control fields;
- structured task lifecycle and Soridormi-owned body-task graph;
- no-motion skill and skill-sequence compilation;
- blocked subsystem and refusal reasons;
- routing hints for Chromie's global graph;
- cursor-based events;
- retry-safe `client_task_ref`;
- timeout expiry;
- explicit `no_motion=true` and `raw_control_allowed=false`.

## Execution boundary

Task preview and submit do not move the robot. A successful skill dry run proves
only that Soridormi can validate and compile the structured request.

Atomic physical behavior uses the named-skill path. Exact compatible body
members that must overlap use the body-activity path:

```text
activity.get_capabilities
  -> activity.create_plan
  -> confirmation when required
  -> safety monitoring during activity.execute_plan
  -> activity.status or activity.cancel
  -> robot.get_status and per-member outcome reconciliation
```

The activity API is effectful in the runtime-backed MuJoCo adapter. It is not a
semantic task executor: the authoritative planner must first select exact body
skills. Speech or singing remains in Chromie's peer Speaking lane.

A future rich task executor may call the skill or activity path internally only
after monitored execution, cancellation, recovery, and completion contracts
are qualified.

## Embodied task schema

A structured task may include:

- task type and retry-safe client reference;
- trace summary;
- structured parameters;
- task and environment context;
- safety constraints;
- timeout and cancellation policy.

Natural-language summaries are trace metadata, not low-level control input.

## Body task lifecycle

```text
accepted
  -> resolving
  -> planning
  -> executing
  -> monitoring
  -> recovering
  -> completed | failed | cancelled | refused
```

The current task API uses this lifecycle as a no-motion contract and monitoring
surface. Physical task execution is not yet implied.

## Skill-backed dry-run compilation

Simple semantic tasks may compile into existing named skills. Structured
sequences may compile into bounded skill sequences. Navigation, perception,
manipulation, and unsafe physical requests remain blocked or refused.

## Information integrity

Soridormi fails closed when a structured task lacks required body information.
It does not invent a target, pose, route, or capability.

Body-wide `safe_idle` is supplied by the live runtime state to task payloads.
The task store cannot infer it merely from its own no-motion lifecycle.

## Paired acceptance coverage

Acceptance cases cover stop redirects, semantic walking, structured gesture
sequences, cross-agent speech/motion coordination, target approach refusal,
navigation refusal, object-delivery refusal, unsafe-request refusal, event
polling, retry identity, timeout, and safe-idle reporting.

## Remaining semantic issues

### Paired contract qualification

Run end-to-end Chromie/Soridormi tests for discovery, proposal metadata,
preview, submit, polling, cancellation, timeout, failure wording, source
revision, and no false physical-completion claims.

### Chromie coordinated-group integration

Implement the companion Chromie coordinator for Social-Attention proposals,
peer Speaking and Activity execution, shared `coordination_id`, interaction
cancellation, and Goal outcome reconciliation.

### Rich-task monitored executor

Introduce a Soridormi-owned executor only for task types whose sensing and
planning dependencies are qualified. It may lower exact subwork through the
existing skill or body-activity paths while retaining task identity, event
ordering, interruption, safe hold, and post-action evidence.

### Sensing and planning expansion

Add perception, navigation, and manipulation only as independently declared
subsystems with schemas, tests, simulator evidence, and refusal conditions.

## Validation

```bash
./scripts/validate_task_agent_contract.sh
python scripts/validate_repository_governance.py
```
