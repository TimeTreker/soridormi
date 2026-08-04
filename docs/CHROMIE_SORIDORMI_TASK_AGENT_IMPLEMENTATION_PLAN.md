# Chromie and Soridormi task-agent implementation

This document describes the implemented contract and remaining semantic work.
Current status lives in `docs/STATUS.md`.

## Boundary contract

Chromie owns user language, clarification, confirmation, memory, global task
state, provider selection, and user-facing reporting.

Soridormi owns robot state, embodied readiness, body-task interpretation,
bounded skill lowering, safety, execution authority, monitoring, recovery, and
refusal.

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

Physical behavior uses:

```text
skill.list
  -> skill.create_plan
  -> safety monitoring
  -> skill.execute_plan
  -> robot.get_status safe-idle confirmation
```

A future task executor may call that path internally only after monitored
execution, cancellation, recovery, and completion contracts are qualified.

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

### Skill-backed monitored task executor

Introduce a Soridormi-owned executor for a minimal set of already-qualified
skills. Retain task identity, event ordering, active body state, interruption,
safe hold, and post-action evidence.

### Sensing and planning expansion

Add perception, navigation, and manipulation only as independently declared
subsystems with schemas, tests, simulator evidence, and refusal conditions.

## Validation

```bash
./scripts/validate_task_agent_contract.sh
python scripts/validate_repository_governance.py
```
