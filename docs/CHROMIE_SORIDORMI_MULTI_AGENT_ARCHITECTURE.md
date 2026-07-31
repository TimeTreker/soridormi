# Chromie and Soridormi multi-agent architecture

This document records the agreed brain/body split between Chromie and
Soridormi. Both systems are agent-like, and both may use an orchestrator or DAG
engine, but they operate at different scopes.

For the staged implementation plan, see
`docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md`.

## Core agreement

Chromie is the cognitive and social agent. It understands the user, keeps
conversation and task context, asks clarifying questions, manages confirmation,
uses memory/search/speech tools, and builds global task DAGs.

Soridormi is the embodied robot agent. It owns robot state, body capability
availability, embodied task planning, sensing/localization hooks, local routing,
gait and skill selection, safety monitoring, safe hold, emergency stop,
recovery, MuJoCo execution, and future hardware execution.

The boundary is MCP:

```text
Chromie global task DAG
  -> Soridormi MCP embodied task or skill request
  -> Soridormi embodied task DAG / state machine
  -> bounded skills, trajectories, gait, controller, recovery
  -> structured status/events back to Chromie
```

Chromie should send what should be achieved and why. Soridormi decides whether
and how the robot body can safely do it.

For named-skill planning, Chromie may attach a `chromie_intent` object with
`execution_mode=proposed`, `execution_semantics=proposal_from_chromie`, and
`requires_runtime_validation=true`. Soridormi validates this as advisory
provenance only. It rejects executable semantics, low-level controls, and
physical coordinates in that metadata, then independently validates and plans
the named skill through its owned runtime boundary.

## Two DAG scopes

### Chromie global DAG

Chromie owns the human-facing and multi-capability DAG. Example for "let's go
to a nearby grocery":

```text
understand request
  -> search candidate groceries
  -> ask user to choose
  -> wait for confirmation
  -> submit Soridormi navigate_to task
  -> monitor Soridormi task status/events
  -> speak progress or completion
```

This DAG may call many MCP capability providers: memory, search, speech, maps,
vision, Soridormi, and user confirmation.

### Soridormi embodied DAG

Soridormi owns the body-facing DAG or state machine. For the confirmed
navigation goal above:

```text
check robot state
  -> resolve target pose or bearing
  -> localize robot
  -> plan route or short local segment
  -> check obstacles and traversability
  -> choose gait/skill
  -> execute monitored segment
  -> replan, stop, recover, or continue
  -> report completed/blocked/failed/safe_idle
```

This DAG is safety-critical and must remain MuJoCo-first before hardware.

## API levels

Soridormi should expose multiple MCP levels, not a single skill endpoint:

```text
soridormi.robot.*       read status, mode, battery, active task, safe_idle
soridormi.safety.*      monitor, stop, cancel, emergency stop
soridormi.skill.*       atomic body skills and skill plans
soridormi.task.*        future embodied task submission/status/events/cancel
```

`soridormi.skill.*` is appropriate for atomic body behaviors such as
`nod_yes`, `look_at_person`, `turn_in_place`, or explicit low-level test cases.

`soridormi.task.*` is the intended future surface for rich embodied requests
such as `navigate_to`, `approach_person`, `inspect_target`, `perform_dance`,
`recover`, or unsupported tasks such as `deliver_object`.

## Examples

### Concrete body command

```text
walk_velocity(vx_mps=0.2, duration_s=10)
```

This is already a concrete body command. It can be useful for tests, simple
explicit requests, or lower-level skill execution, but it should not be the main
interface for rich user goals.

### Human goal

```text
Can you bring me some water?
```

Chromie may understand this as a delivery goal, but Soridormi must check
embodied feasibility. On the current Open Duck Mini v2, Soridormi should refuse
object delivery because the robot has no supported manipulator, gripper, carry,
or handoff capability. It may offer alternatives such as looking toward the
object, navigating near it after target resolution, or reporting that the task
is unsupported.

### Destination goal

```text
Walk forward to the house.
```

This is not a velocity command. Soridormi must refuse it until target
resolution, localization, route planning, local obstacle checking, and bounded
local trajectory planning exist. See
`docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md`.

### Unsafe physical task

```text
Fight that person.
```

This must be refused as unsafe. Chromie may explain the refusal. Soridormi must
not lower it into body motion.

## Task context ownership

Chromie maintains:

- user intent and conversation history;
- parent goals and task DAG state;
- clarifications and confirmations;
- user preferences and memories;
- cross-capability orchestration.

Soridormi maintains:

- robot state and safe/unsafe state;
- embodied task state and substeps;
- local target/route/motion context;
- selected gait, skill, controller, and fallback;
- execution telemetry, progress, blocked state, and recovery state.

## Promotion rule

No new embodied capability should become executable merely because Chromie can
ask for it. Soridormi promotion requires:

- a declared task/skill contract;
- bounded parameters and refusal conditions;
- MuJoCo validation;
- safe-idle and cancellation evidence;
- explicit unsupported status for missing hardware capabilities;
- no exposure of raw natural language, raw perception, joint targets, motor
  commands, torque commands, or `action_14d` outputs to Chromie.
