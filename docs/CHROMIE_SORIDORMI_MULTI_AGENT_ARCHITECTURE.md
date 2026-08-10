# Chromie and Soridormi multi-agent architecture

This document records the agreed brain/body split between Chromie and
Soridormi. Both systems are agent-like, and both may use an orchestrator or DAG
engine, but they operate at different scopes.

For the staged implementation plan, see
`docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md`.

## Core agreement

Chromie has one authoritative Cognitive Core with three concurrent coordination
lanes:

```text
Social-Attention Proposal Lane
Speaking Execution Lane
Activity Execution Lane
```

The lanes are not independent minds. The Cognitive Core owns user meaning,
Goal Association, Goal lifecycle, planning, personality, and authored
communication. Social Attention proposes only. Speaking delivers authored
communication. Activity executes and monitors provider work. One Cognitive
Runtime Coordinator validates timing, confirmation, cancellation, and outcome
reconciliation across the execution lanes.

Soridormi is the embodied robot agent. It owns robot state, body capability
availability, embodied task planning, sensing/localization hooks, local routing,
gait and skill selection, safety monitoring, safe hold, emergency stop,
recovery, MuJoCo execution, and future hardware execution.

The boundary is MCP:

```text
Chromie authoritative plan and coordinated execution group
  -> Speaking Execution Lane for speech or singing
  -> Activity Execution Lane for provider work
       -> Soridormi task, skill, or concurrent body-activity request
       -> bounded body skills, resources, command composition, recovery
  -> structured outcomes reconciled by the Cognitive Runtime Coordinator
```

Chromie should send what should be achieved and why. Soridormi decides whether
and how the robot body can safely do it.

For named-skill planning, Chromie may attach a `chromie_intent` object with
`execution_mode=proposed`, `execution_semantics=proposal_from_chromie`, and
`requires_runtime_validation=true`. Soridormi validates this as advisory
provenance only. It rejects executable semantics, low-level controls, and
physical coordinates in that metadata, then independently validates and plans
the named skill through its owned runtime boundary.


### Stable semantic authority, dynamic capability boundary

Chromie and Soridormi may both contain planners without duplicating authority because
they plan at different scopes. Chromie owns the human-facing Goal and the global plan
across the capabilities currently advertised by Soridormi and peer providers.
Soridormi owns the implementation plan inside each Soridormi capability that Chromie
selects.

The decomposition boundary is deliberately **not fixed**. If Soridormi can guarantee a
complete physical workflow and advertises a composite capability, Chromie treats that
capability as one atomic global-plan leaf. If Soridormi exposes only smaller resource
capabilities, Chromie may compose those advertised leaves with each other and with
capabilities from other providers. Soridormi may expose both levels at once.

Soridormi is free to use deterministic rules, behavior trees, local planners, motion
planners, learned policies, recovery planners, or other internal hierarchy. What it
may not do is reinterpret the human Goal or plan across capabilities owned by other
providers. The live capability catalog is therefore the dynamic SoC/ECU-style boundary:
semantic authority stays with Chromie while provider capability granularity may evolve.

## Coordination and DAG scopes

The lane model and the DAG model are complementary. The Cognitive Core and
planner author one global plan; the coordinator schedules peer execution lanes.


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
soridormi.activity.*    exact concurrent body-skill groups
soridormi.task.*        no-motion embodied task contract/status/events/cancel
```

`soridormi.skill.*` is appropriate for atomic body behaviors such as
`nod_yes`, `look_at_person`, `turn_in_place`, or explicit low-level test cases.

`soridormi.activity.*` is appropriate when the authoritative planner has
selected exact compatible body skills that must overlap. It supports one
primary locomotion member, compatible bounded head/gaze overlays, and
independent visual expressions. Speech is not an activity member; Chromie runs
it as a peer Speaking lane under the same `coordination_id`.

`soridormi.task.*` is implemented as a contract-first, no-motion surface for
rich embodied requests such as navigation, approach, inspection, gesture,
recovery, or unsupported object delivery. Supported requests may compile to
named-skill dry runs; that is not physical task execution. A future monitored
task executor may use the validated skill path internally only after its own
qualification.


### Coordinated walking, gaze, blinking, and speech

```text
Cognitive Core:
  understands "walk toward me while singing and blinking"

Planner:
  selects chromie.vocal.perform
  selects soridormi.walk_velocity
  selects soridormi.look_at_person
  selects soridormi.blink_eyes

Coordinator:
  starts the Speaking and Activity lanes with one coordination_id

Soridormi:
  validates body resource compatibility
  composes locomotion and bounded head overlay into one motor command
  runs eye expression on its independent output
  preempts physical behavior whenever safety requires it
```

Social Attention may propose gaze or blinking, but the Cognitive Core and
planner decide whether the proposal becomes an exact provider request.

## Examples

### Concrete body command

```text
walk_velocity(vx_mps=0.2, duration_s=10)
```

This is already a concrete body command. It can be useful for tests, simple
explicit requests, or lower-level skill execution, but it should not be the main
interface for rich user goals.

### Human resource goal

```text
Can you bring me some water?
```

Chromie represents this as one `AcquireAndDeliverResource` Goal whose
`resource.kind=physical_object`. It then plans against the current catalog. If
Soridormi advertises a complete `acquire_and_deliver_resource` capability, Chromie
may use it as one plan leaf and Soridormi owns the complete provider-local workflow.
If only smaller capabilities such as `acquire_resource` and `deliver_resource` are
advertised, Chromie may compose them in order. Chromie never invents hidden Soridormi
substeps that are absent from the catalog.

The current Open Duck Mini v2 still has no qualified hardware manipulator/gripper
stack. The simulated capability therefore proves the architecture and evidence
contract only; it is not hardware qualification and must not become executable on
hardware merely because the simulation mock exists.

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
- Goal meaning, Goal Association, lifecycle, and global task DAG state;
- social-attention proposals and their acceptance or suppression;
- speaking and activity coordination groups;
- clarifications and confirmations;
- user preferences and memories;
- cross-capability orchestration.

Soridormi maintains:

- robot state and safe/unsafe state;
- embodied task state and substeps;
- local target/route/motion context;
- selected gait, skill, controller, and fallback;
- execution telemetry, progress, blocked state, and recovery state;
- physical resource claims, per-member activity state, final motor-command
  composition, and safety preemption.

Body-wide state remains runtime-owned. Task and capability payloads project
live `safe_idle`, active-motion, and emergency-stop state; task-local
lifecycle does not manufacture those facts.

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

See `docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md` and `docs/SORIDORMI_BODY_CONCURRENCY.md` for the detailed lane and provider contracts.
