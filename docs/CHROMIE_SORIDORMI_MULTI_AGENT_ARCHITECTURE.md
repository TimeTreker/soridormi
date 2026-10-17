# Chromie and Soridormi multi-agent architecture

This document records the agreed brain/body split between Chromie and
Soridormi. Both systems are agent-like, and both may use an orchestrator or DAG
engine, but they operate at different scopes.

For the staged implementation plan, see
`docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md`.

## Core agreement

Chromie has one authoritative Cognitive Core with two concurrent execution
lanes:

```text
Vocal Execution Lane
Activity Execution Lane
```

The lanes are not independent minds. The Cognitive Core owns user meaning,
Goal Association, Goal lifecycle, planning, Value Judgment, personality, and
authored communication. Social Cognition (SC) is Chromie's communication
interface and owns all social expression, including Social Attention, which is a
behavior domain rather than a lane. Vocal delivers authored communication.
Activity executes and monitors provider work. One Cognitive Runtime Coordinator
validates timing, confirmation, cancellation, and outcome reconciliation across
the execution lanes.

Soridormi is the embodied robot agent. It owns robot state, body capability
availability, physical target resolution, sensing/localization hooks, local
routing, body-level realization inside a selected skill (gait, trajectory,
pace), safety monitoring, safe hold, emergency stop, recovery, MuJoCo
execution, and future hardware execution.

The boundary is MCP:

```text
Chromie authoritative plan and coordinated execution group
  -> Vocal Execution Lane for speech or singing
  -> Activity Execution Lane for provider work
       -> Soridormi task, skill, or concurrent body-activity request
       -> bounded body skills, resources, command composition, recovery
  -> structured outcomes reconciled by the Cognitive Runtime Coordinator
```

Chromie selects the exact capability and its semantic arguments. Soridormi
decides only body-level how (gait, trajectory, pace within the selected skill)
and whether the body can safely do it; it does not replace a selected skill.
A body that cannot do something safely reports a limitation, not a refusal:
whether Chromie should do something at all is decided by Chromie's Value
Judgment. The task API's terminal `status: refused` (with its `policy_refusal`
event and `report_refusal` action) is this body limitation: the request is
unavailable or not allowed for the body now. Chromie treats it as a limitation,
never as a Value Judgment refusal.

Chromie describes physical targets semantically and in one agreed format:
`target_ref` or recipient `referent_id` `"requester"` for the person who made the
current request, an object by its description, and a place by its cited name.
Soridormi resolves the actual person, object, or place from its own perception
and body state. A target it cannot uniquely resolve returns a typed
`target_resolution` non-completion (for example `not_observed`, `ambiguous`,
`outside_head_range`, or a place `not_resolved`) instead of a default direction
or the nearest keyword match. An explicit requested value Soridormi does not
execute as requested is reported in `value_adjustments`.

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
  -> choose gait and local trajectory
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
it as a peer Vocal lane under the same `coordination_id`.

`soridormi.task.*` is implemented as a contract-first, no-motion surface for
rich embodied requests such as navigation, approach, inspection, gesture,
recovery, or non-mocked object delivery. Supported requests may compile to
named-skill dry runs; that is not physical task execution. A future monitored
task executor may use the validated skill path internally only after its own
qualification, and it will realize a Chromie-selected skill rather than choose
a different one. Object delivery in simulation is live through the named skill
`acquire_and_deliver_resource`, not through `task.submit deliver_object`.


### Coordinated walking, gaze, blinking, and speech

```text
Cognitive Core:
  understands "walk toward me while singing and blinking"

Planner:
  selects chromie.vocal.perform
  selects soridormi.walk_velocity
  selects soridormi.look_at_person(target_ref="requester")

Social Cognition:
  may add an unrequested blink (soridormi.blink_eyes) as optional expression

Coordinator:
  starts the Vocal and Activity lanes with one coordination_id

Soridormi:
  validates body resource compatibility
  resolves the requester's direction from its own scene observation
  composes locomotion and bounded head overlay into one motor command
  runs eye expression on its independent output
  preempts physical behavior whenever safety requires it
```

Requested body actions are Planner-selected Work. Unrequested gaze or blinking is
optional social expression authored by Social Cognition and bound to its act,
outside Goal-owned plan steps; the Planner does not author it.

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

Delivery is live in simulation: Soridormi walks to the observed resource and back
to the requester, mocks pickup and handover, and returns `resource_outcome`
evidence, or a typed `target_resolution` non-completion when the resource,
recipient, or a cited place cannot be resolved. The current Open Duck Mini v2
still has no qualified hardware manipulator/gripper stack. The simulated
capability therefore proves the architecture and evidence contract only; it is
not hardware qualification and must not become executable on hardware merely
because the simulation mock exists.

### Destination goal

```text
Walk forward to the house.
```

This is not a velocity command. Soridormi reports it as unavailable, without
motion, until target resolution, localization, route planning, local obstacle checking, and bounded
local trajectory planning exist. See
`docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md`.

### Unsafe physical task

```text
Fight that person.
```

Chromie understands the request and its Value Judgment refuses it; there is no
pre-semantic route filter. Chromie may explain the refusal. Soridormi must not
lower it into body motion.

## Task context ownership

Chromie maintains:

- user intent and conversation history;
- Goal meaning, Goal Association, lifecycle, and global task DAG state;
- Social Cognition expression choices, including optional Social Attention;
- vocal and activity coordination groups;
- clarifications and confirmations;
- user preferences and memories;
- cross-capability orchestration.

Soridormi maintains:

- robot state and safe/unsafe state;
- embodied task state and substeps;
- physical target resolution and local route/motion context;
- selected gait, controller, and fallback inside the Chromie-selected skill;
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
