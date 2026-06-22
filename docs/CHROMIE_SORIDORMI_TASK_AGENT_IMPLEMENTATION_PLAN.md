# Chromie and Soridormi task-agent implementation plan

This document turns the Chromie/Soridormi brain/body agreement into a staged
implementation plan. It should be built one step at a time, with each step
leaving behind a contract, tests, and a clear gate.

Related docs:

- `docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md`
- `docs/SORIDORMI_MCP_SERVER.md`
- `docs/mcp_capability_manifest.md`
- `docs/mcp_dag_integration.md`
- `docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md`

Chromie should keep a companion document with the same filename in the Chromie
repository. That document owns the brain-side work: global task graphs,
confirmation, provider selection, Soridormi task submission, monitoring, and
user-facing reporting.

## Current agreement

Chromie owns the human-facing brain layer:

- understand the user request;
- keep conversation, task, memory, and confirmation context;
- resolve ambiguity with the user;
- build the global task graph;
- call MCP tools and capability providers;
- monitor progress and report back to the user.

Soridormi owns the embodied robot layer:

- expose robot capabilities through MCP;
- accept bounded embodied goals or atomic skills;
- maintain robot state, safety state, and active embodied task state;
- run sensing, planning, gait selection, control, monitoring, and recovery;
- decide whether a task is executable, blocked, unsupported, unsafe, cancelled,
  or complete;
- validate motion in MuJoCo before any hardware path.

Both systems may use an orchestrator or DAG engine, but they operate at
different scopes. Chromie owns the global task DAG. Soridormi owns the
body-facing DAG or state machine.

## Build order

### Step 1 - Freeze the boundary contract

Goal: make the interface decision explicit and hard to accidentally violate.

Deliverables:

- shared wording in Soridormi and Chromie docs;
- MCP naming for robot, safety, skill, and task APIs;
- examples that distinguish concrete skill commands from rich goals;
- explicit refusal rules for unsafe, unsupported, or underspecified embodied
  requests.

Gate:

```text
docs clearly state that Chromie does not send raw joint actions, motor commands,
or low-level policy outputs, and Soridormi does not receive raw natural language
as low-level control input
```

### Step 2 - Add the Soridormi task-level MCP surface

Goal: keep the current `soridormi.skill.*` surface, then add a task-level layer
for richer embodied goals.

Initial API direction:

```text
soridormi.task.get_capabilities
soridormi.task.preview
soridormi.task.submit
soridormi.task.status
soridormi.task.events
soridormi.task.cancel
```

This first implementation can be contract-first and local/mock-backed. It does
not need to solve perception, navigation, manipulation, or full autonomy yet.

Gate:

```text
MCP manifest exposes task-level capability contracts, task submission is bounded
by schema validation, and unsupported task types fail closed
```

M11 implementation note: this surface is now declared as contract-only through
`soridormi.task.get_capabilities`, `soridormi.task.preview`,
`soridormi.task.submit`, `soridormi.task.status`, `soridormi.task.events`, and
`soridormi.task.cancel`. The capabilities tool reports Soridormi-owned embodied
readiness, missing body-runtime subsystems, and external dependencies. Preview
returns Soridormi's interpretation without persistence. Submit records
structured requests and returns `no_motion=true`. It also exposes the first
internal lifecycle skeleton:
accepted tasks move through `accepted -> resolving -> planning`. Simple
supported task types can then compile into existing named skills and complete as
`skill_dry_run`; structured multi-step requests can compile as
`skill_sequence_dry_run`; cross-agent or unsupported tasks remain held at
planning or fail closed.

The readiness table is now a Soridormi-owned artifact at
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`, loaded by
the MCP task service. Add new task types or change readiness there first, then
update executor code only when the runtime actually gains a new execution path.

The task status payload now includes `plan_steps` and `blocked_subsystems`.
These fields let Chromie understand Soridormi's embodied interpretation without
receiving low-level controls. For example, `navigate_to_location` can refuse
with sensing, localization, routing, planning, and control steps marked blocked,
while `turn left then nod twice` can dry-run a two-step skill plan.
It also includes `recommended_next_actions`, which are structured routing hints
for Chromie rather than user-facing text or robot-control commands.

The payload also includes `task_graph`, a derived Soridormi body-task DAG with
stable node IDs, sequence edges, current phase, terminal state, blocked
subsystems, and `raw_control_allowed=false`. Chromie may use this for global
monitoring and reporting, but the graph remains a body-runtime view, not a raw
motion-control interface.

### Step 3 - Define the embodied task schema

Goal: define what Chromie may submit to Soridormi when the request is richer
than an atomic skill.

Candidate task types:

- `move_velocity`
- `turn_to_heading`
- `approach_target`
- `navigate_to_location`
- `look_at_target`
- `perform_gesture`
- `skill_sequence`
- `speak_while_moving`
- `stop_now`
- `recover_safe_idle`
- future `deliver_object`

Each task should include:

- task type and unique task id;
- natural-language summary for traceability, not for low-level control;
- structured goal parameters;
- task context from Chromie;
- environment context if available;
- safety constraints;
- timeout and cancellation policy;
- expected status/events;
- inspectable Soridormi `plan_steps` and `blocked_subsystems`;
- recommended next actions for Chromie's global graph;
- pass/fail metrics or refusal reasons.

Gate:

```text
schema tests cover valid tasks, malformed tasks, unsupported tasks, unsafe
tasks, timeout policy, cancellation policy, and safe-idle reporting
```

### Step 4 - Add Soridormi's internal embodied DAG/state-machine skeleton

Goal: give Soridormi an internal execution model before adding real autonomy.

Initial lifecycle:

```text
accepted
  -> resolving
  -> planning
  -> executing
  -> monitoring
  -> recovering
  -> completed | failed | cancelled | refused
```

This skeleton should track state, events, timestamps, safety status, and the
last known body status. It may execute only simple supported tasks at first.

M11 implementation note: the first body-DAG view is `task_graph` on every task
preview/status/submit response. It derives graph nodes from Soridormi
`plan_steps` and exposes sequence edges plus lifecycle metadata while keeping
physical execution disabled on the task API.
The M11B monitoring cursor extends `soridormi.task.events` with
`soridormi.task_events.v1`, `latest_sequence`, `next_after_sequence`,
`terminal`, `safe_idle`, and `poll_recommendation`, so Chromie can monitor a
task without guessing whether to continue polling or stop.
M11C adds `client_task_ref` as Chromie's retry-safe task identity. Repeated
submits with the same reference and identical payload return the original task
with `idempotent_replay=true`; conflicting payloads are rejected. M11D adds
timeout expiry for non-terminal planning-hold tasks: status/events/cancel reads
after `deadline_at` transition the task to terminal `failed` with a
`task_timed_out` event.

Gate:

```text
submit/status/events/cancel behave deterministically in local tests, and every
terminal state reports safe_idle or the reason safe_idle is false
```

### Step 5 - Wire current skills into task execution

Goal: let simple embodied tasks compile into existing safe skills while richer
tasks remain refused or future-gated.

Examples:

- `walk forward for 10 seconds` may compile to `walk_velocity`.
- `turn left then nod twice` may compile to a bounded skill sequence.
- `stop now` maps to stop/cancel/emergency behavior depending on urgency.
- `walk forward to the house` fails closed until target resolution,
  localization, route planning, and local obstacle checks exist.
- `bring me water` fails closed until manipulation, carry, and handoff
  capabilities exist.
- unsafe physical requests fail closed and are not lowered into body motion.

Gate:

```text
task execution uses structured skills only, never raw action_14d, raw motor
commands, or direct joint commands from Chromie
```

### Step 6 - Add acceptance-style tests and scenarios

Goal: test the system at the command/task boundary, not only at low-level motion.

Dry-run and unit scenarios:

- `stop now`
- `turn left then nod twice`
- `come closer slowly`
- `look at me and say hello`
- `walk forward to the house` refuses with `missing_navigation_pipeline`
- `bring me water` refuses with `missing_manipulation_capability`
- unsafe physical requests refuse with `unsafe_task`

Implementation note: these cases now live in
`task_acceptance_cases/mcp_task_acceptance.yaml` and replay through
`tests/test_task_acceptance_cases_m11.py`. The suite validates dry-run success
for executable contracts and fail-closed behavior for navigation,
manipulation, perception, stop-through-task, and unsafe physical requests.
The full no-motion contract milestone is the M11A gate:

```bash
./scripts/validate_task_agent_contract.sh
```

MuJoCo-backed scenarios should be added only for tasks that map to existing
safe skills.

Gate:

```text
unit tests validate task contracts and local state transitions; MuJoCo tests
validate only executable embodied skills with explicit backend/profile flags
```

### Step 7 - Update Chromie integration

Goal: make Chromie call Soridormi at the right abstraction level.

Chromie should:

- keep the global user/task DAG;
- ask clarifying questions and wait for user confirmation;
- submit structured embodied goals to Soridormi;
- call concrete Soridormi skills only when the user request is already concrete;
- monitor Soridormi task events;
- report progress, completion, blocked state, or refusal to the user.

Chromie should not:

- decompose rich embodied goals into raw velocity recipes unless that is the
  actual user intent and it is safe;
- send raw joint actions, torque commands, motor commands, or low-level policy
  outputs;
- treat Soridormi refusal as something to bypass.

Gate:

```text
Chromie docs and integration tests show global task planning on Chromie's side
and embodied task execution on Soridormi's side
```

## Recommended patch sequence

Build this as small, reviewable patches:

1. Contract docs and MCP naming.
2. Task schema and validation tests.
3. MCP manifest entries and local/mock task API.
4. Internal task state machine and event log.
5. Skill-backed dry-run execution for simple supported tasks.
6. Dry-run acceptance scenarios and fail-closed cases.
7. Chromie integration update.
8. MuJoCo-backed executable task validation.

The first behavior-changing patch should avoid pretending that navigation,
perception, or manipulation already exist. It should make those missing
capabilities visible through structured refusal reasons.

## Not now

Do not add these until the contract and test harness exist:

- raw natural-language-to-action policy input;
- raw perception directly into the low-level 14D action policy;
- hardware execution;
- manipulation or object delivery claims;
- open-loop gait replacement;
- unsafe physical task lowering;
- hidden fallbacks that report success after partial or failed execution.

## Validation expectation

Docs-only changes:

```bash
rg -n "task-agent implementation plan|soridormi.task.submit|Step 4" docs
python -c "from pathlib import Path; p = Path('docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md'); assert p.read_text().count(chr(96) * 3) % 2 == 0"
```

Code or manifest changes:

```bash
./scripts/validate_task_agent_contract.sh
```

MuJoCo-backed executable task changes:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/evaluate_scenario_rollout.sh --scenario flat_walk_varied_speed_v1 --backend mujoco --profile open_duck_forward
```

Optional visual inspection:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```
