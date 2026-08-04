# Soridormi MCP DAG integration contract

Soridormi does not own the global LLM DAG planner. Chromie owns that layer.
Soridormi only exports robot-body MCP capabilities plus a small task-graph
contract that tells Chromie how those tools must be composed safely.

## Boundary

Soridormi exports robot tools such as:

- `soridormi.robot.get_status`
- `soridormi.motion.create_plan`
- `soridormi.motion.execute_plan`
- `soridormi.motion.stop`
- `soridormi.activity.get_capabilities`
- `soridormi.activity.compile`
- `soridormi.activity.execute`
- `soridormi.activity.status`
- `soridormi.activity.cancel`
- `soridormi.task.get_capabilities`
- `soridormi.task.preview`
- `soridormi.task.submit`
- `soridormi.task.status`
- `soridormi.task.events`
- `soridormi.task.cancel`
- `soridormi.safety.monitor_motion`
- `soridormi.safety.emergency_stop`

Soridormi does not export:

- `chromie.speak`
- `chromie.listen`
- `chromie.ask_confirmation`
- user-facing TTS/ASR tools

Chromie aggregates both sides in its global capability registry.

## Export

Export the capability bundle, including the DAG contract:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.export_capabilities > /tmp/soridormi_capabilities.json
```

Export only the DAG contract:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.export_capabilities --dag-contract-only
```

## Required composition

A safe short-motion DAG should follow this shape:

1. `soridormi.robot.get_status`
2. `soridormi.motion.create_plan`
3. `chromie.ask_confirmation`
4. `soridormi.safety.monitor_motion` during `soridormi.motion.execute_plan`
5. `soridormi.motion.execute_plan`
6. `chromie.report`

`stop` and `emergency_stop` may preempt any running motion task. Raw motor,
joint, and torque APIs must remain outside LLM-visible manifests.

## Coordinated speaking and body activity

Chromie has one Cognitive Core and three coordination lanes: Social-Attention
Proposal, Speaking Execution, and Activity Execution. The lanes do not select
meaning independently. The authoritative planner chooses exact capabilities,
and the Cognitive Runtime Coordinator preserves start, dependency, and
cancellation relationships.

For “walk toward me while singing and blinking,” the safe shape is:

1. the Cognitive Core resolves the Goal and exact target;
2. the planner selects `chromie.vocal.perform`, a locomotion skill, and
   `blink_eyes` or a bounded gaze skill;
3. `soridormi.activity.get_capabilities` checks current body concurrency;
4. `soridormi.activity.compile` validates exact body members and resources;
5. Chromie's coordinator starts the Speaking lane and
   `soridormi.activity.execute` using the same `coordination_id`;
6. `soridormi.safety.monitor_motion` and `soridormi.activity.status` monitor
   physical execution;
7. interaction cancellation propagates to the selected lanes, while Soridormi
   may independently preempt physical work for safety;
8. completion speech waits for authoritative lane outcomes.

Social Attention may propose gaze, blinking, acknowledgment, or suppression,
but the proposal is accepted, rejected, or transformed by the single Cognitive
Core before exact capabilities are dispatched. The Activity lane invokes and
monitors the selected provider calls; it does not reinterpret the Goal.

Soridormi never receives speech meaning as a body member. It owns physical
resource arbitration, command composition, emergency stop, recovery, and final
body evidence.

A rich embodied task DAG should keep global reasoning in Chromie and submit only
structured body goals to Soridormi:

1. `chromie` resolves the user request, ambiguity, and confirmation.
2. `soridormi.task.get_capabilities` reads Soridormi-owned embodied readiness when Chromie needs current support/missing-subsystem state.
3. `soridormi.task.preview` inspects Soridormi's no-motion embodied interpretation when clarification, refusal explanation, or pre-confirmation planning is useful.
4. `soridormi.task.submit` records the structured embodied goal when Chromie decides to create the task.
5. `soridormi.task.status` or `soridormi.task.events` reports progress.
6. `soridormi.task.cancel`, `soridormi.motion.stop`, or
   `soridormi.safety.emergency_stop` handles cancellation or safety stop.
7. `chromie.report` tells the user the result.

In embodied task contract, `soridormi.task.preview` and `soridormi.task.submit` are contract-only
and no-motion. Preview uses `preview_id` and does not persist a task record.
Submit uses `task_id` and records lifecycle state. They are schema and lifecycle
boundaries, not navigation, manipulation, or autonomous execution claims.
`soridormi.task.get_capabilities` is the body-runtime readiness source for
which task types are dry-run ready, held, safety-redirected, or future-blocked.
That readiness source is the Soridormi config file
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`; Chromie
should treat the MCP response as the runtime view of that body-owned contract.
`soridormi.task.status` reports `phase`, `terminal`, and
`allowed_next_phases`; simple supported tasks may complete as `skill_dry_run`,
and bounded multi-step requests may complete as `skill_sequence_dry_run`.
Cross-agent tasks currently stop at the Soridormi-owned `planning` hold.
Blocked tasks can expose `plan_steps` and `blocked_subsystems` so Chromie's
global DAG can report why Soridormi refused or held the request without
inventing a lower-level workaround.
They also expose `task_graph`, which is Soridormi's body-side DAG view with
node IDs, sequence edges, current phase, terminal state, and
`raw_control_allowed=false`. Chromie can read this graph for monitoring, but it
must not merge Soridormi's body graph into raw motor or policy outputs.
`recommended_next_actions` provides the routing bridge back to Chromie: preview
may recommend `submit_task_when_confirmed`, stop requests recommend the
dedicated stop tools, and missing navigation/manipulation/perception recommends
reporting or clarifying instead of lowering to `walk_velocity`.

## Tool services

Soridormi includes a small in-process dry-run tool service:

```python
from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
```

It implements the robot-body tools declared in the manifest, but motion execution
is dry-run only. It validates bounded velocity commands, creates short-lived plan
IDs, records contract-only task requests, and refuses execution after an
emergency stop. It never sends motor, joint, or torque commands.

A CLI wrapper is available for smoke tests and future adapter work:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.call_tool soridormi.robot.get_status
PYTHONPATH=src python -m soridormi_runtime.mcp.call_tool \
  soridormi.motion.create_plan \
  --args-json '{"commands":[{"vx":0.08,"vy":0.0,"yaw":0.0,"duration_s":1.0}]}'
```

The Streamable HTTP server uses this dry-run service by default. For
runtime-backed simulation, `SoridormiRuntimeToolService` owns the existing
Soridormi robot/controller interfaces and executes bounded velocity segments:

```bash
./scripts/run_runtime_mcp_server.sh
```

The runtime adapter is preemptible by stop, cancel, emergency stop, and MCP
request cancellation. It currently rejects hardware modes because the
`HardwareRobot` backend is still a placeholder.
