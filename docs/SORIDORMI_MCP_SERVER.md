# Soridormi MCP server

Soridormi publishes its robot-facing capabilities through an MCP Streamable
HTTP service. Chromie remains a separate deployment and connects as an MCP
client. The broader two-agent agreement is documented in
`docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md`.

```text
chromie-agent container
        |
        | MCP Streamable HTTP
        v
soridormi-mcp container
        |
        v
Soridormi safe tool/runtime boundary
```

## Safety boundary

The server exposes the tools declared in `soridormi_runtime.mcp.manifest`.
The current public surface includes read-only robot state, bounded velocity
motion plans, named-skill plans, contract-only embodied task requests, safety
controls, and hidden test-only provider fault injection. Plan, task, and
emergency-stop state are shared across HTTP requests within one server process.

The default adapter wraps `SoridormiLocalToolService`, so motion and named-skill
execution are dry-run only and never send motor commands:

```bash
./scripts/run_mcp_server.sh
```

The runtime adapter drives the existing Soridormi robot/controller interfaces.
It can execute bounded velocity and scripted head skills in MuJoCo `sim` mode,
and is deliberately limited to `sim` until `HardwareRobot` is implemented:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/run_runtime_mcp_server.sh
```

Do not run the standalone runtime loop and the runtime MCP adapter against the
same robot backend at the same time. The adapter owns the control loop while a
plan is active.

`motion.stop`, `motion.cancel`, and `safety.emergency_stop` can preempt between
control ticks. Cancelling an in-flight MCP request also transitions the robot
to safe hold. `robot.get_status` reports `safe_idle=true` only when no task is
active and emergency stop is clear. Emergency-stop state remains active until
the MCP process is restarted; inspect robot state before resuming.

Chromie should prefer the named-skill path for body requests:

```text
soridormi.skill.list
  -> soridormi.skill.create_plan
  -> soridormi.safety.monitor_motion during execution
  -> soridormi.skill.execute_plan
  -> soridormi.robot.get_status for safe-idle confirmation
```

The lower-level `soridormi.motion.*` tools remain available for bounded
velocity-plan integration tests and emergency controls. Neither path exposes
raw joint targets, motor commands, torque commands, or low-level 14D policy
outputs to Chromie.

Rich embodied requests should use the task-level MCP surface:

```text
soridormi.task.get_capabilities
soridormi.task.preview
soridormi.task.submit
soridormi.task.status
soridormi.task.events
soridormi.task.cancel
```

This M11 surface is contract-first and no-motion. `soridormi.task.preview`
returns Soridormi's embodied interpretation without creating a persistent task
record. `soridormi.task.submit` validates and records structured embodied goals,
returns `execution_mode=contract_only` or a dry-run compilation mode, and
refuses missing or unsafe capability paths such as navigation, perception,
manipulation, and immediate stop-through-task. Until the task state machine and
execution adapter are implemented, it must not be treated as proof that the
robot actually performed the submitted goal.

`soridormi.task.get_capabilities` is Soridormi's own readiness catalog. It
declares which task types can dry-run now, which are planning holds, which
redirect to safety tools, and which are blocked by missing Soridormi subsystems
such as localization, route planning, target tracking, manipulation, or handoff.
It also separates external dependencies, such as Chromie-owned speech
coordination, from missing Soridormi body-runtime capability.
The catalog is backed by
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`; update
that file when Soridormi gains or loses embodied task readiness, then let the
MCP tool project it at runtime.

The task service exposes an internal lifecycle skeleton through `phase`,
`terminal`, `allowed_next_phases`, and structured events. Simple supported task
types can compile into existing Soridormi named skills and complete as
`execution_mode=skill_dry_run` with `no_motion=true`. Multi-step structured
requests can compile into a bounded `skill_sequence` and complete as
`execution_mode=skill_sequence_dry_run`, for example `turn_in_place` followed by
`nod_yes`. Cross-agent or unsupported tasks remain held at the Soridormi-owned
`planning` boundary or fail closed. The longer-term implementation should let
Soridormi run its own sensing, planning, gait, monitoring, and recovery loop
internally after a task has been accepted.

`soridormi.task.events` is the M11B monitoring cursor. Callers pass
`after_sequence` and receive a versioned `soridormi.task_events.v1` payload with
the matching event slice, `latest_sequence`, `next_after_sequence`, terminal
state, safe-idle state, and `poll_recommendation`. Chromie can keep polling
with the returned cursor while a task is active, cancel if needed, and stop
polling once `terminal=true`.

`client_task_ref` is the M11C retry key. Chromie may include it on
`soridormi.task.submit`; if the same reference and payload are submitted again,
Soridormi returns the original `task_id` with `idempotent_replay=true` instead
of creating a duplicate task. Reusing a `client_task_ref` with a different
payload is rejected. `soridormi.task.status`, `soridormi.task.events`, and
`soridormi.task.cancel` can look up the task by either Soridormi `task_id` or
Chromie `client_task_ref`.

M11D adds timeout expiry for no-motion planning holds. A non-terminal task that
passes its `deadline_at = created_at + timeout_s` transitions to terminal
`failed` when status, events, or cancel is read. The response reports
`expired=true`, `timeout_elapsed_s`, and a `task_timed_out` event. If the task
used `cancellation_policy=emergency_stop_on_timeout`, Soridormi reports the
reason `task_timeout_emergency_stop_required` and recommends the dedicated
emergency-stop path; it still does not send physical motion through the task
API.

Task status also includes `plan_steps` and `blocked_subsystems`. These fields
explain Soridormi's embodied interpretation: for a blocked navigation task the
steps may show sensing, localization, routing, local planning, and control
layers; for a dry-run sequence the steps show the named body skills that would
be used. They are inspectable planning metadata only. They are not raw joint,
motor, torque, actuator, or `action_14d` commands.

The same response includes `task_graph`, a Soridormi-owned body-task DAG view.
It wraps the plan steps with stable node IDs, sequence edges, current phase,
terminal state, blocked subsystems, and explicit `raw_control_allowed=false`.
This graph is for Chromie's monitoring and routing. Chromie still owns the
global user/task graph; Soridormi owns only the embodied body graph beneath the
submitted task.

Task preview/submit responses also include `recommended_next_actions`. These
are machine-readable routing hints for Chromie, such as
`submit_task_when_confirmed`, `monitor_task_or_cancel`,
`report_blocked_or_clarify`, `call_dedicated_stop_tool`, or
`do_not_lower_to_velocity_recipe`. They are not execution receipts and they do
not authorize lower-level robot control.

Task-level acceptance examples live in
`task_acceptance_cases/mcp_task_acceptance.yaml`. They replay through the local
MCP task service using both preview and submit paths and are intentionally
no-motion: they validate the contract boundary, refusal reasons, and skill
compilation without claiming the robot physically executed the task.

Run the M11A task-agent contract gate from the Soridormi repo root:

```bash
./scripts/validate_m11_task_agent_contract.sh
```

A compact local demo is available:

```bash
./scripts/demo_task_mcp_contract.sh
```

It shows a Chromie-style handoff: each case starts from a user command, displays
the structured task payload Chromie would send, then shows Soridormi's no-motion
task response. It runs `get_capabilities`, previews a blocked navigation
request, submits a `turn_in_place` plus `nod_yes` skill sequence dry-run,
submits a blocked water delivery request, and previews an immediate stop
redirect. The output is JSON when run with `--json`; the default output is a
concise human-readable summary. Every case reports `no_motion=true`.

## Run the dry-run container

```bash
./scripts/run_mcp_server.sh
```

This builds and starts the dedicated `soridormi-mcp` container from
`compose.mcp.yaml`. It publishes:

```text
http://127.0.0.1:8000/mcp
```

Override `SORIDORMI_MCP_PORT` in `.env` when port 8000 is unavailable.

Chromie should use a reachable host address, for example:

```env
SORIDORMI_MCP_URL=http://host.docker.internal:8000/mcp
```

Keep `host.docker.internal` in Chromie's `NO_PROXY` list so robot-control MCP
traffic does not pass through a general HTTP proxy.

## Run directly for development

```bash
python -m soridormi_runtime.mcp.http_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode sim
```

Run the runtime adapter directly only from an environment with Soridormi
runtime dependencies, policy assets, and a reachable simulator:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy \
python -m soridormi_runtime.mcp.http_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode sim \
  --adapter runtime
```

The server uses stateless MCP transport with JSON responses. Application state
such as created plans remains process-local and protected against concurrent
tool calls.
