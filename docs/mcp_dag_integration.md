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

## Local dry-run tool shim

Soridormi now includes a small in-process MCP-style tool service:

```python
from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
```

It implements the robot-body tools declared in the manifest, but motion execution
is dry-run only. It validates bounded velocity commands, creates short-lived plan
IDs, and refuses execution after an emergency stop. It never sends motor, joint,
or torque commands.

A CLI wrapper is available for smoke tests and future adapter work:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.call_tool soridormi.robot.get_status
PYTHONPATH=src python -m soridormi_runtime.mcp.call_tool \
  soridormi.motion.create_plan \
  --args-json '{"commands":[{"vx":0.08,"vy":0.0,"yaw":0.0,"duration_s":1.0}]}'
```

This is not the final MCP server. It is the robot-side tool core that a future
stdio or HTTP MCP server should wrap.
