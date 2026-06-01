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
