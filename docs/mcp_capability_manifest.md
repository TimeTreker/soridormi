# Soridormi MCP Capability Manifest

Soridormi owns robot-body capabilities and exports them as an MCP-ready local
manifest. Chromie owns the global registry, LLM router, TTS/ASR, confirmation,
and cross-agent DAG planning.

Soridormi therefore exports only `soridormi.*` tools:

- `soridormi.robot.get_status`
- `soridormi.robot.get_mode`
- `soridormi.robot.get_battery`
- `soridormi.motion.create_plan`
- `soridormi.motion.execute_plan`
- `soridormi.motion.stop`
- `soridormi.motion.cancel`
- `soridormi.skill.list`
- `soridormi.skill.create_plan`
- `soridormi.skill.execute_plan`
- `soridormi.safety.monitor_motion`
- `soridormi.safety.emergency_stop`

The safe provider profiles are `sim`, recommendation-only
`hardware_shadow`, and no-motion `hardware_dry_run`. Named-skill execution in
these profiles validates the provider contract without sending actuator or
simulator commands.

The export also contains hidden `soridormi.testing.configure_fault` and
`soridormi.testing.clear_faults` tools. They are `llm_visible=false`,
restricted to test orchestration, and declare the supported provider-readiness
fault scenarios in `metadata.provider_readiness`.

It intentionally does **not** export `chromie.speak`, `chromie.listen`, or
`chromie.ask_confirmation`; those belong to Chromie.

## Export

From the Soridormi repo root:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.export_capabilities > soridormi_capabilities.json
```

Select a safe profile with `--mode`, for example:

```bash
PYTHONPATH=src python -m soridormi_runtime.mcp.export_capabilities \
  --mode hardware_shadow > soridormi_capabilities.json
```

Then Chromie can merge it into the global capability registry:

```bash
PYTHONPATH=agent python -m app.list_capabilities \
  --manifest /path/to/soridormi_capabilities.json \
  --llm-context --language zh
```

## Safety boundary

The manifest exposes short velocity-plan and safety tools. It does not expose
raw motor, joint, torque, or backend APIs to the LLM layer.
