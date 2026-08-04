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

The safe provider profiles are `sim`, recommendation-only `hardware_shadow`,
and no-motion `hardware_dry_run`. The manifest is shared by the local dry-run
adapter and the runtime adapter:

- local adapter: motion and named-skill execution are no-motion provider
  contract checks;
- runtime adapter: named-skill execution may move the MuJoCo robot in `sim`
  mode through bounded velocity or scripted head/body skills; body-activity
  execution may run resource-compatible locomotion, bounded gaze, and visual
  expressions concurrently;
- hardware shadow/dry-run profiles remain no-motion until a hardware adapter is
  implemented and separately commissioned.

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

For the full task-agent contract gate, including task capability
readiness, manifest export, acceptance cases, `task_graph`, and docs checks:

```bash
./scripts/validate_task_agent_contract.sh
```

## Safety boundary

The manifest exposes named body skills, short velocity-plan tools,
resource-aware body-activity tools, safety controls, task-level contract tools,
and status checks. `soridormi.robot.get_status` includes `safe_idle`,
`activity_idle`, and `active_lanes` so Chromie can monitor body execution
without receiving low-level commands.

`soridormi.activity.*` is the exact concurrent body-execution surface. Its
members are already-selected named skills. Every member declares:

```text
ability_class
control_coupling
write_resources
optional concurrency envelope
```

The plan validator permits one primary locomotion/whole-body controller and
multiple compatible subtle expressions, while enforcing one writer per
physical resource. Bounded head/gaze overlays are composed into the final
motor command. Eye animation is an independent visual output. Speech and
singing remain an external peer lane owned by Chromie and are linked through
`coordination_id`, never inserted into Soridormi's physical plan.

The activity executor reports per-member results, aggregate status, resource
claims, cancellation state, and the invariants
`one_final_motor_command_authority=true` and `safety_authority=soridormi`.

The `soridormi.task.*` tools are intentionally contract-first in embodied task contract.
`soridormi.task.get_capabilities` is read-only and reports Soridormi-owned
embodied readiness by task type: readiness state, required subsystems, ready
subsystems, missing subsystems, external dependencies, and whether persistent
submission is currently allowed. The preview and submit tools
accept structured embodied task payloads and return `no_motion=true`.
Its readiness table is sourced from
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`, so the
manifest, local MCP service, and runtime MCP service all report the same
Soridormi-owned task boundary.
`soridormi.task.preview` is non-persistent and uses `preview_id`.
`soridormi.task.submit` records task status/events and uses `task_id`. Simple
tasks such as `move_velocity`, `turn_to_heading`, `look_at_target`, and
`perform_gesture` can compile into existing named skills and complete as
`execution_mode=skill_dry_run`; this validates the skill plan without
commanding MuJoCo or hardware through the task API. A `skill_sequence` task can
dry-run a bounded ordered list of named skills and reports
`execution_mode=skill_sequence_dry_run` plus `skill_sequence` step metadata.
They fail closed for missing navigation, perception, manipulation,
emergency-stop recovery, unsafe physical tasks, and any payload that contains
raw low-level robot control fields. Task status includes `phase`, `terminal`,
`allowed_next_phases`, `skill_id`, `skill_sequence`, and duration/summary
metadata so Chromie can monitor the Soridormi-owned embodied lifecycle without
receiving low-level control details.

Task status also includes `plan_steps` and `blocked_subsystems`.
`plan_steps` expose Soridormi-owned embodied layers such as sensing,
localization, routing, planning, skill execution, safety, interaction, and
manipulation. They exist to explain acceptance, dry-run compilation, or
fail-closed refusal. They must not be interpreted as actuator commands or a
low-level policy input.

Task status also exposes `task_graph`, a derived Soridormi body-task DAG. It
uses stable node IDs and sequence edges around the plan steps, reports the
current task phase and terminal state, and always marks
`raw_control_allowed=false` in the embodied task contract surface. This gives Chromie a
monitorable body graph without transferring low-level control ownership.

`soridormi.task.events` exposes the task event cursor contract. The response is
versioned as `soridormi.task_events.v1` and includes `status`, `phase`,
`terminal`, `safe_idle`, `returned_count`, `latest_sequence`,
`next_after_sequence`, `has_more`, and `poll_recommendation`. This lets Chromie
track long-running Soridormi tasks with an explicit cursor instead of inferring
task completion from raw event lists.

The retry-safe task contract uses `client_task_ref`.
`soridormi.task.submit` accepts optional
`client_task_ref`; a duplicate submit with the same reference and identical
payload returns the original Soridormi `task_id` with `idempotent_replay=true`.
The same reference with a different payload is rejected. `task.status`,
`task.events`, and `task.cancel` accept either `task_id` or `client_task_ref`.

No-motion planning holds expose timeout visibility and expiry. Task status payloads include
`deadline_at`, `expired`, and `timeout_elapsed_s`. Non-terminal planning-hold
tasks that exceed `timeout_s` become terminal `failed` tasks with a
`task_timed_out` event and timeout-specific `recommended_next_actions`.

Task responses also include `recommended_next_actions`. This list is for
Chromie's global task graph: it can tell Chromie to submit after confirmation,
monitor or cancel a persistent task, call the dedicated stop tools, report a
blocked capability, or avoid lowering a rich goal into a velocity recipe. It is
not user-facing copy and does not claim physical completion.

Soridormi does not expose raw motor, joint, torque, backend APIs, or low-level
14D policy outputs to the LLM layer.
