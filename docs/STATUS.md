# Soridormi current status

This is the repository's only current-state summary. Update it in the same
change that alters a public capability, safety boundary, promotion claim, or
current work priority. Do not copy candidate-by-candidate metrics here.

## System boundary

Soridormi is the embodied body/cerebellum runtime. Chromie is the separate
cognitive and social brain.

```text
human/environment
  -> Chromie: conversation, goals, confirmation, global orchestration
  -> structured proposal or embodied task
  -> Soridormi: validation, body planning, safety, execution, monitoring
  -> MuJoCo now; qualified hardware later
```

Chromie proposals are advisory. Soridormi rejects executable intent metadata,
raw controls, and physical coordinates, then independently validates every
body plan.

## Verified repository surface

- Official Open Duck policy parity and replay/comparison tooling are retained as
  the trusted baseline.
- The runtime supports the official observation contract and declared
  command-conditioned context-policy profiles.
- Named skills are catalogued, schema-validated, bounded, interruptible where
  declared, and executable through the runtime adapter in MuJoCo.
- The MCP service exposes robot state, safety tools, bounded motion plans, named
  skills, and a task-level contract surface.
- Task capabilities, preview, submit, status, events, and cancel are implemented
  as no-motion contract operations.
- Task requests reject low-level control fields recursively and fail closed for
  unsafe, unsupported, navigation, perception, and manipulation requests.
- Task events support cursor-based polling, retry-safe client references, and
  timeout expiry.
- Runtime-backed status reports the live Soridormi source revision for paired
  evidence with Chromie.
- Generated data and evidence directories are ignored by git.

## Important distinction

The task API may compile a supported task into a named-skill dry run, but it
does not physically execute that task. Physical motion must use the validated
skill or bounded motion execution path. Chromie must not report physical
completion from a task dry-run result.

Body-wide `safe_idle` is owned by the runtime state. Task records project that
state; they do not infer safe idle merely because emergency stop is clear.

Target-oriented tasks require explicit structured targets and directions.
Soridormi does not silently substitute a person, pose, or coordinate.

## Not currently claimed

- General physical task execution through the task API
- Autonomous navigation, target tracking, obstacle avoidance, manipulation, or
  object delivery
- A hardware backend
- Hardware-ready locomotion or broad terrain/generalization qualification
- Completion inferred from a plan, dry run, offline score, or partial evidence

## Current blockers

- Rich embodied tasks need sensing, localization, local planning, monitored
  execution, and recovery before the task API may issue motion.
- Hardware work remains blocked behind simulator qualification, commissioning,
  watchdog, limits, stop, and operator evidence.
- Locomotion candidates require retained scenario evidence and human visual
  review before any broader promotion claim.

## Current work order

- Keep semantic contracts independent of implementation sequence labels.
- Keep body-state projections grounded in the live Soridormi runtime.
- Close paired Chromie/Soridormi contract qualification without weakening the
  body authority boundary.
- Improve locomotion/generalization only through explicit scenario and safety
  gates.
- Begin hardware work only through the staged commissioning contract.

## Required gates

```bash
python scripts/validate_repository_governance.py
SORIDORMI_TASK_AGENT_USE_DOCKER=0 ./scripts/validate_task_agent_contract.sh
pytest -q
```
