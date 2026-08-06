# Soridormi current status

This is the repository's only current-state summary. Update it in the same
change that alters a public capability, safety boundary, promotion claim, or
current work priority. Do not copy candidate-by-candidate metrics here.

## System boundary

Soridormi is currently the embodied body/cerebellum runtime. Chromie is the
separate cognitive and social brain.

```text
human/environment
  -> Chromie: conversation, goals, confirmation, authorization, interaction orchestration
  -> structured proposal or embodied request
  -> Soridormi: body validation, planning, safety, execution, monitoring
  -> MuJoCo now; qualified hardware later
```

The approved target broadens Soridormi into the platform execution plane while
preserving Chromie semantic authority:

```text
Chromie Interaction Orchestrator
  -> immutable platform-neutral execution envelope
Soridormi Execution Runtime
  -> body, vocal, media, platform-facing capability execution and evidence
Soridormi Platform Provider
  -> MuJoCo, physical robot, desktop audio, sensors, drivers, and hardware safety
```

This target is documented but **not implemented**. Current Chromie still owns
TTS synthesis/playback and much of cross-provider scheduling. Current Soridormi
primarily owns body execution. Documentation must not turn the target into a
runtime claim.

## Verified repository surface

- Official Open Duck policy parity and replay/comparison tooling are retained as
  the trusted baseline.
- The runtime supports the official observation contract and declared
  command-conditioned context-policy profiles.
- Named skills are catalogued, schema-validated, bounded, interruptible where
  declared, and executable through the runtime adapter in MuJoCo.
- Exact concurrent body activities are validated and executable through
  `soridormi.activity.*`: one primary locomotion member may run with compatible
  bounded head overlays and independent visual expressions.
- The runtime composes locomotion and head overlays into one final motor command;
  `blink_eyes` remains an independent non-motor visual output.
- The MCP service exposes robot state, safety tools, bounded motion plans, named
  skills, concurrent body activities, and a task-level contract surface.
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
Soridormi does not silently substitute a person, pose, coordinate, vocal mode,
media item, or other user outcome.

## Not currently claimed

- the general Soridormi Execution Runtime envelope for body, vocal, media, and
  platform-facing capabilities;
- speech, expressive speech, recitation, singing, or humming execution inside
  Soridormi;
- music, recording, stream, or sound-effect playback inside Soridormi;
- normalized microphone/speaker or sensor Platform Contracts;
- synchronized multimodal prepare/start/cancel across body, vocal, and media;
- a physical or desktop platform provider that implements the approved general
  Platform Contract;
- Chromie's three-lane runtime implementation in this repository;
- live balance-margin-based overlay suspension or a production WBC backend;
- concurrent nodding, bowing, shaking, or other unqualified standalone gestures
  during locomotion;
- general physical task execution through the task API;
- autonomous navigation, target tracking, obstacle avoidance, manipulation, or
  object delivery;
- hardware-ready locomotion or broad terrain/generalization qualification;
- completion inferred from a plan, prepared state, dry run, offline score, or
  partial evidence.

## Current blockers

- The companion Chromie repository must first repair vocal-mode Goal and Planner
  semantics. The retained walk/sing/blink episode still fails before a valid
  Soridormi request, and Soridormi must not absorb that semantic defect.
- A reviewed immutable Chromie-to-Soridormi execution/evidence envelope is
  required before body, vocal, or media paths migrate.
- Vocal providers need explicit mode declarations and mode-specific evidence;
  expressive TTS is not evidence of singing.
- Rich embodied tasks need sensing, localization, local planning, monitored
  execution, and recovery before the task API may issue motion.
- Hardware work remains blocked behind simulator qualification, commissioning,
  watchdog, limits, stop, and operator evidence.
- Locomotion candidates require retained scenario evidence and human visual
  review before any broader promotion claim.

## Current work order

- Keep the current body runtime, concurrent-body contracts, cancellation, and
  MuJoCo evidence passing while the new boundary is designed.
- Coordinate the vocal-mode Goal/Planner repair in Chromie; do not add phrase
  rules or silently reinterpret singing inside Soridormi.
- Define the immutable execution envelope and private Platform Contract.
- Introduce a Soridormi Execution Runtime facade around existing body behavior
  before adding vocal or media execution.
- Add vocal/TTS, media, platform audio/sensors, and multimodal coordination only
  through the semantic, gate-driven work in the execution roadmaps.
- Keep body-state and future audio/media projections grounded in live runtime or
  platform evidence.
- Improve locomotion/generalization only through explicit scenario and safety
  gates.
- Begin hardware work only through the staged commissioning contract.

## Required gates

```bash
python scripts/validate_repository_governance.py
./scripts/validate_body_concurrency.sh
SORIDORMI_TASK_AGENT_USE_DOCKER=0 ./scripts/validate_task_agent_contract.sh
pytest -q
```
