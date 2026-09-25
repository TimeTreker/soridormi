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

The maintained boundary keeps speech, singing, TTS synthesis, audible playback,
and user-level interruption in Chromie's Speaking lane. Media playback is a
peer Activity capability rather than a Soridormi body capability. Soridormi
remains the embodied provider beneath Activity; changing the vocal or media
hosting boundary requires a separate evidence-backed architecture decision.

## Motion argument and semantic facade boundary

Current source amendment, 2026-09-16; prior integration evidence base `0af3d09`
on `codex/turn-count`, paired Chromie base `2e18f86a`. SC communication and Work remain separate.
Resource mappings retain upstream contracts and add route-to-source coverage.
Validation passes: 798 tests / two skips, body 165, task 147, governance, compile
and manifest checks. Native semantic/signed-direction failures remain unqualified. Resume by fetching
upstream, preserving dirty work, initializing submodules and rebuilding providers.
Generated social-eyes XML stays local; no third-party submodule revision changes.

`turn_in_place`, `curve_walk`, and `sidestep` now publish semantic left/right
facades; Planner owns direction and positive magnitude while the trusted provider
adapter alone realizes signed `yaw_radps` / `vy_mps`. Provider execution schemas
remain unchanged. `turn_in_place` count remains 1-8 with the existing 20-second
total limit. General `walk_velocity` is not force-fit because its signed forward
range is asymmetric; `look_at_person` still needs trusted target resolution rather
than a sign transform, while its requested duration remains Planner-supplied.
Manifest validation covers facade schemas, realization targets and semantic
`argument_realization` names. Native paired direction qualification remains open.
See [Soridormi Skill Execution](SORIDORMI_SKILL_EXECUTION.md).

Governance/compile pass; full suite: 790 passed / 2 skipped; concurrency: 156 passed.
Full-suite evidence: Chromie `.chromie/acceptance/failed-case-repair-20260914/`,
`soridormi-final-canonical.log` (local, not Git). Earlier simulator evidence does not qualify
physical speaker/robot behavior or resolve current Planner semantic failures.

## Verified repository surface

- Official Open Duck policy parity and replay/comparison tooling are retained as
  the trusted baseline.
- The runtime supports the official observation contract and declared
  command-conditioned context-policy profiles.
- Named skills are catalogued, schema-validated, bounded, interruptible where
  declared, and executable through the runtime adapter in MuJoCo.
- Resource execution now follows a dynamic capability-boundary contract: the
  provider may advertise granular acquisition/delivery capabilities, the complete
  composite, or both. Chromie owns composition across public leaves; Soridormi owns
  local planning inside each selected capability.
- `acquire_and_deliver_resource` is exported as a simulation-only scripted/mock
  named skill for the `physical_object + physical_handover` semantic scope. It
  returns explicit `resource_outcome` acquisition/delivery evidence so paired
  Chromie/Soridormi architecture can be exercised end to end without claiming a
  real manipulator or hardware grasp capability.
- The separate scenario runner reads a world-map/dynamic-element file, builds
  bodies with MuJoCo `MjSpec`, starts Soridormi, and applies timed pose events.
  The default scenario places a mock bottle on a fixed table 10 m ahead and
  fixed chairs left of Chromie. `observe_scene` reads current simulation geometry;
  it is not camera or physical perception. Focused and live MuJoCo tests cover
  creation and movement. Chromie does not poll it on ordinary turns; no live
  scene-to-speech proof exists.
- The generated MuJoCo visual-body overlay supplies proportioned jointless
  arms, detailed fixed-pose hands, and cosmetic lower-leg shells in the Open
  Duck palette. It does not change the official XML/URDF or 14-actuator policy;
  its display geoms have no dynamics, collision, sensing, or completion authority.
- `wave_hand`, `celebrate`, and `hug_gesture` are exported as simulation-only
  visual arm expressions. They sequence overlay poses through the independent
  `visual.arms` activity resource, restore `rest`, send no motor command, and do
  not claim physical arm movement, contact, manipulation, or hardware support.
- Exact concurrent body activities are validated and executable through
  `soridormi.activity.*`: one primary locomotion member may run with compatible
  bounded head overlays and independent visual expressions.
- The runtime composes locomotion and head overlays into one final motor command;
  eye and arm expressions remain independent non-motor visual outputs.
- Every currently executable social named skill exports explicit
  `behavior_domains` metadata. This lets Chromie's trusted Activity runtime
  validate an accepted Social Attention decoration against the same live
  provider contract used for execution instead of relying only on a brain-side
  catalog preset.
- Paired dirty-working-tree diagnostic
  `chromie/.chromie/acceptance/voice-log-water-fixed-67` completed two
  Social Attention `blink_eyes` event calls and the scripted
  `acquire_and_deliver_resource` request. The latter returned
  `no_motion=false`, explicit mocked acquisition/delivery evidence, and final
  standing safe idle. This is MuJoCo diagnostic evidence, not literal
  100-metre travel, physical water handling, or hardware qualification.
- The paired working tree passes repository governance and compile checks. A
  dependency-complete, full-checkout runtime container passes 748 tests with
  two target-dependent skips; focused body concurrency passes 128 tests and the
  task-agent contract passes 133. The current host Python lacks the declared
  `pyzmq` dependency, so the required host pytest/body gates stop during import
  collection and are not reported as passed.
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

## 2026-08-07 repository reproducibility audit

The paired Chromie post-merge audit exercised Soridormi from a fresh host and
container validation path. It found four implementation/contract mismatches at
the repository boundary; none changed the embodied authority model or public
capability surface:

- repository governance scanned generated Python bytecode and could report a
  retired term that existed only in `__pycache__`;
- the advertised host `pytest -q` gate depended on an undeclared ambient
  `PYTHONPATH`;
- container validation mounted only `scripts/`, so it did not validate the
  current checkout, and the task gate selected the generic runtime service
  rather than the declared `mcp-runtime` profile;
- the HTTP persistence regression inherited an operator backend and supplied a
  motion duration below the provider schema minimum, making a deterministic
  contract test environment-dependent; and
- the new host-wrapper regression inherited the container recursion sentinel
  when executed by the complete container gate, bypassing its fake Docker
  boundary and recursively launching the static suite; and
- GitHub Actions installed the declared host development dependencies but left
  the wrapper in `auto` mode. Because Docker is also present on the runner, the
  wrapper selected Compose before `.env` existed and failed image-variable
  interpolation instead of testing the prepared host checkout; and
- `.env.example` omitted the required runtime/simulator image variables and
  policy profile while retaining older CUDA base tags, so the workflow's
  documented sample-to-Compose validation could not interpolate any target.

The validators now exclude generated bytecode, declare `src` test imports,
mount the full checkout, select the MCP runtime profile, and make the HTTP test
explicitly dry-run with schema-valid arguments. The runtime image includes the
validator's `rg` dependency. The wrapper regression now explicitly owns a
synthetic host boundary by removing the inherited in-container sentinel.
The static workflow now explicitly selects that job's prepared host environment
while local `auto` mode remains container-first. Focused regressions cover the
generated-file, container-mount, recursion, and workflow-environment boundaries.
The example environment now matches `setup_env.sh` image/base defaults and a
regression requires it to satisfy every required Compose interpolation key.

Before delivery, repository governance passed, body concurrency passed 114
tests with four target-dependent skips, the replaceable-policy static suite
passed 90 tests, the task-agent contract passed 126 tests through its automatic
Docker wrapper, the full suite passed 728 tests with five target-dependent
skips, and `compileall` passed. A paired headless MuJoCo diagnostic then executed
the exact ordered compound request `walk_velocity(vx_mps=0.2,
duration_s=10)`, `nod_yes(count=2)`, and `turn_in_place`, and an independent
20-second `walk_velocity` run was cancelled only after provider start; both
ended standing and body-wide safe-idle with no active task. Those diagnostics
used the live endpoint's reported source revision `1c15371`; they are simulator
evidence, not physical-robot or candidate-release evidence. Final promotion
still requires clean revision-bound replay after merge.

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

- Chromie's three-lane runtime implementation in this repository; this repo
  defines only the Soridormi provider-side contract
- Speech, singing, TTS, or audible playback execution inside Soridormi
- Media playback, microphone/speaker adaptation, or a unified body/vocal/media
  execution runtime inside Soridormi
- Live balance-margin-based overlay suspension or a production WBC backend
- Concurrent nodding, bowing, shaking, or other unqualified standalone gestures
  during locomotion
- General physical task execution through the task API
- Autonomous real-world navigation, target tracking, obstacle avoidance, qualified
  manipulation, or non-mocked object delivery
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

- Qualify concurrent locomotion, gaze overlay, and visual expression in live
  MuJoCo and retain evidence for cancellation and safety preemption.
- Update the companion Chromie repository with the single-core three-lane model
  and coordinated-group lifecycle.
- Keep semantic contracts independent of implementation sequence labels.
- Keep body-state projections grounded in the live Soridormi runtime.
- Close paired Chromie/Soridormi contract qualification without weakening the
  body authority boundary. Chromie's vocal-mode repair requires exact body
  receipts from Soridormi but no Soridormi vocal or media implementation.
- Do not reopen TTS, playback, media, or audio-device migration work without
  retained evidence of a concrete blocker and an explicit boundary exception
  that updates contributor rules, durable docs, and machine-checked contracts
  together.
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
