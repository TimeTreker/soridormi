# Soridormi target and capability roadmap

This document defines the durable target and semantic capability sequence.
Current status lives in `docs/STATUS.md`; candidate metrics live in generated
evidence.

## Target

Build a reusable platform execution runtime that keeps Chromie independent from
the current simulator, robot, audio stack, sensors, and device drivers while
preserving strict cognitive, authorization, execution, and safety boundaries.
Open Duck Mini v2 remains the primary body platform and policy baseline, but it
is one Platform Provider rather than the public meaning of Soridormi.

Soridormi will:

- preserve a trusted official Open Duck policy baseline;
- expose the same platform-neutral execution contracts across MuJoCo, desktop,
  and future hardware providers;
- execute bounded body, vocal, media, platform-perception, and device
  capabilities selected and authorized by Chromie;
- validate, prepare, schedule, monitor, interrupt, recover, refuse, and produce
  normalized per-member evidence independently;
- keep one private Platform Contract for simulator, robot, audio, sensor,
  controller, driver, calibration, and hardware-safety adaptation;
- support replaceable policies and providers with reproducible evaluation; and
- never expose raw low-level body, device, or provider authority to a language
  model.

## Execution-runtime and platform-provider foundation

Retain and generalize the current runtime/simulator separation into two logical
containers:

```text
soridormi-runtime <-> soridormi-platform
```

The execution runtime contains stable capabilities, resources, preparation,
timing, cancellation, recovery, and evidence. The platform provider contains
MuJoCo or one qualified physical/desktop adapter and all device-specific code.

Gate: execution runtime code imports no MuJoCo, robot SDK, sound-device backend,
or platform driver; the private Platform Contract preserves semantics across
providers.

## Runtime and body-policy parity

Maintain stable Robot API, observation, action, controller, profile, logging,
replay, and official parity contracts for the Open Duck body provider.

Gate: unexplained parity divergence is closed and baseline tools remain usable.

## Replaceable policy and evidence loop

Support versioned profiles, model validation, deterministic datasets, training,
closed-loop scenarios, teacher comparison, packaging, retention, and rollback.

Gate: model/profile compatibility and required MuJoCo scenarios pass; offline
loss alone is insufficient.

## Named body skills

Expose bounded semantic body behavior with availability, argument schemas,
effects, interruption, refusal, and safe-idle confirmation.

Gate: no caller supplies raw joint, motor, torque, coordinate, or policy-action
authority.

## Concurrent body activities

Execute exact compatible body skills concurrently through declared physical
resources and control coupling. Support one primary locomotion/whole-body
controller, bounded motor-command overlays, independent visual expressions,
per-member status, and global physical preemption.

Gate: response meaning remains Chromie-owned; one writer owns each physical
resource; one final Soridormi motor-command authority exists; incompatible
activity fails closed; cancellation and emergency stop restore safe physical
state.

## Vocal execution

Execute typed Vocal Plans authored by Chromie through provider-declared modes,
streaming, timing, interruption, delivery, and terminal evidence. Speech,
expressive speech, recitation, singing, and humming are related vocal modes, not
proof of one another.

Gate: Soridormi does not rewrite content or vocal mode; expressive TTS cannot
advertise singing without mode-specific evidence; unsupported modes return an
exact unavailable result.

## Media execution

Execute existing music, recordings, streams, and sound effects as Activity
capabilities with independent playback state, progress, cancellation, and
completion evidence.

Gate: media playback is never reported as singing; shared audio output or mixer
does not merge vocal and media semantics.

## Multimodal execution coordination

Coordinate compatible body, vocal, and media members through provider-local
prepared state, monotonic start, declared resources, cancellation, recovery,
and measured evidence.

Gate: best-effort, synchronized, and atomic guarantees are distinct; requested
simultaneity is not silently serialized or upgraded beyond retained evidence.

## Embodied task contract

Accept richer structured goals, keep a Soridormi-owned body-task lifecycle and
graph, expose blocked subsystems and events, and fail closed for missing
capability.

Gate: contract behavior is deterministic, retry-safe, timeout-safe, no-motion,
and paired with Chromie integration tests.

## Monitored task execution

Add physical task execution only after sensing, planning, skill selection,
monitoring, cancellation, recovery, and completion evidence exist.

Gate: a task cannot claim completion from preview, prepared state, dry run,
partial execution, or unsupported fallback.

## Generalization and control improvement

Expand scenario coverage, context features, clearance/stability, recovery,
rough terrain, turning, lateral behavior, and held-out evaluation. Add WBC,
residual, or adaptive control only behind explicit contracts and evidence.

Gate: retained candidates beat the reference on named objectives without safety
or generalization regression.

## Hardware commissioning

Implement a hardware Platform Provider through read-only state, dry-run
commands, limits, watchdog, independent stop, low-power tests, standing,
tethered movement, audio/sensor qualification, and broader validation.

Gate: no hardware actuation before simulator, commissioning, and operator
evidence are complete.

## Hardware direction

Near-term robot deployment prioritizes Jetson Orin-class platforms that fit the
Open Duck Mini power, size, and software envelope. AGX Orin and Orin NX are the
primary engineering targets; Orin Nano-class deployment may use a reduced
profile. Jetson Thor remains exploratory, not a prerequisite.

## Clearance qualification tools

```bash
./scripts/report_clearance_candidate_history.sh
./scripts/validate_clearance_engineering_process.sh
```

These commands summarize retained evidence and validate the offline
clearance process. They do not train, launch MuJoCo, or authorize hardware.
