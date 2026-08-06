# Soridormi architecture

## System split

```text
Chromie cognitive/social brain
  -> Goal meaning, response meaning, authorization, and immutable platform-neutral execution envelope
Soridormi Execution Runtime
  -> provider-local validation, preparation, resources, timing, execution, cancellation, recovery, and evidence
Soridormi Platform Provider
  -> MuJoCo, physical robot, desktop audio, sensors, controllers, and device drivers
```

Chromie owns human meaning and global interaction orchestration. It has one
Cognitive Core with Social-Attention, Speaking, and Activity coordination lanes.
The lanes describe how user responsibilities are completed; they are not
independent minds, processes, or platform adapters.

Soridormi owns platform-facing execution. It may execute body, vocal, media,
platform-perception, and device capabilities, but it cannot reinterpret a Goal,
rewrite authored communication, widen confirmation, or choose a different user
outcome. Platform-specific fields and providers remain private to Soridormi.

This is the approved target boundary. The current implementation is narrower:
Soridormi primarily executes body capabilities, while Chromie still owns TTS
synthesis/playback and much of cross-provider coordination. Migration is issue-
sized and evidence-gated; this document does not claim it has occurred.

## Two-container execution/platform split

```text
soridormi-runtime <---- private stable Platform Contract ----> soridormi-platform
```

The target production deployment has two logical containers:

- **Soridormi Execution Runtime:** stable high-level capability contracts,
  provider declarations, resource arbitration, preparation, scheduling,
  synchronization, interruption, cancellation, recovery, and normalized
  per-member evidence. It must not import MuJoCo, robot SDKs, desktop audio
  backends, or device-specific drivers.
- **Soridormi Platform Provider:** exactly one active simulator, physical-robot,
  or desktop-platform adapter. It owns MuJoCo or hardware integration, audio and
  sensor devices, controllers, drivers, calibration, state estimation, and
  hardware safety.

The existing runtime/simulator separation is the retained body foundation for
this target. A hardware provider must implement the same private Platform
Contract without changing public capability meaning. The two containers may
share one host, accelerator, mixer, or release image; co-location does not erase
the contract or safety boundary.

## Chromie Interaction Orchestrator versus Soridormi Execution Coordinator

The Chromie Interaction Orchestrator remains in Chromie. It owns session and
turn lifecycle, VAD/ASR coordination over normalized input streams,
Gateway/Core dispatch, Goal state, response authorship, confirmation, user-level
cancellation scope, immutable authorization, and end-to-end evidence
correlation.

The Soridormi Execution Coordinator owns how an authorized request runs on the
current platform: provider selection from declared capabilities, preparation,
resource compatibility, monotonic start, provider-local cancellation, stop,
recovery, and execution receipts. It may reject work, but it may not change its
semantic lane, vocal mode, content, Goal ownership, or confirmation state.

## Package versus process

The `soridormi_runtime` source package also contains optional MCP, evaluation,
policy packaging, and training support for repository convenience. That does
not make those dependencies part of the production execution process.

- execution runtime: no MuJoCo, robot SDK, sound-device, or desktop-viewer
  dependency;
- platform provider: MuJoCo or one qualified physical/desktop adapter and its
  device dependencies;
- training/evaluation: explicit optional dependencies;
- MCP: platform-neutral capability projection and validated runtime calls;
- hardware: target-specific implementation behind the private Platform
  Contract.

## Configuration ownership

Code defines behavior; versioned configuration defines platform structure.
Robot-specific actuator names, model paths, slices, limits, viewer settings,
audio-device identities, sensor mappings, and calibration belong under
Soridormi-owned platform configuration, not in Chromie or generic execution
logic.

## Capability layers

Current body surfaces remain authoritative until migrated:

```text
robot.*     body state and mode
safety.*    monitoring, stop, cancel, emergency stop
motion.*    bounded engineering motion plans
skill.*     named atomic body behaviors
activity.*  exact concurrent body-skill groups and per-member status
task.*      richer embodied contract, lifecycle, and body-task graph
```

The approved target adds platform-neutral execution families without freezing
the final wire names in this architecture document:

```text
vocal.*     speech, expressive speech, recitation, singing, humming, and other declared vocal modes
media.*     play, pause, resume, seek, stop, volume, and media status
platform.*  normalized input/output streams and read-only platform state
execution.* multimodal group preparation, start, cancellation, and evidence
```

The exact public schemas belong to their implementation Issues and API review.
No capability may expose raw joint, motor, torque, controller-array, device-
index, or SDK-specific authority to Chromie.

## Vocal and media semantics

Vocal output and media playback may share an audio mixer and speaker, but they
are different semantic and lifecycle contracts:

```text
"sing a song a cappella" -> Chromie Speaking lane -> vocal mode=singing
"play a song"            -> Chromie Activity lane -> media playback capability
```

Soridormi advertises the exact vocal modes supported by its configured provider.
Expressive TTS is evidence for expressive speech only; it must not advertise or
claim singing without mode-specific validation. If singing is unavailable,
Soridormi returns an exact unsupported outcome. It never substitutes ordinary
speech, media playback, blinking, or attention expression and calls that
singing.

## State and evidence authority

Body state, active motion, audio-stream state, media state, emergency stop,
provider health, and `safe_idle` are execution-runtime or platform facts. Task
and capability payloads project those facts; they do not derive them from
Chromie Goal state or task-local assumptions.

Plan creation, preview, and offline compilation are non-effectful. Effectful
execution remains behind explicit authorization, runtime calls, cancellation,
monitoring, and terminal evidence. Prepared state is not started state; started
state is not completed evidence.

## Platform-independence invariant

Simulation, desktop, and hardware providers expose the same high-level
execution semantics. Backend selection, feasibility, device identity, limits,
refusal, and hardware safety remain Soridormi-owned. Chromie does not lower a
Goal differently based on MuJoCo, sound-device, TTS-backend, robot-SDK, or
hardware implementation details.
