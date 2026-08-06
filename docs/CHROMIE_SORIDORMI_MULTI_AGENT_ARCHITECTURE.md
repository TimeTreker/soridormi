# Chromie and Soridormi multi-agent architecture

This document records the agreed brain/execution/platform split between Chromie
and Soridormi. Both systems may use coordinators or DAG engines, but they operate
at different scopes and share only immutable contracts.

## Core agreement

Chromie has one authoritative Cognitive Core with three semantic coordination
lanes:

```text
Social-Attention Proposal Lane
Speaking Lane
Activity Lane
```

The lanes are not independent minds. The Cognitive Core owns user meaning,
Goal Association, Goal lifecycle, planning, personality, authored
communication, vocal mode, and user-level temporal intent.

The Chromie Interaction Orchestrator remains in Chromie. It owns session and
turn lifecycle, Gateway/Core dispatch, confirmation, user-level cancellation
scope, immutable authorization, and end-to-end evidence correlation. It must not
become a device or provider scheduler.

Soridormi is the platform execution agent. It owns provider-local capability
validation, preparation, resource arbitration, execution, timing, cancellation,
recovery, and normalized evidence for body, vocal, media, sensor, and device
work. It may reject a request but cannot reinterpret a Goal, rewrite authored
content, change vocal mode, or widen authorization.

## Two-container Soridormi target

```text
Chromie Interaction Orchestrator
  -> immutable platform-neutral execution envelope
Soridormi Execution Runtime container
  -> stable capabilities, resources, prepare/start/cancel, recovery, evidence
Soridormi Platform Provider container
  -> MuJoCo OR physical robot OR desktop platform, devices, drivers, safety
```

The Platform Contract is private to Soridormi. Chromie must not depend on
MuJoCo, robot SDKs, joint arrays, controller identities, ALSA/PulseAudio,
sound-device indexes, TTS backend names, camera SDKs, or calibration data.

The current implementation is a migration baseline, not the target: Soridormi
already separates runtime from simulator for body execution, while Chromie still
owns TTS/playback and cross-provider coordination. Each responsibility moves
only after equivalent tests, cancellation, and target evidence exist.

## Brain and execution DAG scopes

### Chromie global DAG

Chromie owns the human-facing and multi-capability DAG. Example:

```text
understand request
  -> preserve independent Goals
  -> ask for missing context or confirmation
  -> authorize platform-neutral execution members
  -> submit immutable group to Soridormi or a peer information provider
  -> monitor exact evidence
  -> reconcile every Goal
  -> compose truthful response
```

This DAG may call memory, external information, user confirmation, and
Soridormi. It never compiles motor commands, audio devices, or platform-specific
work.

### Soridormi execution DAG

Soridormi owns the provider-facing execution DAG:

```text
validate envelope and capability support
  -> validate resources and safety
  -> prepare required members
  -> establish provider-local start relation
  -> execute and monitor
  -> stop, recover, degrade according to authorized policy, or continue
  -> return per-member and aggregate evidence
```

Body subgraphs may contain localization, gait, controllers, and recovery.
Vocal subgraphs may contain TTS or singing providers, streaming, timing marks,
and audio output. Media subgraphs may contain playback, seek, pause, and mixer
operations. None of these subgraphs author user meaning.

## Capability surfaces

Retained current body surfaces remain:

```text
soridormi.robot.*
soridormi.safety.*
soridormi.motion.*
soridormi.skill.*
soridormi.activity.*
soridormi.task.*
```

The migration will add reviewed platform-neutral surfaces for:

```text
vocal rendering and vocal-mode support
media playback and media state
normalized platform input/output streams
multimodal execution groups and evidence
```

The final names and schemas belong to implementation Issues and API review.
Soridormi capability declarations, not names or user phrases, are authoritative
for supported modes, resources, concurrency, interruption, and evidence.

## Singing, TTS, and media

Singing and speaking are both generated vocal output, but they are distinct
vocal modes. A provider may support speech and expressive speech without
supporting stable singing. Soridormi advertises the supported subset; Chromie
may claim completion only from matching mode-specific evidence.

```text
"sing a song a cappella" -> Speaking -> vocal mode=singing
"recite this"            -> Speaking -> vocal mode=recitation
"play a song"            -> Activity -> media playback
```

Playing music is a capability execution, not TTS. Vocal and media providers may
share a mixer and speaker, but keep distinct Goal, progress, cancellation, and
completion semantics.

For a request such as walking while singing and blinking:

```text
Chromie:
  creates separate body, vocal, and expression responsibilities
  preserves the requested parallel relation
  authorizes exact capabilities without platform details

Soridormi:
  verifies body/vocal/expression support and resources
  prepares compatible members
  starts them according to its declared timing contract
  returns per-member evidence
```

If singing is unavailable, neither Chromie nor Soridormi may substitute ordinary
speech, media playback, blinking, or attention expression and claim that singing
occurred. Any alternative is explicit, confirmation-bound when material, and
preserves independent Goal outcomes.

## Context ownership

Chromie maintains:

- user intent, conversation history, Goal meaning, and Goal lifecycle;
- personality and response authorship;
- vocal mode and requested temporal relation;
- clarifications, confirmations, and user-level cancellation scope;
- cross-capability orchestration and end-to-end evidence correlation.

Soridormi Execution Runtime maintains:

- capability availability and provider declarations;
- provider-local execution groups, resources, preparation, timing, and state;
- body, vocal, media, platform-perception, and device execution telemetry;
- per-member cancellation, recovery, failure, and evidence;
- normalized runtime and platform health.

Soridormi Platform Provider maintains:

- simulator or hardware state;
- controllers, drivers, audio and sensor devices;
- calibration, state estimation, and hardware safety;
- platform-specific identities and limits.

## Promotion rule

No new capability becomes executable merely because Chromie can ask for it.
Soridormi promotion requires:

- a declared semantic capability and supported-mode contract;
- bounded parameters, resources, interruption, and refusal conditions;
- simulator or platform validation appropriate to the capability;
- cancellation, failure, restart, and terminal evidence;
- explicit unsupported status for missing platform abilities;
- no exposure of raw natural language, joint targets, motor commands, torque
  commands, device indexes, SDK objects, or provider-private payloads to
  Chromie.

Centralized deployment is allowed. Centralized semantic and safety authority is
not: co-location may share compute and devices, but the Chromie/Soridormi and
runtime/platform contracts remain explicit.
