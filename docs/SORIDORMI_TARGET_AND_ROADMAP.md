# Soridormi target and capability roadmap

This document defines the durable target and semantic capability sequence.
Current status lives in `docs/STATUS.md`; candidate metrics live in generated
evidence.

## Target

Build a reusable Open Duck Mini v2 body/cerebellum runtime that:

- preserves a trusted official-policy baseline;
- runs the same body contracts against MuJoCo and future hardware;
- exposes bounded skills and structured embodied tasks to Chromie;
- validates, monitors, interrupts, recovers, and refuses independently;
- supports replaceable policies and reproducible training/evaluation;
- never exposes raw low-level body control as LLM authority; and
- remains an embodied body/cerebellum provider unless retained evidence proves
  that a broader platform-execution boundary is required.

## Runtime and parity foundation

Maintain stable Robot API, observation, action, controller, profile, logging,
replay, and official parity contracts.

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

Gate: speech, singing, TTS, and playback remain Chromie-owned or peer
capabilities outside Soridormi; one writer owns each physical resource; one
final Soridormi motor-command authority exists; incompatible body activity fails
closed; cancellation and emergency stop restore safe physical state.

## Boundary-change admission gate

Backend portability alone does not justify moving vocal, media, microphone, or
speaker execution into Soridormi. Chromie's current TTS provider contract
already permits backend replacement without changing the cognitive or body
boundary, and exact provider-prefixed capability identity must be preserved.

A broader Soridormi hosting proposal may enter active work only when retained
evidence identifies a concrete blocker such as frame-accurate embodied lip-sync,
measured cross-provider start skew, cancel-to-silence regression, unavoidable
platform-adaptation leakage, or unresolved shared-resource contention. The
proposal must compare the existing peer-provider design, define latency and
cancellation budgets, preserve rollback, and update all contributor and machine
contracts together. Architecture analogy or future deployment elegance alone is
not an admission gate.

## Embodied task contract

Accept richer structured goals, keep a Soridormi-owned body-task lifecycle and
graph, expose blocked subsystems and events, and fail closed for missing
capability.

Gate: contract behavior is deterministic, retry-safe, timeout-safe, no-motion,
and paired with Chromie integration tests.

## Monitored task execution

Add physical task execution only after sensing, planning, skill selection,
monitoring, cancellation, recovery, and completion evidence exist.

Gate: a task cannot claim completion from preview, dry run, partial execution,
or unsupported fallback.

## Generalization and control improvement

Expand scenario coverage, context features, clearance/stability, recovery,
rough terrain, turning, lateral behavior, and held-out evaluation. Add WBC,
residual, or adaptive control only behind explicit contracts and evidence.

Gate: retained candidates beat the reference on named objectives without safety
or generalization regression.

## Hardware commissioning

Implement a hardware backend through read-only state, dry-run commands, limits,
watchdog, independent stop, low-power tests, standing, tethered movement, and
broader validation.

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
