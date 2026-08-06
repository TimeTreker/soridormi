# Soridormi execution roadmap

The roadmap is gate-driven and semantic. Current completion and blockers live
in `docs/STATUS.md`. Each item below is an Issue-sized capability or boundary;
do not merge the queue into one rewrite or encode its order in runtime names.

## Retained foundations

- Shared Robot API and simulator/runtime separation
- Official policy parity and permanent comparison tooling
- Replaceable policy profiles and packaging
- Scenario-aware data, training, and closed-loop evaluation
- Bounded named skills and scripted interaction behaviors
- Resource-aware concurrent body activities with one final motor command
- MCP robot, safety, motion, skill, activity, and task surfaces
- No-motion embodied task lifecycle, graph, events, idempotency, and timeout
- Source-revision projection for paired evidence

These foundations remain open to regression fixes; retained does not mean
future changes may bypass their gates.

## Semantic contract integrity

Runtime payloads and rules describe behavior semantically, not through a
temporary project-sequence number.

Acceptance:

- no sequence label is required to interpret a capability or refusal;
- Chromie metadata remains advisory;
- raw controls, device indexes, and physical coordinates are rejected;
- active docs agree with runtime contracts.

## Body-state projection integrity

Body-wide state is reported from the runtime.

Acceptance:

- `safe_idle=false` while physical motion is active or emergency stop is set;
- task/capability payloads use the caller-provided runtime snapshot;
- a safe-idle task cannot complete while the body is not safe idle;
- source revision remains provenance, never authorization.

## Explicit target integrity

A target-oriented task carries an explicit structured target and direction.
Soridormi does not silently substitute a person, pose, coordinate, vocal mode,
media item, or user outcome.

Acceptance:

- missing target information fails closed with a structured planning failure;
- explicit labels are preserved as references;
- physical coordinates and device details remain Soridormi-owned.

## Chromie-Soridormi execution-boundary migration

### Vocal-mode semantic prerequisite

The companion Chromie repository first repairs the retained walk/sing/blink
failure at Goal and Planner boundaries. Singing and humming remain Speaking;
playing existing music remains Activity. Soridormi must receive separate exact
responsibilities and must never repair semantic meaning with provider guesses.

Gate:

- vocal lane and output mode are typed independently from provider need;
- vocal Goals cannot carry invented resource-acquisition contracts;
- no outcome references a nonexistent step;
- unsupported singing remains exact unavailable evidence rather than ordinary
  speech, media playback, or body substitution;
- the retained live episode reaches an exact execution request or an honest
  per-goal unavailable result.

### Immutable execution and evidence envelope

Define the public boundary carrying interaction, Plan, Goal, authorization,
confirmation, exact members, temporal relation, deadline, cancellation policy,
prepared/start lifecycle, fingerprints, and normalized per-member evidence.

Gate:

- Chromie owns semantic validation and authorization;
- Soridormi owns provider-local execution;
- no raw motor, audio-device, simulator, or SDK field crosses the boundary;
- replay and stale-plan rejection are deterministic;
- no current execution path is removed.

### Soridormi Execution Runtime facade

Route existing body compile, execute, cancel, recovery, and evidence paths
through the new envelope without changing behavior. Keep the current adapter
until body replay, cancellation, and live MuJoCo receipts are equivalent.

### Vocal and TTS execution

Add a platform-neutral Vocal Plan and provider-declared modes such as speech,
expressive speech, recitation, singing, humming, and nonverbal vocalization.
Support streaming, timing marks, interruption, prepared start, output delivery,
and mode-specific terminal evidence.

Gate:

- ordinary speech retains text, ordering, cancellation, and delivery behavior;
- expressive TTS does not advertise singing without mode-specific evidence;
- unsupported modes fail honestly;
- Soridormi cannot rewrite content or persona intent;
- Chromie no longer selects TTS backends or sound devices after migration.

### Media playback execution

Implement playback of existing music, recordings, streams, and sound effects as
Activity capabilities with play, pause, resume, seek, stop, volume, progress,
and terminal evidence. Vocal and media may share a mixer but not Goal,
cancellation, or completion semantics.

### Platform audio and sensor adaptation

Generalize the current simulator/backend boundary into one private Platform
Contract for MuJoCo, physical robot, or desktop platform. Move microphone,
speaker, camera, sensor, driver, calibration, and hardware-safety adaptation
behind that contract. Chromie consumes normalized streams and retains VAD, ASR,
Gateway, and user-level barge-in meaning.

### Provider-local multimodal coordination

Add prepared states, a monotonic start barrier, provider-declared compatibility,
per-member required/optional policy, group/member cancellation, measured start
skew, overlap, and terminal evidence for body, vocal, and media members.

Gate:

- requested simultaneity is never silently serialized and called exact;
- resource conflicts reject before effects begin;
- emergency stop and required safety preemption remain authoritative;
- synchronized or atomic claims require retained evidence.

### Boundary qualification and compatibility removal

Run exact-revision source, live-service, vocal, media, MuJoCo, cancellation,
device-loss, recovery, latency, and shared-accelerator evidence before deleting
legacy Chromie TTS/playback, dispatch, or platform adapters.

## Paired Chromie integration qualification

Verify provider discovery, immutable authorization, execution submission,
event monitoring, cancellation, completion wording, and paired source evidence
without weakening either repository's authority boundary.

## Monitored embodied execution

Design the body task executor only after sensing/planning dependencies are
declared. Start with task types that compile to already-qualified named skills.
Keep navigation, approach, manipulation, and delivery blocked until their
subsystems exist.

## Locomotion generalization

Use retained scenario gates and human visual inspection to improve clearance,
turning, transitions, rough terrain, and held-out behavior. Preserve rejected
candidate evidence under generated artifact directories.

## Hardware commissioning

Start only through the read-only, dry-run, safety, and operator-evidence
process. Hardware remains disabled by default.

## Repository gates

```bash
python scripts/validate_repository_governance.py
./scripts/validate_body_concurrency.sh
SORIDORMI_TASK_AGENT_USE_DOCKER=0 ./scripts/validate_task_agent_contract.sh
pytest -q
```

## Clearance qualification tools

```bash
./scripts/report_clearance_candidate_history.sh
./scripts/validate_clearance_engineering_process.sh
```

These commands summarize retained evidence and validate the offline
clearance process. They do not train, launch MuJoCo, or authorize hardware.
