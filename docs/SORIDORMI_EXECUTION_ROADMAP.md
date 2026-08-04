# Soridormi execution roadmap

The roadmap is gate-driven and semantic. Current completion and blockers live
in `docs/STATUS.md`.

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
- raw controls and physical coordinates are rejected;
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
Soridormi does not silently substitute a person, pose, or coordinate.

Acceptance:

- missing target information fails closed with a structured planning failure;
- explicit labels are preserved as references;
- physical coordinates remain Soridormi-owned.

## Concurrent cognitive and embodied execution

Preserve one Cognitive Core with Social-Attention Proposal, Speaking Execution,
and Activity Execution lanes. The Soridormi provider executes exact compatible
body members through physical resource arbitration and final command
composition.

Acceptance:

- social attention remains proposal-only;
- speaking and activity are peer execution lanes;
- speech is not a Soridormi body member;
- at most one primary locomotion member is accepted;
- head/gaze overlays stay inside declared envelopes;
- visual expressions do not write motor commands;
- interaction cancellation is propagated by Chromie;
- physical safety preemption remains independent and Soridormi-owned;
- per-member and aggregate outcomes prevent unsupported completion claims.

## Paired Chromie integration qualification

Verify provider discovery, proposal metadata, task preview/submit, event
monitoring, cancellation, completion wording, and paired source evidence without
weakening either repository's authority boundary.

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
