# Soridormi project SOP

This SOP defines the durable engineering loop. Current capability and blocker
status lives only in `docs/STATUS.md`.

## Target and ownership

Soridormi is the Open Duck Mini v2 body/cerebellum runtime. Chromie owns
conversation, memory, user-facing reasoning, confirmation, and global task
orchestration. Soridormi owns robot state, body feasibility, skill lowering,
locomotion, safety, execution, monitoring, recovery, and backend selection.

```text
Same RobotState and MotorCommand contracts.
Same body-skill semantics.
Different simulator or hardware backend.
```

Chromie never authorizes raw joints, motors, torques, physical coordinates, or
`action_14d`. Soridormi may refuse every request.

## Trusted baseline

Run the official Open Duck policy as the permanent teacher/reference. Retain
the official baseline, replay, parity, first-divergence, and trace-comparison
tools. Later policies may replace execution, not baseline evidence.

## Parity qualification

Reproduce observation construction, command and gait-phase handling, policy
inference, action mapping, loop order, reset behavior, and closed-loop rollout
through Soridormi's own API/runtime/backend contracts.

A tuning change does not close an unexplained parity failure.

## Closed-loop evaluation

Promotion requires MuJoCo behavior, not only offline error. Evaluate fall/reset
state, displacement, speed tracking, drift, contacts, clearance, action/joint
ranges, stability, interruption, and safe-idle recovery as relevant.

## Data and policy contracts

The official teacher baseline uses:

```text
observation[101] -> action_14d
```

Command-conditioned policy profiles use:

```text
observation[101] + desired_command[3] -> action_14d
```

Future task, environment, or history features require an explicit versioned
training/runtime contract. A model may run only when profile, input shape,
feature ordering, and runtime producer match.

Raw natural language and raw perception never enter the low-level action policy.

## Training and replacement loop

```text
collect qualified teacher rollouts
validate scenario and distribution coverage
split without rollout leakage
train a candidate
validate the model/profile contract
run bounded MuJoCo evaluation
compare with the retained reference
diagnose a named failure
retain, reject, or iterate with evidence
```

Offline loss is diagnostic, never a promotion gate.

## Skill boundary

Externally callable body behavior is a bounded named skill or structured body
context. Soridormi validates availability, parameters, current state, safety,
interruptibility, and backend support before execution.

## Task boundary

The task API accepts richer structured embodied goals and returns Soridormi's
body interpretation, lifecycle, blocked subsystems, events, and routing hints.
It is currently no-motion. Supported tasks may compile to named-skill dry runs;
physical execution remains on the validated skill/motion path until a monitored
task executor is qualified.

Task records project live body state. They do not invent targets or infer
`safe_idle` from the absence of emergency stop alone.

## Hardware bridge

Hardware work is gated and fail-closed:

```text
read-only state
command dry run
limits and watchdog
independent stop
low-power single-joint test
standing
tethered low-speed motion
broader qualification
```

No simulator result silently authorizes hardware.

## Patch and validation

Deliver plain git patches unless the user requests another format. Every patch
must include integrity checks and scope-appropriate functional validation.

```bash
git apply --check ~/Downloads/<patch>.patch
git apply ~/Downloads/<patch>.patch
python scripts/validate_repository_governance.py
```

See `docs/PATCH_DELIVERY_AND_VALIDATION.md`.
