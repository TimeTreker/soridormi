# Soridormi project SOP

This document defines the project backbone. Future work should first strengthen this path; helper scripts and reports are secondary unless they directly support one of these steps.

## SOP-0: Target definition

Goal: Open Duck Mini v2 walks reliably in MuJoCo first, then the same Soridormi runtime controls hardware through a hardware backend.

Core invariant:

```text
Same runtime.
Same policy interface.
Same RobotState.
Same MotorCommand.
Different backend.
```

## SOP-1: Official baseline

Run the official Open Duck policy as the trusted teacher. Record reference traces and rollouts.

Outputs:

```text
data/official_baseline/*.trace.jsonl
data/logs/policy_open_duck_forward_*.mcap
```

## SOP-2: Soridormi parity

Run the same policy through Soridormi. Match observation, action, motor target history, loop order, reset state, and rollout state against the official baseline.

Exit criteria:

```text
first-divergence analyzer reports no meaningful mismatch
history offsets match official loop order
same policy contract is used by sim and future hardware
```

## SOP-3: Rollout evaluation

Evaluate whether a policy actually walks in MuJoCo, not just whether its offline action error is low.

Primary metrics:

```text
rollout duration
reset/fall count
forward displacement
forward speed
lateral drift
yaw drift
action magnitude
joint magnitude
contacts / obvious instability
```

## SOP-4: Data collection

Export supervised training data from trusted successful rollouts.

Contract:

```text
observation: 101 floats
action: 14 floats
sample: obs -> policy action
```

## SOP-5: Policy training

Train replacement policies for the same high-level runtime slot:

```text
obs[101] -> continuous_actions[14]
```

This is not torque learning, not a MuJoCo dynamics model, and not a replacement for the low-level position controller.

## SOP-6: Policy deployment

Export trained policies into runtime-compatible profiles and model artifacts. The runtime should not need special training-only code paths.

Minimum deployment check:

```text
./scripts/check_policy_model.sh --profile <profile>
./scripts/run_policy_rollout_smoke.sh <profile> --steps <N>
```

## SOP-7: Policy improvement loop

Use rollout failures to improve the policy distribution.

Loop:

```text
train candidate
run candidate in MuJoCo
compare against teacher rollout
diagnose failure mode
relabel candidate states with teacher policy
merge dataset
retrain
promote better candidate
```

## SOP-8: Hardware bridge

Implement the real robot backend while keeping the existing runtime, policy, observation, and command contracts.

Phases:

```text
read-only hardware state streaming
motor command dry-run
limits and watchdog
single-joint low-power test
standing pose
low-speed tethered walk
```

## SOP-9: Hardware safety and staged rollout

Before walking on hardware, add hard safety boundaries:

```text
emergency stop
joint limits
velocity/current/torque limits
command timeout
watchdog heartbeat
thermal / voltage checks
operator checklist
log everything
```

## SOP-10: Patch delivery and validation

Future LLM sessions must deliver plain `.patch` files unless the user asks for another format. The user normally downloads patches to `~/Downloads`, so user-facing commands should use that path.

Every patch response must include:

```text
1. Patch integrity check: git apply --check ~/Downloads/<patch>.patch
2. Functional validation: tests, CLI smoke checks, sim commands, or docs sanity checks that prove the patch behavior
```

For docs-only changes, functional validation is still required: check that the expected files/sections exist and that Markdown fences are balanced. For code changes, run the relevant unit tests and compile checks. For sim or training changes, separate local/unit validation from live MuJoCo validation.

See `docs/PATCH_DELIVERY_AND_VALIDATION.md`.
