# M6 forward-walk simulation pass report

Status: **passed for the forward-walk smoke condition**.

This document records the first real M6 simulation success after the residual fine-tuning loop.
It should be read as a milestone report, not as a claim that the policy is fully robust across all commands or ready for untethered hardware.

## Tested condition

```text
teacher profile: open_duck_forward
candidate profile: residual_open_duck
command: vx=0.15, vy=0.0, yaw=0.0
rollout limit: 1000 policy steps
control rate: 50 Hz
sim duration: about 20 seconds
provider requirement: CUDAExecutionProvider
```

## Result

```text
Result: PASS
```

Both profiles completed the bounded rollout:

```text
policy records: 1000 / 1000
robot duration: 19.98 s / 19.98 s
reset count: 0 / 0
```

The residual policy improved the forward-walk rollout metrics in this test:

```text
forward displacement:
  open_duck_forward:     1.87665 m
  residual_open_duck:    1.99915 m
  candidate/reference:   1.06528

forward speed:
  open_duck_forward:     0.0939264 m/s
  residual_open_duck:    0.100058 m/s
  candidate/reference:   1.06528

lateral absolute drift:
  open_duck_forward:     0.353745 m
  residual_open_duck:    0.0291078 m
  candidate/reference:   0.0822846

action abs max:
  open_duck_forward:     0.960559
  residual_open_duck:    0.985185
```

The candidate stayed close to the teacher action scale while greatly reducing lateral drift in this run.

## What this proves

M6 is now complete enough for the **forward-walk simulation milestone**:

```text
teacher baseline
→ residual fine-tuning reward
→ residual ONNX policy
→ runtime profile
→ bounded MuJoCo rollout
→ default-vs-fine-tuned rollout comparison
→ PASS
```

This is the first Soridormi policy-improvement loop that improves beyond simply cloning the default policy.

## What this does not prove yet

This result does **not** prove final sim robustness. Before hardware walking, test:

```text
longer rollouts
multiple random seeds / repeated resets
forward speed grid
positive/negative yaw commands
lateral velocity commands
start/stop transitions
standing recovery and stop safety
viewer inspection for foot slip or unstable gait
```

Do not move directly from this result to aggressive real-robot walking.

## Recommended status wording

Use this exact wording in docs and handoffs:

```text
M6 is complete enough in simulation for the forward-walk case. The residual policy exports to ONNX, passes CUDA model validation, completes a 1000-step MuJoCo rollout with zero resets, and improves forward displacement/speed plus lateral drift versus the default policy in the tested vx=0.15 condition. Broader command-grid validation remains required before hardware walking.
```
