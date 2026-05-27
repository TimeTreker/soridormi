# M6 summary: training and replacement-policy loop

M6 moved Soridormi from model-replacement infrastructure into a real replacement-policy learning loop.

The important contract is still:

```text
policy observation: 101 floats
policy action: 14 floats
runtime slot: high-level walking policy
```

The trained policy predicts high-level action offsets that are later mapped to target joint positions. It does not learn torques, motor currents, MuJoCo dynamics, or the low-level controller.

## What M6 completed

### M6.1: training dataset export

Runtime logs can be converted into supervised policy samples:

```text
obs[101] -> action[14]
```

Typical command:

```bash
./scripts/export_training_dataset.sh data/logs/policy_open_duck_forward_*.mcap
```

The dataset keeps policy observations, policy actions, raw actions when available, motor commands, compact robot state, next-state summaries, policy command metadata, and debug metadata.

### M6.2: dataset preparation

Exported datasets can be validated and split deterministically into train/val/test sets.

Typical command:

```bash
./scripts/prepare_training_dataset.sh \
  data/training_datasets/open_duck_forward_supervised.jsonl \
  --output-dir data/training_datasets/open_duck_forward_prepared
```

### M6.3: dataset statistics and normalization

Prepared datasets can produce statistics and normalization artifacts.

Typical command:

```bash
./scripts/summarize_training_dataset.sh \
  data/training_datasets/open_duck_forward_prepared
```

The normalization artifact is computed from the train split only.

### M6.4: linear behavior-cloning baseline

A deterministic NumPy ridge-regression baseline can be trained as a smoke test:

```bash
./scripts/train_behavior_clone.sh \
  data/training_datasets/open_duck_forward_prepared \
  --output-dir data/training_runs/open_duck_forward_linear_bc
```

This baseline proved the dataset path could produce a learned artifact.

### M6.5: linear BC runtime profile

The linear baseline can run through the same Soridormi runtime loop using a profile with:

```yaml
model:
  kind: linear_behavior_clone
```

This kept the observation builder, action mapper, controller loop, and logging path unchanged.

### M6.6: offline policy evaluation

Runtime profiles can be evaluated against prepared supervised datasets before running MuJoCo.

Typical command:

```bash
./scripts/evaluate_policy_profile.sh <profile> \
  data/training_datasets/open_duck_forward_prepared \
  --max-test-mae 0.05
```

### M6.7: end-to-end training pipeline runner

The pipeline runner can orchestrate export, prepare, summarize, train, profile creation, evaluation, acceptance, and packaging for a candidate. This is useful, but it should not replace understanding each backbone step.

### M6.8: candidate leaderboard

Multiple evaluated candidates can be ranked by held-out metrics.

### M6.9: candidate promotion

A promotable candidate can be copied into a named runtime profile with an audit record.

### M6.10: bounded rollout smoke

Runtime can now stop automatically after finite steps or seconds:

```bash
./scripts/run_policy_rollout_smoke.sh <profile> --steps 1000
```

This is important because replacement policies should be tested in bounded MuJoCo rollouts before longer experiments.

### M6.11: bounded rollout acceptance

A bounded rollout log can be turned into pass/fail metrics such as duration, resets, action magnitude, joint magnitude, and base displacement.

### M6.12: neural behavior-cloning trainer and ONNX export

Soridormi can now train a real neural replacement policy and export it as ONNX.

Typical command:

```bash
./scripts/train_neural_behavior_clone.sh \
  data/training_datasets/open_duck_forward_prepared \
  --output-dir data/training_runs/neural_bc_open_duck \
  --profile-name neural_bc_open_duck \
  --epochs 50 \
  --hidden-sizes 256,256 \
  --device cuda \
  --force-profile
```

Expected artifacts:

```text
data/training_runs/neural_bc_open_duck/neural_behavior_clone.pt
data/training_runs/neural_bc_open_duck/neural_behavior_clone.onnx
configs/policies/neural_bc_open_duck.yaml
```

The ONNX export accepts a dynamic batch dimension, but the runtime contract remains one 101D observation producing one 14D action per policy step.

### M6.13: teacher-vs-candidate rollout comparison

Actual MuJoCo rollouts can be compared between the trusted teacher and a candidate policy.

Typical workflow:

```bash
./scripts/run_policy_rollout_smoke.sh open_duck_forward --steps 1000
teacher="$(ls -t data/logs/policy_open_duck_forward_*.mcap | head -1)"

./scripts/run_policy_rollout_smoke.sh neural_bc_open_duck --steps 1000
candidate="$(ls -t data/logs/policy_neural_bc_open_duck_*.mcap | head -1)"

./scripts/compare_policy_rollouts.sh "$teacher" "$candidate"
```

This is more important than offline MAE because it measures closed-loop behavior.

### M6.14: rollout failure diagnosis

Rollout comparison output can be classified into failure modes:

```text
stability_or_fall
early_termination
weak_forward_locomotion
slow_forward_velocity_tracking
lateral_drift
action_saturation
action_amplification
```

### M6.15: DAgger-style relabeling

Candidate rollout observations can be relabeled by the teacher policy. This captures states that the candidate actually visits, not only the teacher's successful trajectory distribution.

Typical command:

```bash
./scripts/relabel_policy_rollout.sh \
  data/logs/policy_neural_bc_open_duck_*.mcap \
  --teacher-profile open_duck_forward \
  --output data/training_datasets/neural_bc_dagger_relabel.jsonl \
  --require-provider CUDAExecutionProvider
```

### M6.16: retrain and promote iteration

The iteration command connects failure rollout, relabeling, dataset merge, retraining, offline evaluation, and optional promotion.

Typical command:

```bash
./scripts/run_policy_iteration.sh neural_bc_open_duck_iter1 \
  --candidate-log data/logs/policy_neural_bc_open_duck_*.mcap \
  --base-dataset data/training_datasets/open_duck_forward_supervised.jsonl \
  --teacher-profile open_duck_forward \
  --output-root data/policy_iterations/neural_bc_open_duck_iter1 \
  --epochs 50 \
  --hidden-sizes 256,256 \
  --device cuda \
  --max-test-mae 0.01 \
  --require-provider CUDAExecutionProvider \
  --promote-to neural_bc_open_duck_best \
  --force-profile \
  --force-promote
```

## Current M6 exit criteria

M6 is complete when the repo can do the following from data and code already in the project:

```text
export teacher rollout data
prepare and summarize supervised dataset
train neural ONNX replacement policy
create runtime profile
run bounded MuJoCo rollout
compare candidate rollout against teacher rollout
diagnose failure
relabel candidate states with teacher policy
retrain and promote a better candidate
```

That loop is now implemented.

## Lessons from M6

1. Offline action MAE is useful but not sufficient. Closed-loop rollout comparison is the real test.
2. The policy slot is `obs[101] -> action[14]`; do not confuse it with torque control.
3. More helper scripts are not the project backbone. The backbone is train, deploy, roll out, compare, improve.
4. DAgger-style relabeling is the natural next improvement once a cloned policy drifts from teacher states.
5. M7 should not wait for a perfect learned policy; hardware backend work can start while M6 candidates continue improving in sim.

## Recommended next milestone

Move to M7: hardware bridge.

Start with read-only and dry-run hardware plumbing, not walking immediately.

Suggested first M7 sections:

```text
M7.1 hardware backend interface plan and safety contract
M7.2 Open Duck Mini hardware backend skeleton
M7.3 read-only hardware state streaming into RobotState
M7.4 motor command dry-run sink with logs
M7.5 safety limits, watchdog, emergency stop plumbing
```
