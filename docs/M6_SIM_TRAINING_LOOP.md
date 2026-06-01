# M6 Simulation Training Loop

M6 is not complete just because the simulator can run a policy. Treat M6 as complete only after a candidate policy has been trained, exported, validated, and compared in MuJoCo against the default teacher.

This patch adds the direct training path needed before hardware work:

```text
MuJoCo sim server
→ teacher policy rollout
→ supervised JSONL dataset with scenario/rollout IDs
→ grouped train/val/test split by rollout/scenario
→ neural behavior-clone ONNX/profile
→ model contract check
→ MuJoCo rollout comparison
```


## 0. Current priority: evaluate commanded walking first

The current main branch already has teacher collection, command-grid arguments, grouped dataset splitting, neural BC export, rollout comparison, residual policy scaffolding, and walking reward code. The next Soridormi milestone is therefore not another interface layer; it is proving command-conditioned walking in MuJoCo.

Before collecting a large dataset, run or add a commanded-walk evaluation suite that measures the default teacher across:

```text
stop
slow/medium/fast forward
turn left/right in place
forward curves left/right
small lateral commands
short command-switching sequences
```

Use the results to decide which command regions are safe for data collection and which need smaller ramps, shorter durations, or exclusion from the first BC dataset.

## 1. Start the simulator

In one terminal:

```bash
./scripts/run_sim_server.sh
```

Use the normal MuJoCo viewer/debug scripts if you need to inspect stability before collecting data.

## 2. Collect teacher-policy data directly from sim

In another terminal:

```bash
./scripts/collect_teacher_dataset.sh \
  --profile open_duck_forward \
  --output data/training_datasets/open_duck_forward_teacher_live.jsonl \
  --episodes 4 \
  --steps-per-episode 1000 \
  --command-x 0.15
```

The output is already a `soridormi.policy_supervision.v1` JSONL dataset with `observation[101]` and `action[14]`. It also includes `scenario_id`, `rollout_id`, `command_id`, and `command_index` metadata so the prepare step can avoid splitting adjacent timesteps across train/val/test. Increase episodes and vary commands once the smoke path works.

Recommended real collection grid:

```text
x velocity: 0.00, 0.05, 0.10, 0.15, 0.20
lateral y:  -0.05, 0.00, 0.05
yaw:        -0.20, 0.00, 0.20
```

You can collect a small command grid directly:

```bash
./scripts/collect_teacher_dataset.sh \
  --profile open_duck_forward \
  --output data/training_datasets/open_duck_forward_teacher_grid.jsonl \
  --episodes 2 \
  --steps-per-episode 600 \
  --command-x-values 0.00,0.05,0.10,0.15 \
  --command-y-values=-0.03,0.00,0.03 \
  --command-yaw-values=-0.15,0.00,0.15
```

Collect small grids first. Do not train from one perfect short rollout and expect robust walking.

## 3. Train a teacher behavior-clone policy

Fast one-command path:

```bash
./scripts/run_teacher_policy_training.sh \
  --profile open_duck_forward \
  --candidate neural_bc_teacher_live \
  --output-root data/training_pipelines/neural_bc_teacher_live \
  --episodes 4 \
  --steps-per-episode 1000 \
  --command-x 0.15 \
  --force-profile
```

For a small command-grid BC candidate, use comma-separated command values:

```bash
./scripts/run_teacher_policy_training.sh \
  --profile open_duck_forward \
  --candidate neural_bc_teacher_grid \
  --output-root data/training_pipelines/neural_bc_teacher_grid \
  --episodes 2 \
  --steps-per-episode 600 \
  --command-x-values 0.00,0.05,0.10,0.15 \
  --command-yaw-values=-0.15,0.00,0.15 \
  --split-group-field source_log \
  --force-profile
```

Equivalent manual path:

```bash
./scripts/prepare_training_dataset.sh \
  data/training_datasets/open_duck_forward_teacher_live.jsonl \
  --output-dir data/training_pipelines/neural_bc_teacher_live/prepared \
  --seed 123 \
  --split-group-field source_log

./scripts/summarize_training_dataset.sh \
  data/training_pipelines/neural_bc_teacher_live/prepared

./scripts/train_neural_behavior_clone.sh \
  data/training_pipelines/neural_bc_teacher_live/prepared \
  --output-dir data/training_pipelines/neural_bc_teacher_live/neural_bc \
  --profile-name neural_bc_teacher_live \
  --profile-template open_duck_forward \
  --force-profile

./scripts/check_policy_model.sh --profile neural_bc_teacher_live
```

## 4. Prevent leakage in train/val/test splits

Do not randomly split individual timesteps for real evaluation. Neighboring timesteps from one rollout are almost identical, so random sample-level splitting can make validation loss look excellent while the closed-loop policy still fails. For teacher rollout datasets, split by whole rollout:

```bash
./scripts/prepare_training_dataset.sh \
  data/training_datasets/open_duck_forward_teacher_live.jsonl \
  --output-dir data/training_pipelines/neural_bc_teacher_live/prepared \
  --seed 123 \
  --split-group-field source_log
```

Use `--split-group-field scenario_id` when you want stronger generalization testing across command scenarios. That may require more command scenarios so validation/test are not empty. Keep `source_log` for the first smoke run because it holds out whole episodes without holding out every sample of a command.

## 5. Run the trained candidate in MuJoCo

```bash
./scripts/run_policy_experiment.sh neural_bc_teacher_live --steps 1000
```

Then compare it with the default teacher:

```bash
./scripts/compare_policy_rollouts.sh \
  data/rollouts/open_duck_forward_latest.jsonl \
  data/rollouts/neural_bc_teacher_live_latest.jsonl
```

Use the exact latest rollout files produced by your run scripts.

## 6. RL / residual policy path

Behavior cloning copies the teacher. It is useful for proving the data/model/deploy path, but it should not be expected to improve beyond the teacher.

For improvement, use the residual path after the teacher BC loop is validated:

```bash
./scripts/train_residual_policy.sh open_duck_forward \
  --output-dir data/rl_finetune/residual_open_duck \
  --profile-name residual_open_duck \
  --iterations 5 \
  --population 16 \
  --steps-per-episode 300 \
  --residual-scale 0.05 \
  --force-profile

./scripts/check_policy_model.sh --profile residual_open_duck

./scripts/run_residual_finetune_comparison.sh residual_open_duck \
  --teacher-profile open_duck_forward \
  --steps 1000
```

The current residual trainer is a safe cross-entropy search over a bounded constant residual. It is a training scaffold, not full PPO/SAC yet. Replace it with a larger residual actor only after the teacher dataset loop and rollout comparison are reliable.

## M6 completion gate

Do not start hardware walking until all of these pass:

1. Teacher/default profile runs in MuJoCo.
2. A collected teacher dataset validates and splits.
3. A neural BC candidate exports ONNX and passes `check_policy_model.sh`.
4. The candidate runs in MuJoCo for a bounded rollout.
5. Default-vs-candidate comparison is generated.
6. Residual RL candidate either improves metrics or is explicitly rejected.


## Commanded free-walk acceptance before hardware

Treat any candidate as experimental until it passes a command-suite rollout comparison. A candidate must survive fixed commands and conservative command-switching scenarios before hardware work resumes. A good first acceptance report should include per-scenario pass/fail, survival time, termination reason, displacement, velocity tracking, lateral drift, yaw tracking, upright/height error, and action smoothness.

See `docs/SORIDORMI_FREE_WALK_PLAN.md` for the updated Soridormi-first roadmap.
