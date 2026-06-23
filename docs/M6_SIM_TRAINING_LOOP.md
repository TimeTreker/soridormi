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


## Policy context contract

The training target is a context-conditioned policy, not a single fixed-speed walking demo. The long-term contract is documented in:

```text
docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md
```

Near-term M6 remains simple: robot observation plus continuous velocity command should produce the 14D action. Future stages should add task context and environment context without changing the safety boundary:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Teacher data should therefore record more than observation/action pairs whenever possible. Keep `scenario_id`, rollout grouping, applied/target commands, command ramp metadata, terrain labels, and future skill/task labels so BC, closed-loop evaluation, and residual/RL can debug failures by scenario rather than by file name.


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

In one terminal, start the MuJoCo backend explicitly. The default functional-test mode is headless/no-viewer:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

For visual inspection, run the same backend with the passive viewer enabled:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

Use the viewer only as an inspection aid; the headless MuJoCo command should remain the default validation command.

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

### Random command-sequence collection

For freer walking behavior, collect piecewise random command sequences after the fixed command suite passes. Keep the MuJoCo backend explicit. The random collector owns its MuJoCo collection lifecycle, so do not start a separate `run_sim_server.sh` for this command. Use `--viewer` on the collector command for visual inspection.

```bash
./scripts/collect_random_teacher_dataset.sh \
  --profile open_duck_forward \
  --output data/teacher_random_walk/dataset.jsonl \
  --episodes 100 \
  --steps-per-episode 800 \
  --vx-range -0.03,0.15 \
  --vy-range -0.03,0.03 \
  --yaw-range -0.20,0.20 \
  --command-hold-steps 80,250 \
  --command-ramp-steps 20 \
  --backend mujoco \
  --viewer \
  --json | python -m json.tool
```

Negative range values are valid in either shell style, so both `--vx-range -0.03,0.15` and `--vx-range=-0.03,0.15` are supported. The collector ramps each new target command over `--command-ramp-steps` control steps by default. Keep this nonzero for continuous-speed BC datasets, because the robot should learn smooth speed changes such as slow -> normal -> fast -> stop, not only abrupt command jumps. Use `--command-ramp-steps 0` only for explicit step-response debugging.

The collector records both the applied command and the target command plus command segment/ramp metadata so later training/evaluation can distinguish fixed-command grids from random continuous command-transition data. Continue only if the JSON result reports `ok: true` and a positive `sample_count`.

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

The residual trainer supports three bounded cross-entropy-search actors:

- `--actor-kind constant`: the original 14D residual bias scaffold.
- `--actor-kind phase_contact`: a 70-parameter linear actor over bias, left/right
  foot contact, and cosine/sine imitation phase, followed by `tanh`.
- `--actor-kind command_state`: a 120-parameter sagittal-leg actor over desired
  velocity, contacts, gait phase, hip/knee/ankle offsets, and previous
  hip/knee/ankle actions.
- `--actor-kind command_state_mlp`: a nonlinear four-hidden-unit actor with the
  same feature/output contract and a linear skip path.

The residual trainer can be scored across multiple velocity conditions. Add an
optional fourth comma-separated value to weight harder commands, which is useful
when the next M10 run should emphasize start/stop and turning clearance without
dropping the near-passing flat-walk case:

```bash
./scripts/train_residual_policy.sh context_stage1_three_scenario_10ep_e80 \
  --actor-kind command_state_mlp \
  --training-command 0.125,0,0,1.0 \
  --training-command 0.06,0,0,2.0 \
  --training-command 0.09,0,0.12,3.0 \
  --swing-clearance-weight 0.5 \
  --low-clearance-penalty-weight 0.5
```

This remains a conservative training scaffold, not full PPO/SAC. Both actors
share the same residual ONNX deployment contract and action safety envelope.

The first clearance-focused phase/contact candidate improved median swing
clearance and locomotion metrics but did not pass the absolute M10 gate:

```text
candidate: m10_phase_contact_clearance_cem3x8_s53
flat:       0.01023m -> 0.01134m
start-stop: 0.00759m -> 0.00973m
curve:      0.00632m -> 0.00724m
```

All three scenarios remained below `0.015m`, so this candidate is experimental
and must not be promoted.

The gate-aligned revision adds episode median-clearance and low-clearance-ratio
terms:

```bash
--actor-kind command_state \
--episodic-clearance-weight 5 \
--episodic-low-clearance-penalty-weight 4
```

Its first larger candidate, `m10_command_state_gate_cem4x12_s67`, improved
every scenario without falling:

```text
flat:       0.01023m -> 0.01314m
start-stop: 0.00759m -> 0.01080m
curve:      0.00632m -> 0.00759m
```

Total three-scenario distance increased from `0.733m` to `1.070m`, but the
candidate still failed the absolute `0.015m` gate. Further linear CEM scaling
is not recommended.

The nonlinear actor can warm-start from that linear checkpoint:

```bash
--actor-kind command_state_mlp \
--initial-checkpoint /data/rl_finetune/m10_command_state_gate_cem4x12_s67/residual_policy.pt
```

Candidate `m10_command_state_mlp_cem4x14_s79` produced the strongest result:

```text
flat:       0.01023m -> 0.01471m
start-stop: 0.00759m -> 0.01152m
curve:      0.00632m -> 0.01025m
total distance: 0.733m -> 1.275m
```

It had no falls and a maximum stuck ratio of `0.0365`, but still failed G10.
Flat walking is only `0.00029m` below the median-clearance target; start/stop
and turning remain the larger blockers.

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

## M10 sequence and worst-case residual training knobs

The residual trainer can now score candidates against both fixed commands and
single-reset command sequences. Use fixed commands to preserve nominal flat walk
coverage, and use sequences to target start/stop or curve clearance:

```bash
--training-command 0.125,0,0,1.0 \
--training-sequence '2.5|0,0,0,50;0.06,0,0,100;0,0,0,50' \
--training-sequence '3.0|0.09,0,0,50;0.09,0,0.12,150;0.09,0,0,100'
```

The optional sequence prefix before `|` is the sequence weight. Each segment is
`VX,VY,YAW,STEPS` and runs in order after one simulator reset.

`--worst-case-score-weight` blends the weighted mean candidate score with the
lowest scenario score. Use this when a new clearance objective improves the hard
start/stop or turn case but risks regressing the near-passing flat case. A value
such as `0.35` keeps average progress useful while making the weakest scenario
visible to CEM selection.

When commands and sequences have different lengths, add
`--score-normalization per_step` so the optimizer compares each objective by
score per requested simulator step. This prevents a shorter start/stop sequence
from becoming the artificial worst case simply because it has fewer total reward
steps, and keeps worst-case pressure on the true low-clearance objective.

If the bottleneck objective has `low_clearance_ratio == 1.0`, add
`--episodic-clearance-gap-weight` so CEM distinguishes shallow misses from deep
misses. Unlike the low-clearance-ratio penalty, this gap term scales with the
mean normalized distance below the target clearance, which is useful when both
flat and turning are below the `0.015m` gate but turning is much lower.

Add `--final-score-breakdown` to re-score the best residual after training and
write a per-command/per-sequence score table into `residual_train_metrics.json`
and `residual_train_report.md`. The breakdown now includes completed steps,
termination state, median/min/max swing clearance, low-clearance ratio, and
episode-level clearance adjustment for each training objective. Sequence
objectives also include per-segment diagnostics, so the report can distinguish
startup, cruise, turning, and stop segments without rerunning the whole
experiment. This costs one extra evaluation pass over the configured objectives,
but it makes the next bottleneck visible before running the full M10
scenario-suite/clearance-readiness pipeline.

For the current documented clearance-refinement recipe, use the host wrapper:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/train_clearance_residual_policy.sh --dry-run
./scripts/train_clearance_residual_policy.sh
```

The wrapper warm-starts from `m10_command_state_mlp_cem4x14_s79`, preserves the
retained checkpoint's `residual_scale 0.1` by default, preserves a flat command,
emphasizes start/stop and curve sequences, uses per-step normalization, adds the
clearance-gap term, and writes the final score breakdown. Override
`--residual-scale` only when deliberately changing the deployed residual scale.
It does not replace the full scenario-suite and clearance-readiness promotion
gates.

2026-06-23 update: stacked residual-teacher refinement is now available in
`scripts/train_residual_policy.sh` through the runtime residual profile path.
The current best retained M10 clearance candidate is
`clearance_liftscale_stack_s143_step090_offset005`. It reuses the `s127`
`contact_phase_lift` ONNX with `phase.step_increment=0.9`,
`phase.offset=0.05`, and explicit `residual_scale=0.16`. It remains blocked by
G10 low-clearance ratio, but its full suite reaches total distance `~2.150 m`,
all p50 swing clearances above `0.015 m`, and no falls. Next clearance training
should beat s143's ratios (`flat ~0.268`, `start-stop ~0.257`,
`curve ~0.308`) while preserving distance. The narrow phase/scale/postprocess
brackets and small stacked continuations through `s173` did not pass the gate.
Follow-up action-scale, pre-roll, command-ramp, opt-in clearance-reflex, and
startup-tail continuation probes through
`clearance_s201_microreflex_s207` also did not pass. The closest
curve-only live result was `clearance_s177_tail_stack_s201` at low-clearance
ratio `~0.257`, p50 `~0.01846 m`, distance `~0.717 m`, and no fall; it still
missed the `0.25` ratio gate and should not be promoted.

`scripts/train_residual_policy.sh` now also supports per-objective
low-clearance regression penalties with repeated
`--reference-low-clearance-ratio` values and
`--low-clearance-regression-penalty-weight`. Use one ratio per training command
or sequence objective. The final score breakdown records the reference ratio
and regression penalty for each objective, which helps reject candidates before
the full suite.

Guarded probes from `s143` through `clearance_s143_refguard_stack_s215`,
`clearance_s143_gateguard_stack_s217`, and
`clearance_s143_curvegateguard_stack_s219` did not finish M10. `s217` and
`s219` each passed start-stop (`~0.245` and `~0.241` low-clearance ratio), but
both regressed flat and curve against `s143`; `s219` regressed curve to
`~0.340`. Do not promote these profiles. The next training attempt should
target lower-tail/startup clearance through a new teacher or reward/actor
redesign rather than another narrow profile scalar sweep, MLP rerun, reflex
wrapper, or low-clearance-ratio penalty retune.

Use the clearance readiness helper as both an absolute G10 gate and a retained
best comparison. For exploratory candidates that are still expected to miss
G10, omit `--strict` and require improvement against `s143` before retaining the
candidate. Reference improvement means preserving movement/no-fall behavior,
improving at least one low-clearance bottleneck, and not regressing the
low-clearance ratio in any required scenario:

```bash
./scripts/analyze_clearance_readiness.sh \
  --profile-name <candidate_profile> \
  --suite-dir artifacts/scenario_eval/<candidate_profile> \
  --reference-profile-name clearance_liftscale_stack_s143_step090_offset005 \
  --reference-suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 \
  --output-dir artifacts/clearance_readiness/<candidate_profile> \
  --json \
  --require-reference-improvement
```

For promotion evidence, add `--strict`; a candidate still must pass the absolute
clearance gate before follow-camera visual inspection or teacher comparison can
be promotion evidence.
