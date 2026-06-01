# M6 Simulation Status

Current conclusion: **M6 is not finished as a proven command-conditioned free-walking result in simulation yet.**

M6 has built most of the **simulation learning backbone**:

- policy observation/action contract: `obs[101] -> action[14]`
- teacher behavior-cloning data path
- leakage-safe grouped train/val/test split path
- neural behavior-clone ONNX export path
- MuJoCo rollout comparison path
- RL fine-tuning environment path
- walking-quality reward function
- residual-policy fine-tuning path

But a simulation milestone is only complete when teacher and candidate policies have been run and compared across a command-conditioned free-walk suite in MuJoCo, not just one fixed forward command.

## Immediate correction

Do not move to hardware walking yet. The next work item is a commanded free-walk evaluation gate: run teacher/candidate policies across stop, forward, turn, curve, lateral, and command-switching scenarios, then use those results to decide what training data or residual fine-tuning is needed. See `docs/SORIDORMI_FREE_WALK_PLAN.md` and `docs/M6_SIM_TRAINING_LOOP.md`.

## Completion definition

Call M6 complete only after all of the following are true:

1. The default/teacher policy can run in MuJoCo through Soridormi.
2. A commanded free-walk evaluation suite covers stop, forward, turn, curve, lateral, and command-switching scenarios.
3. Teacher rollouts produce a per-scenario report with survival time, termination reason, velocity tracking, drift, upright/height error, and action metrics.
4. A neural BC or residual candidate exports to ONNX and passes `check_policy_model.sh`.
5. The candidate runs in MuJoCo across the same command suite.
6. Teacher-vs-candidate comparison is generated per scenario, not just as a single aggregate score.
7. The candidate is accepted only if it is stable and safe across the suite and improves at least one chosen metric without regressions on critical safety metrics.

## Current practical status

The latest residual fine-tuning run reached a best score, which means the optimization loop ran, but ONNX export failed because the training environment did not include `onnxscript`.

After applying the dependency/export patch and rebuilding the training runtime image, rerun residual training and then rollout comparison.

## Important distinction

M6 has not trained a dynamics model.

M6 trains or fine-tunes a policy:

```text
policy_input:  obs[101]
policy_output: action[14]
```

The output action is mapped to joint target offsets by Soridormi. It is not direct torque control.

## Teacher data source

Teacher data should come from the default/Open Duck policy running many MuJoCo rollouts.

A small dataset is only useful for smoke testing the training pipeline. A real dataset should include many episodes across:

- forward velocities
- yaw commands
- lateral commands
- different rollout lengths
- perturbations
- phase offsets
- failure and recovery states, when safe to collect

## Why residual fine-tuning

The default policy is a baseline, not a perfect target.

Behavior cloning can copy the teacher, but cannot reliably improve beyond it. Residual fine-tuning is the preferred next step:

```text
teacher_action + bounded_residual -> final_action
```

## Current priority update

Soridormi should prioritize locomotion quality before platform orchestration. MCP/LLM integration is useful later, but it does not make Open Duck Mini walk better by itself. The immediate Soridormi work is:

```text
commanded free-walk evaluation
→ command-distribution teacher data
→ neural BC closed-loop comparison
→ residual fine-tuning if needed
→ sim acceptance gate
→ hardware bring-up only after sim acceptance
```
