# M6 Simulation Status

Current conclusion: **M6 is not finished as a proven walking-improvement result in simulation yet.**

M6 has built most of the **simulation learning backbone**:

- policy observation/action contract: `obs[101] -> action[14]`
- teacher behavior-cloning data path
- leakage-safe grouped train/val/test split path
- neural behavior-clone ONNX export path
- MuJoCo rollout comparison path
- RL fine-tuning environment path
- walking-quality reward function
- residual-policy fine-tuning path

But a simulation milestone is only complete when the final candidate has been trained, exported, run, and compared against the default policy in MuJoCo.

## Immediate correction

Do not move to hardware walking yet. The next work item is the sim training loop: collect teacher-policy rollouts from MuJoCo, train a behavior-clone or residual candidate, export it, and compare it against the default policy in simulation. See `docs/M6_SIM_TRAINING_LOOP.md`.

## Completion definition

Call M6 complete only after all of the following are true:

1. The default policy can run in MuJoCo through Soridormi.
2. The residual fine-tuned policy exports to ONNX.
3. The residual profile passes `check_policy_model.sh`.
4. The residual profile runs in MuJoCo with bounded rollout.
5. Default-vs-residual rollout comparison is generated.
6. The residual policy is at least safe and stable, and ideally improves one or more walking metrics.

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
