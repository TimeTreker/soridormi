# M6 Simulation Development Runbook

This runbook describes how to run the Soridormi simulation policy-improvement workflow.

The goal is to improve a high-level walking policy in MuJoCo while keeping the runtime contract:

```text
obs[101] -> action[14]
```

## 0. Repository setup

From the Soridormi repo root:

```bash
./scripts/setup_env.sh
./scripts/add_submodules.sh
./scripts/build_sim.sh
./scripts/build_runtime_training.sh
```

The training image should include PyTorch, ONNX, and ONNX export dependencies.

Verify:

```bash
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
source /opt/venvs/runtime/bin/activate
python - <<PY
import torch
import onnx
import onnxscript
import onnxruntime as ort
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("onnx:", onnx.__version__)
print("onnxscript: OK")
print("providers:", ort.get_available_providers())
PY
'
```

Expected provider list should include `CUDAExecutionProvider`.

## 1. Start the MuJoCo sim server

Open terminal A:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 \
SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1 \
SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1 \
SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1 \
SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Optional viewer:

```bash
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 \
SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1 \
SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1 \
SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1 \
SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Keep this terminal open.

## 2. Verify the default policy

Open terminal B:

```bash
./scripts/check_policy_model.sh \
  --profile open_duck_forward \
  --require-provider CUDAExecutionProvider
```

Run a bounded default rollout:

```bash
./scripts/run_policy_rollout_smoke.sh open_duck_forward --steps 1000
```

Save the latest teacher log path:

```bash
teacher_log="$(ls -t data/logs/policy_open_duck_forward_*.mcap | head -1)"
echo "$teacher_log"
```

## 3. Generate supervised teacher data

```bash
./scripts/export_training_dataset.sh "$teacher_log" \
  --output data/training_datasets/open_duck_forward_supervised.jsonl
```

Prepare splits:

```bash
./scripts/prepare_training_dataset.sh \
  data/training_datasets/open_duck_forward_supervised.jsonl \
  --output-dir data/training_datasets/open_duck_forward_prepared \
  --seed 123
```

Summarize and write normalization artifacts:

```bash
./scripts/summarize_training_dataset.sh \
  data/training_datasets/open_duck_forward_prepared
```

## 4. Train a neural behavior-clone candidate

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

Check exported ONNX:

```bash
./scripts/check_policy_model.sh \
  --profile neural_bc_open_duck \
  --require-provider CUDAExecutionProvider
```

Run bounded rollout:

```bash
./scripts/run_policy_rollout_smoke.sh neural_bc_open_duck --steps 1000
candidate_log="$(ls -t data/logs/policy_neural_bc_open_duck_*.mcap | head -1)"
echo "$candidate_log"
```

Compare teacher and candidate:

```bash
./scripts/compare_policy_rollouts.sh "$teacher_log" "$candidate_log" \
  --min-candidate-policy-records 800 \
  --min-candidate-duration 10 \
  --max-candidate-resets 0 \
  --min-forward-ratio 0.5 \
  --max-lateral-abs 0.25 \
  --max-action-abs 5.0
```

## 5. Run the RL fine-tuning environment smoke

```bash
./scripts/run_rl_finetune_env.sh \
  --profile open_duck_forward \
  --steps 20 \
  --residual-scale 0.05 \
  --target-height 0.30 \
  --fall-height 0.14 \
  --min-upright 0.65 \
  --output data/rl_finetune_env/m618_reward_smoke.json
```

This checks the environment/reward loop before training residual policy.

## 6. Train residual fine-tuned policy

With sim server still running:

```bash
./scripts/train_residual_policy.sh open_duck_forward \
  --output-dir data/rl_finetune/residual_open_duck \
  --profile-name residual_open_duck \
  --iterations 5 \
  --population 16 \
  --steps-per-episode 300 \
  --residual-scale 0.05 \
  --force-profile
```

Expected artifacts:

```text
data/rl_finetune/residual_open_duck/residual_policy.pt
data/rl_finetune/residual_open_duck/residual_policy.onnx
data/rl_finetune/residual_open_duck/residual_train_metrics.json
data/rl_finetune/residual_open_duck/residual_train_report.md
configs/policies/residual_open_duck.yaml
```

Check the residual profile:

```bash
./scripts/check_policy_model.sh \
  --profile residual_open_duck \
  --require-provider CUDAExecutionProvider
```

