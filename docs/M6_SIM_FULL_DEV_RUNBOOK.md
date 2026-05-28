# M6 simulation full development runbook

This runbook describes the complete Soridormi M6 simulation workflow from environment setup through residual fine-tuning and rollout comparison.

The policy contract remains:

```text
policy input:  obs[101]
policy output: action[14]
```

The action is a high-level policy output that Soridormi maps to target joint-position offsets. It is not torque control.

## 1. Build the environment

From the Soridormi repo root:

```bash
./scripts/setup_env.sh
./scripts/add_submodules.sh
./scripts/build_sim.sh
./scripts/build_runtime_training.sh
```

Verify training/runtime dependencies:

```bash
docker compose -f compose.sim.yaml run --rm runtime bash -lc '
source /opt/venvs/runtime/bin/activate
python - <<PY_INNER
import torch
import onnx
import onnxscript
import onnxruntime as ort
print("torch:", torch.__version__)
print("torch cuda:", torch.cuda.is_available())
print("onnx:", onnx.__version__)
print("onnxscript: OK")
print("onnxruntime providers:", ort.get_available_providers())
PY_INNER
'
```

Expected providers should include `CUDAExecutionProvider`.

## 2. Start MuJoCo sim server

Terminal A:

```bash
SORIDORMI_SIM_BACKEND=mujoco SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1 SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1 SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1 SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1 SORIDORMI_AUTO_RESET=1 ./scripts/run_sim_server.sh
```

Optional viewer:

```bash
SORIDORMI_MUJOCO_VIEWER=1 SORIDORMI_SIM_BACKEND=mujoco SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE=1 SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1 SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1 SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1 SORIDORMI_AUTO_RESET=1 ./scripts/run_sim_server.sh
```

Keep this terminal open while running experiments.

## 3. Validate the default teacher policy

Terminal B:

```bash
./scripts/check_policy_model.sh   --profile open_duck_forward   --require-provider CUDAExecutionProvider
```

Run a bounded rollout:

```bash
./scripts/run_policy_rollout_smoke.sh open_duck_forward   --steps 1000   --require-provider CUDAExecutionProvider
```

## 4. Train residual policy

```bash
./scripts/train_residual_policy.sh open_duck_forward   --output-dir data/rl_finetune/residual_open_duck   --profile-name residual_open_duck   --iterations 5   --population 16   --steps-per-episode 300   --residual-scale 0.05   --force-profile
```

Expected artifacts:

```text
data/rl_finetune/residual_open_duck/residual_policy.pt
data/rl_finetune/residual_open_duck/residual_policy.onnx
data/rl_finetune/residual_open_duck/residual_train_metrics.json
data/rl_finetune/residual_open_duck/residual_train_report.md
configs/policies/residual_open_duck.yaml
```

## 5. Validate residual policy profile

```bash
./scripts/check_policy_model.sh   --profile residual_open_duck   --require-provider CUDAExecutionProvider
```

Expected result:

```text
Runtime contract: OK
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Result: OK
```

The residual ONNX may use a dynamic batch dimension such as `['batch', 101]`; this is acceptable as long as the feature/action dimensions are `101` and `14`.

## 6. Compare default vs residual rollout

```bash
./scripts/run_residual_finetune_comparison.sh residual_open_duck   --teacher-profile open_duck_forward   --steps 1000   --require-provider CUDAExecutionProvider
```

A passing run should produce a report under:

```text
data/policy_rollout_comparisons/
```

Minimum acceptable forward-walk milestone:

```text
Result: PASS
policy records: 1000 / 1000
reset count: 0 / 0
candidate forward ratio >= 1.0 is preferred
candidate lateral drift should not be worse than teacher
candidate action abs max should remain bounded
```

## 7. Broader validation before hardware

The forward-walk pass is not enough for hardware walking. Run a command grid before M7 walking tests.

Suggested grid:

```text
vx:       0.05, 0.10, 0.15, 0.20
yaw_rate: -0.3, 0.0, 0.3
vy:       -0.03, 0.0, 0.03
steps:    1000 or 2000
repeats:  at least 3 per condition
```

For every run, inspect:

```text
reset count
forward displacement and speed
lateral drift
yaw drift
action magnitude
joint/contact anomalies
viewer behavior, if possible
```

## 8. Common failures

### Sim server timeout

Symptom:

```text
zmq.error.Again: Resource temporarily unavailable
```

Meaning: runtime could not get a sim API reply. Start `run_sim_server.sh` first and keep it open.

### Missing CUDA provider

Symptom:

```text
Required provider CUDAExecutionProvider not active
```

Rebuild the training/runtime image and verify ONNX Runtime providers.

### Missing ONNX export dependency

Symptom:

```text
ModuleNotFoundError: No module named 'onnxscript'
```

Run:

```bash
./scripts/build_runtime_training.sh
```

Then verify `import onnxscript` inside the runtime container.

### Read-only configs

Profile-generating workflows need writable configs. Use the project scripts rather than manually entering the runtime container; they set the correct mount mode when needed.

## 9. Project interpretation

M6 is a sim-learning backbone, not a final locomotion product. The current residual policy is a conservative first improvement mechanism. Future work may replace the residual optimizer with PPO, SAC, recurrent actors, or richer context-aware policies while preserving the runtime contract.
