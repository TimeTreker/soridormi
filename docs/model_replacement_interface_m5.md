# M5.1 Model replacement interface

M5 starts after the official Open Duck policy can be reproduced through the
Soridormi runtime. The goal is to make ONNX model replacement boring: a new model
should be selectable by profile, validated before runtime, and documented by a
single observation/action contract.

## Contract export

Use the contract exporter to inspect the runtime interface for a profile:

```bash
./scripts/export_policy_contract.sh open_duck_forward
```

For machine-readable output:

```bash
./scripts/export_policy_contract.sh open_duck_forward --json
```

The exporter does not load the ONNX file and does not start MuJoCo. It statically
combines:

- the selected `configs/policies/*.yaml` profile;
- the robot actuator/default-pose/action-mapping config;
- the canonical 101D Open Duck observation layout;
- the 14D action-to-motor target contract.

It fails if the profile's declared model IO shape is incompatible with the
runtime observation/action sizes, or if an optional declared joint order does not
match the robot contract.


## M5.2 profile/model preflight gate

`check_policy_model.sh --profile NAME` is the runtime preflight gate for model
replacement. It now validates two layers before a policy is run:

1. the static Soridormi contract exported by `policy_contract.py`; and
2. the actual ONNX file input/output metadata.

This means a profile fails fast if it declares an observation size, action size,
model IO shape, dtype, or optional joint order that does not match Soridormi's
runtime interface, even before MuJoCo or the runtime loop start.

Run the gate directly:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward
```

Use JSON output for CI or release automation:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward --json
```

`run_policy_experiment.sh PROFILE` already calls this gate unless
`SORIDORMI_SKIP_POLICY_CHECK=1` is set. Keep the skip flag for emergency local
debugging only; replacement models should pass the preflight gate before runtime.


## M5.3 ONNX execution providers

Soridormi now uses one ONNX provider selection path for both runtime inference and
`check_policy_model.sh`. By default it prefers CUDA when ONNX Runtime reports
`CUDAExecutionProvider`, then keeps CPU as a fallback:

```text
CUDAExecutionProvider,CPUExecutionProvider
```

TensorRT is intentionally not selected by default even when available because it
can introduce engine-build/cache behavior during policy-debug runs. Request it
explicitly when you want to experiment with it.

Check which providers are available and active for a profile:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward
```

Force a provider order:

```bash
SORIDORMI_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  ./scripts/check_policy_model.sh --profile open_duck_forward
```

Require GPU preflight failure if CUDA is not actually selected/activated:

```bash
./scripts/check_policy_model.sh \
  --profile open_duck_forward \
  --require-provider CUDAExecutionProvider
```

For CPU-only debugging:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward --cpu
# or
SORIDORMI_USE_CUDA_PROVIDER=0 ./scripts/run_policy_experiment.sh open_duck_forward
```

## Runtime contract

Current Open Duck-compatible replacement models must use:

```text
input:  obs                  shape [1, 101] dtype tensor(float)
output: continuous_actions   shape [1, 14]  dtype tensor(float)
```

Observation segments:

| Segment | Size | Meaning |
|---|---:|---|
| `gyro_xyz` | 3 | IMU angular velocity |
| `accelerometer_xyz` | 3 | IMU acceleration after configured policy bias |
| `command` | 7 | `x`, `y`, `yaw`, `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` |
| `joint_offsets` | 14 | `joint_position - policy_default_position` |
| `joint_velocities_scaled` | 14 | joint velocity multiplied by `dof_vel_scale` |
| `last_action` | 14 | previous policy action |
| `last_last_action` | 14 | action from two inference steps ago |
| `last_last_last_action` | 14 | action from three inference steps ago |
| `motor_targets` | 14 | previous speed-limited motor targets |
| `feet_contacts` | 2 | left/right foot contacts |
| `imitation_phase` | 2 | phase reference |
| **Total** | **101** | |

Action mapping:

```text
raw_target = default_position + action_scale * action
target = speed_limit(raw_target, previous_target, max_motor_velocity * dt)
target = optional_ctrlrange_clip(target)
```

## Replacing a model

1. Copy an existing profile under `configs/policies/`.
2. Change `model.path` to the new ONNX file.
3. Update `model.input_name`, `model.output_name`, shapes, and dtypes if the
   exporter used different names.
4. Run:

```bash
./scripts/export_policy_contract.sh my_profile
./scripts/check_policy_model.sh --profile my_profile
```

Only run the policy after both checks pass.

## Optional profile contract metadata

Profiles may include a static declaration. This is useful when profiles are
shared with trained artifacts:

```yaml
metadata:
  format_version: 1
  policy_family: open_duck_mini_v2

contract:
  observation_size: 101
  action_size: 14
  joint_names:
    - left_hip_yaw
    - left_hip_roll
    # ... all 14 in action order
```

`contract.joint_names` is optional because the robot config remains the source of
truth for actuator order. If present, it must exactly match the runtime order.
