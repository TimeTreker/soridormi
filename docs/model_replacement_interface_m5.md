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
