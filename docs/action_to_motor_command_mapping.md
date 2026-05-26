# M3.3 Action-to-MotorCommand Mapping

M3.3 adds the bridge from a 14-dimensional ONNX policy action to a Soridormi `MotorCommand`.

This section still does **not** enable closed-loop ONNX walking. It only validates the mapping:

```text
RobotState -> ObservationBuilder -> OnnxPolicy -> action[14] -> PolicyActionMapper -> MotorCommand
```

## Mapping rule

The first mapping follows the Open Duck inference convention:

```text
motor_target = default_pose + action_scale * action
```

Default values:

```text
action_scale = 0.25
kp_default = value from action_mapping or default_pose.gains or 10.0
kd_default = value from action_mapping or default_pose.gains or 0.5
```

The MuJoCo backend still clips controls to the actuator `ctrlrange`. The mapper also supports optional per-actuator `ctrlrange` if actuator entries include it.

## Optional config snippet

Add this to `configs/robots/open_duck_mini_v2.yaml` if you want explicit tuning values:

```yaml
action_mapping:
  action_scale: 0.25
  kp_default: 10.0
  kd_default: 0.5
  torque_default: 0.0
  clip_to_limits: true
```

## Probe

```bash
./scripts/probe_action_mapper.sh
```

Expected output:

```text
Policy
------
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
action shape: [14]

MotorCommand
------------
joint count: 14
position min/max: ...
Probe OK
```

## Next section

M3.4 will add an experimental runtime mode:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy
```

That mode should initially be tested only with:

```bash
SORIDORMI_MUJOCO_FIXED_BASE=1
```

or:

```bash
SORIDORMI_MUJOCO_ZERO_GRAVITY=1
```
