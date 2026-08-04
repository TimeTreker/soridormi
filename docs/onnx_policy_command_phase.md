# ONNX policy command, gait phase, and motor-speed limits

command, gait-phase, and speed limits makes the experimental ONNX policy loop dynamic. Earlier policy runtime foundation issues proved that the ONNX model loads and that the runtime can turn actions into `MotorCommand`. This section adds the pieces that make the Open Duck policy observation less static:

- a 7D command vector
- a gait/imitation phase oscillator
- motor target speed limiting
- feedback of latest motor targets into the next observation

This is still an experimental mode. It does not guarantee stable walking.

## Command vector

The observation command layout is:

```text
[x_velocity, y_velocity, yaw_velocity, neck_pitch, head_pitch, head_yaw, head_roll]
```

Environment variables:

```bash
SORIDORMI_COMMAND_X=0.05
SORIDORMI_COMMAND_Y=0.0
SORIDORMI_COMMAND_YAW=0.0
SORIDORMI_NECK_PITCH=0.0
SORIDORMI_HEAD_PITCH=0.0
SORIDORMI_HEAD_YAW=0.0
SORIDORMI_HEAD_ROLL=0.0
```

## Phase oscillator

The phase vector is:

```text
[cos(2π phase), sin(2π phase)]
```

Environment variables:

```bash
SORIDORMI_PHASE_FREQUENCY=1.0
SORIDORMI_PHASE_ENABLED=1
SORIDORMI_PHASE_OFFSET=0.0
```

## Action mapping

Policy action mapping uses:

```text
raw_target = default_pose + action_scale * action
```

Then, if enabled:

```text
target = clip(raw_target, previous_target ± max_motor_velocity * dt)
```

Environment variables:

```bash
SORIDORMI_ACTION_SCALE=0.25
SORIDORMI_MAX_MOTOR_VELOCITY=5.24
```

## Recommended test

Start MuJoCo with viewer and auto reset:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Run the experimental policy command:

```bash
./scripts/run_onnx_walk_runtime.sh
```

Or customize:

```bash
SORIDORMI_COMMAND_X=0.03 \
SORIDORMI_PHASE_FREQUENCY=1.0 \
SORIDORMI_ACTION_SCALE=0.20 \
SORIDORMI_MAX_MOTOR_VELOCITY=5.24 \
./scripts/run_onnx_walk_runtime.sh
```

## Expected behavior

Success means:

- ONNX runtime starts in `onnx_policy` mode
- command vector is nonzero when requested
- imitation phase changes over time
- motor targets change smoothly due to speed limiting
- robot does something more dynamic than static standing
- auto-reset catches falls

The robot may still fall. That is acceptable at this stage.
