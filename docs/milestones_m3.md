# M3 Milestones

## M3.0 ONNX policy inspection

Status: implemented.

Goal:

```text
load BEST_WALK_ONNX_2.onnx
inspect input/output names and shapes
run dummy inference
```

## M3.1 Observation builder

Status: implemented by this update.

Goal:

```text
RobotState -> 101D Open Duck ONNX observation
observation -> ONNX inference -> 14D action
```

This does not yet control the robot.

## M3.2 Policy wrapper

Next target.

Goal:

```text
persistent ONNX Runtime session
action history updates
provider reporting
MCAP logging of observation/action
```

## M3.3 Action to MotorCommand mapper

Goal:

```text
action[14] -> default_pose + action * action_scale -> MotorCommand
```

## M3.4 onnx_policy runtime mode

Goal:

```text
SORIDORMI_RUNTIME_MODE=onnx_policy
```

## M3.5 MuJoCo policy smoke test

Goal:

```text
run policy with viewer, auto-reset, and MCAP logging
```
