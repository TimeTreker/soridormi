# M3 Milestones

## M3.0 ONNX inspection

- Inspect ONNX inputs and outputs.
- Verify dummy inference.

## M3.1 Observation builder

- Convert `RobotState` to `[1, 101]` ONNX observation.

## M3.2 Persistent ONNX policy wrapper

- Load policy once.
- Maintain action history through `ObservationBuilder`.
- Return 14D action vectors.

## M3.3 Action-to-MotorCommand mapper

- Convert 14D policy action to motor target positions.
- Use `default_pose + action_scale * action`.
- Emit Soridormi `MotorCommand`.
- Keep observation `motor_targets` up to date.

## M3.4 Experimental ONNX runtime mode

- Add `SORIDORMI_RUNTIME_MODE=onnx_policy`.
- Test first in fixed-base or zero-gravity debug mode.

## M3.5 Normal MuJoCo policy test

- Run with viewer, auto-reset, and MCAP logging.
