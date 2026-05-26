# M3 Policy Inference Milestones

## M3.0 ONNX inspection

- Inspect ONNX model inputs/outputs.
- Confirm dummy inference works.

## M3.1 Observation builder

- Convert `RobotState` to `obs` with shape `[1, 101]`.
- Keep policy action history.

## M3.2 Persistent ONNX policy wrapper

- Load ONNX session once.
- Prefer CUDAExecutionProvider when available.
- Build observation from `RobotState`.
- Return `continuous_actions` as shape `[14]`.
- Do not control motors yet.

## M3.3 Action-to-command mapper

- Convert 14 policy actions into `MotorCommand` position targets.
- Apply action scale and limits.

## M3.4 `onnx_policy` runtime mode

- Run full `RobotState -> obs -> ONNX -> action -> MotorCommand` loop in simulation.

## M3.5 Policy evaluation in MuJoCo

- Run with viewer, auto-reset, and MCAP logging.
