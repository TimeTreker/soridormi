# M3 Policy Inference Roadmap

## M3.0 ONNX model inspection

Load the Open Duck Mini ONNX policy, print input/output metadata, and run one dummy inference.

## M3.1 Observation builder

Convert `RobotState` into the exact observation tensor expected by the policy.

## M3.2 Action mapper

Convert the ONNX output/action tensor into a safe `MotorCommand`.

## M3.3 Runtime policy mode

Add:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy
```

## M3.4 Policy logging

Record policy observations and actions into MCAP logs.

## M3.5 Policy simulation test

Run policy mode in MuJoCo with viewer and auto-reset enabled.
