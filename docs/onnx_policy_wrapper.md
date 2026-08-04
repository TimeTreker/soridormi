# Persistent ONNX policy wrapper

persistent ONNX policy wrapper adds a persistent ONNX Runtime policy wrapper.

The wrapper loads the ONNX session once, owns an `ObservationBuilder`, keeps action history, and returns a 14-dimensional action vector for each `RobotState`.

It does **not** convert actions to motor commands yet. That happens in action-to-motor-command mapping.

## Probe

```bash
./scripts/probe_onnx_policy_wrapper.sh
```

Expected output:

```text
Selected providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Input:  obs shape=[1, 101]
Output: continuous_actions shape=[1, 14]
Action shape: [14]
Probe OK
```

## Programmatic use

```python
from soridormi_runtime.onnx_policy import OnnxPolicy

policy = OnnxPolicy()
action = policy.compute_action(state)
```

## Notes

The wrapper uses JSON/config-driven joint ordering through `ObservationBuilder.from_robot_config()`. The default action history starts at zero and updates after every inference step.
