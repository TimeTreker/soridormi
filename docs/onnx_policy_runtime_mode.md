# M3.4 ONNX Policy Runtime Mode

This milestone adds an explicit experimental runtime mode:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy
```

The path is:

```text
RobotState -> OnnxPolicy -> 14D action -> PolicyActionMapper -> MotorCommand
```

This mode is opt-in. The default runtime mode remains `hold`.

## Recommended first test

Start MuJoCo with viewer and auto-reset:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Then start the ONNX policy runtime with logging:

```bash
./scripts/run_onnx_policy_runtime.sh
```

The robot may fall. That is expected at this stage. The goal is to verify:

- ONNX inference runs in the runtime loop.
- Actions are converted into `MotorCommand` messages.
- MuJoCo receives commands without crashing.
- MCAP logs are written for later inspection.

## Useful environment variables

```bash
SORIDORMI_POLICY_PATH=/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
SORIDORMI_RUNTIME_LOG=1
SORIDORMI_RUNTIME_LOG_FORMAT=mcap
CONTROL_HZ=50
```

## Success criteria

- Runtime starts in `onnx_policy` mode.
- CUDAExecutionProvider is used by the policy wrapper when available.
- Runtime loop keeps running until stopped or until MuJoCo auto-reset triggers.
- `data/logs/*.mcap` is created when logging is enabled.

## Next milestone

M3.5 should add policy-specific logging fields such as raw action, clipped targets, observation summary, and action history.
