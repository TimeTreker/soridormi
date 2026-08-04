# ONNX policy debug logging

policy debug logging adds policy-specific runtime logging for closed-loop ONNX debugging.

The normal runtime state and command topics are still logged:

```text
/soridormi/robot_state
/soridormi/motor_command
/soridormi/runtime_status
```

When `SORIDORMI_RUNTIME_MODE=onnx_policy`, the controller now exposes extra
policy debug payloads. MCAP logs write them as separate JSON topics:

```text
/soridormi/policy_action
/soridormi/policy_debug
/soridormi/policy_observation_stats
```

JSONL logs keep these fields inside each `runtime_step` line.

## Why this matters

If the robot falls repeatedly, state and motor commands are not enough. We also
need to know whether the policy loop is producing healthy internal values:

- command vector
- gait/imitation phase vector
- raw 14D action statistics
- mapped motor target statistics
- observation statistics
- action scale and motor velocity limit

## Conservative debug run

Start MuJoCo with viewer and auto-reset:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Run a small walking command:

```bash
SORIDORMI_COMMAND_X=0.01 \
SORIDORMI_PHASE_FREQUENCY=1.0 \
SORIDORMI_ACTION_SCALE=0.10 \
SORIDORMI_MAX_MOTOR_VELOCITY=3.0 \
SORIDORMI_RUNTIME_LOG=1 \
SORIDORMI_RUNTIME_LOG_FORMAT=mcap \
./scripts/run_onnx_walk_runtime.sh
```

Inspect the newest log:

```bash
./scripts/inspect_latest_log.sh
```

Expected topics include:

```text
/soridormi/policy_action
/soridormi/policy_debug
/soridormi/policy_observation_stats
```

The summary should also print a compact latest policy snapshot.
