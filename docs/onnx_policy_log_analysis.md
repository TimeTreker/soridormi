# ONNX policy log analysis

policy debug logging added policy-specific runtime log topics:

- `/soridormi/policy_action`
- `/soridormi/policy_debug`
- `/soridormi/policy_observation_stats`

policy log analysis adds an analyzer that summarizes those topics and detects reset cycles from
robot-time drops. This is meant to help diagnose repeated falls before tuning
random gait parameters.

## Run a conservative logged walk test

Terminal 1:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

Terminal 2:

```bash
SORIDORMI_COMMAND_X=0.01 \
SORIDORMI_PHASE_FREQUENCY=1.0 \
SORIDORMI_ACTION_SCALE=0.10 \
SORIDORMI_MAX_MOTOR_VELOCITY=3.0 \
SORIDORMI_RUNTIME_LOG=1 \
SORIDORMI_RUNTIME_LOG_FORMAT=mcap \
./scripts/run_onnx_walk_runtime.sh
```

## Analyze the newest log

```bash
./scripts/analyze_latest_policy_log.sh
```

The analyzer prints:

- topic counts
- robot-time duration
- detected reset cycles
- action statistics
- motor target statistics
- joint position statistics
- observation norm statistics
- latest command, phase, action scale, and max motor velocity
- a compact diagnosis list

## How to use the output

If the analyzer says policy actions are nearly zero, first verify that command
and phase are nonzero in the runtime startup log.

If actions are large and falls are immediate, reduce `SORIDORMI_ACTION_SCALE`
and `SORIDORMI_MAX_MOTOR_VELOCITY` before changing other values.

If reset cycles are consistently short, compare Soridormi's observation mapping,
default pose, reset pose, phase frequency, and action ordering against the Open
Duck inference script before tuning gains.

## JSON output

For scripts or notebooks:

```bash
python -m soridormi_runtime.analyze_policy_log /data/logs/runtime_xxx.mcap --json
```
