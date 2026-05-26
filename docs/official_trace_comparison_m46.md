# M4.6 Official vs Soridormi Trace Comparison

M4.4 proved that the official Open Duck Mini v2 MuJoCo + ONNX inference loop walks forward in the same Docker/MuJoCo environment. M4.5 moved Soridormi closer to that loop, but the robot can still wiggle without useful forward displacement.

M4.6 stops guessing by recording comparable per-policy-step traces from both systems.

## What is recorded

The official baseline now writes:

```text
/data/official_baseline/latest_official_baseline.trace.jsonl
```

Each line contains:

```text
observation[101]
action[14]
motor_targets[14]
joint_positions[14]
joint_velocities[14]
contacts[2]
phase[2]
command[7]
base_position_xyz[3]
default_actuator[14]
```

Soridormi runtime now also logs `/soridormi/policy_observation` in MCAP/JSONL so its observation vector can be compared directly with the official observation vector.

## Recommended workflow

Run the official reference:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
```

Run Soridormi with the same profile:

```bash
./scripts/run_official_compatible_policy_server.sh open_duck_forward
```

In another terminal:

```bash
./scripts/run_policy_experiment.sh open_duck_forward
```

Compare the latest official trace against the latest Soridormi policy log:

```bash
./scripts/compare_latest_official_soridormi_trace.sh
```

Or pass explicit paths:

```bash
./scripts/compare_official_soridormi_trace.sh \
  /data/official_baseline/latest_official_baseline.trace.jsonl \
  /data/logs/policy_open_duck_forward_YYYYMMDD_HHMMSS.mcap
```

## How to read the report

The comparison report prints mean absolute error and max absolute difference for:

```text
observation
action/raw_action
motor_targets
joint_positions
joint_velocities
contacts
phase
command
```

It also splits the 101D observation into named segments:

```text
gyro_xyz
accelerometer_xyz
command
joint_offsets
joint_velocities_scaled
last_action
last_last_action
last_last_last_action
motor_targets
feet_contacts
imitation_phase
```

The worst segment usually tells us what to port next.

## Reset at experiment start

Policy profiles now support:

```yaml
runtime:
  reset_at_start: true
```

This sends a reset request to the simulator before the runtime controller starts. It prevents a policy experiment from connecting to an already-fallen simulator state.
