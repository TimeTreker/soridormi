# ONNX Observation Mapping

M3.1 adds a Soridormi observation builder for the Open Duck Mini v2 ONNX policy.

The inspected policy expects:

```text
input:  obs                  shape [1, 101] float32
output: continuous_actions   shape [1, 14]  float32
```

The 101-dimensional observation vector is built as:

| Component | Size |
|---|---:|
| IMU gyro xyz | 3 |
| IMU accelerometer xyz | 3 |
| Command vector | 7 |
| Joint position error, `q - default_pose` | 14 |
| Joint velocity scaled by `0.05` | 14 |
| Last action | 14 |
| Last-last action | 14 |
| Last-last-last action | 14 |
| Motor targets | 14 |
| Foot contacts | 2 |
| Imitation phase | 2 |
| **Total** | **101** |

The initial implementation uses safe defaults:

- command: zeros unless `SORIDORMI_POLICY_COMMAND` is set
- foot contacts: zeros
- imitation phase: zeros
- action history: zeros
- motor targets: `default_pose.positions`

Environment variables:

```bash
SORIDORMI_OBS_DOF_VEL_SCALE=0.05
SORIDORMI_ACTION_SCALE=0.25
SORIDORMI_OBS_ACCEL_X_BIAS=1.3
SORIDORMI_POLICY_COMMAND="0,0,0,0,0,0,0"
```

Probe one full observation plus ONNX inference:

```bash
./scripts/probe_policy_observation.sh
```

This does not control the robot. It only proves:

```text
RobotState -> obs[101] -> ONNX Runtime -> action[14]
```
