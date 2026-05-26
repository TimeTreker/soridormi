# M4.9: Official sensor/contact compatibility

M4.8 aligned Soridormi's imitation phase with the official Open Duck trace. The
next largest trace mismatches were IMU observations and foot contacts.

M4.9 keeps the ONNX policy path unchanged, but makes the MuJoCo server closer to
the official Open Duck inference reset and observation behavior.

## What changed

The official-compatible policy server now enables these MuJoCo flags by default:

```bash
SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1
SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1
SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1
```

### Official reset sequence

When `SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE=1`, the backend reproduces the
Open Duck inference startup/reset sequence more closely:

1. `mj_step(model, data)` once from freshly reset data.
2. Copy `model.keyframe("home").qpos` into `data.qpos`.
3. Copy `model.keyframe("home").ctrl` into `data.ctrl`.
4. Preserve qvel instead of forcing it to zero.
5. Skip the extra `mj_forward()` used by the generic Soridormi reset path.

This matters because the pretrained policy observes gyro and accelerometer data.
Changing the reset sequence can change the early sensor trace and cause action
history to diverge.

### Official sensor mode

When `SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE=1`, the backend requires the MuJoCo
`gyro` and `accelerometer` sensors and reads them directly from `sensordata`. It
no longer silently falls back to qvel/default acceleration in official mode.

The observation builder still applies the Open Duck accelerometer bias from the
policy profile, normally `[1.3, 0.0, 0.0]`.

### Official contact mode

When `SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE=1`, foot contacts use the same body
contact semantics as the official Open Duck inference path:

```text
left:  foot_assembly   vs floor
right: foot_assembly_2 vs floor
```

The older geom fallback remains available only when official contact mode is off.

## Run

Start the official-compatible server:

```bash
./scripts/run_official_compatible_policy_server.sh open_duck_forward
```

Second terminal:

```bash
./scripts/run_policy_experiment.sh open_duck_forward
```

Then compare:

```bash
./scripts/compare_latest_official_soridormi_trace.sh
```

Expected improvement target:

```text
accelerometer_xyz mean_mae lower
feet_contacts mean_mae lower
forward_x closer to official
```

If IMU/contact improve but forward motion is still weaker, the next target is to
compare the exact first diverging action-history and motor-target steps.
