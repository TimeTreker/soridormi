# Default Pose Tuning Notes

## Test setup

Fixed-base simulator:

```bash
./scripts/run_fixed_base_stand_server.sh
```

Standing runtime:

```bash
./scripts/run_stand_runtime.sh
```

Normal-gravity simulator:

```bash
./scripts/run_normal_gravity_stand_server.sh
```

## Current chosen pose

Paste the current `default_pose.positions` block here when you find a good one.

```yaml
default_pose:
  positions:
    left_hip_yaw:
    left_hip_roll:
    left_hip_pitch:
    left_knee:
    left_ankle:

    neck_pitch:
    head_pitch:
    head_yaw:
    head_roll:

    right_hip_yaw:
    right_hip_roll:
    right_hip_pitch:
    right_knee:
    right_ankle:
```

## Runs

| Run | Hip pitch | Knee | Ankle | Fixed-base result | Normal-gravity result | Notes |
|---|---:|---:|---:|---|---|---|
| 001 | ±0.05 | ±0.10 | ±0.05 |  |  |  |
| 002 | ±0.10 | ±0.20 | ±0.10 |  |  |  |
| 003 | ±0.15 | ±0.30 | ±0.15 |  |  |  |
| 004 |  |  |  |  |  |  |
| 005 |  |  |  |  |  |  |

## Visual checklist

| Check | Result | Notes |
|---|---|---|
| Left/right legs symmetric |  |  |
| Feet roughly under body |  |  |
| Ankles not twisted |  |  |
| Knees bend naturally |  |  |
| Hip pitch signs look correct |  |  |
| No violent snap in fixed-base mode |  |  |
| No violent snap in normal gravity |  |  |
