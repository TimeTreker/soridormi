# Default pose tuning

default pose tuning is a visual/debugging qualification issue. The goal is to tune `default_pose` while keeping the
MuJoCo base fixed so the robot does not immediately fall away from the camera.

This is **not** a walking controller and not a balance controller.

## What to edit

Tune only this section:

```yaml
default_pose:
  positions:
    ...
```

Do not change `reset_pose` during default pose tuning unless you are deliberately starting a reset-pose
tuning section.

## Fixed-base test

Terminal 1:

```bash
./scripts/run_fixed_base_stand_server.sh
```

Terminal 2:

```bash
./scripts/run_stand_runtime.sh
```

Expected result:

- the MuJoCo viewer opens
- the base/body stays fixed
- joints move smoothly toward `default_pose`
- left and right legs look symmetric
- feet are roughly under the body
- ankles do not look twisted

## Normal-gravity sanity check

After the fixed-base pose looks reasonable, stop the fixed-base server and run:

```bash
./scripts/run_normal_gravity_stand_server.sh
```

In another terminal:

```bash
./scripts/run_stand_runtime.sh
```

Expected result:

- the robot may fall
- the robot should not snap violently
- the robot should not disappear immediately
- legs should move smoothly toward the tuned pose

A slow fall is acceptable at this stage.

## Suggested tuning sequence

Use small changes only.

| Run | Left hip pitch | Left knee | Left ankle | Right hip pitch | Right knee | Right ankle |
|---|---:|---:|---:|---:|---:|---:|
| 001 | -0.05 | 0.10 | -0.05 | 0.05 | -0.10 | 0.05 |
| 002 | -0.10 | 0.20 | -0.10 | 0.10 | -0.20 | 0.10 |
| 003 | -0.15 | 0.30 | -0.15 | 0.15 | -0.30 | 0.15 |

Stop increasing if the pose looks unnatural.

## Restarting after changes

The standing controller reads the config when the runtime starts. After every YAML edit:

1. Stop runtime with `Ctrl+C`
2. Start it again:

```bash
./scripts/run_stand_runtime.sh
```

You usually do not need to restart the sim server unless you changed `reset_pose`.

## Success criteria

default pose tuning is complete when:

- fixed-base mode keeps the body stable
- stand mode smoothly moves legs to `default_pose`
- left/right legs look symmetric
- feet are roughly under the body
- normal-gravity mode falls slowly rather than violently
- notes are recorded in `docs/notes/default_pose_tuning.md`
