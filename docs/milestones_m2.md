# M2 Milestones

## Completed before M2.7

- M2.1 config-driven MuJoCo backend
- M2.2 optional MuJoCo viewer mode
- M2.3 standing runtime mode
- M2.3a zero-gravity debug mode
- M2.4 joint direction validation
- M2.5 reset/default pose support
- M2.6 fixed-base debug mode

## M2.7 Default pose tuning

Use fixed-base mode to tune `default_pose.positions`.

```bash
./scripts/run_fixed_base_stand_server.sh
```

In another terminal:

```bash
./scripts/run_stand_runtime.sh
```

Then do a normal-gravity sanity check:

```bash
./scripts/run_normal_gravity_stand_server.sh
```

In another terminal:

```bash
./scripts/run_stand_runtime.sh
```

## Next milestone

M2.8 should add simulator safety/fall detection and reset behavior.
