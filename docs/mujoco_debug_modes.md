# MuJoCo debug modes

These modes are useful before the robot has a real balance controller.

## Zero gravity

Run the MuJoCo backend without gravity:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_MUJOCO_ZERO_GRAVITY=1 \
./scripts/run_sim_server.sh
```

Then run the runtime loop in another terminal:

```bash
SORIDORMI_RUNTIME_MODE=stand \
SORIDORMI_STAND_RAMP_SECONDS=5.0 \
./scripts/run_runtime_loop.sh
```

This does not test balance. It only helps verify joint mapping, actuator signs,
default pose targets, and runtime-to-simulator command flow.

## Normal gravity

Unset `SORIDORMI_MUJOCO_ZERO_GRAVITY` or set it to `0`:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
./scripts/run_sim_server.sh
```
