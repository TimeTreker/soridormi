# Workflow

## PC simulation loop

Build once after dependency, Dockerfile, or base-image changes:

```bash
./scripts/setup_env.sh
./scripts/build_sim.sh
```

Terminal 1:

```bash
./scripts/run_sim_server.sh
```

Terminal 2:

```bash
./scripts/run_runtime_loop.sh
```

The Python environments are built into the images:

```text
/opt/venvs/sim
/opt/venvs/runtime
```

So daily runs do not require `bootstrap_simulator` or `bootstrap_runtime`.

## When to rebuild

Rebuild when you change:

- `pyproject.toml`
- a Dockerfile
- system packages
- CUDA/cuDNN base images
- Python dependencies

Normal edits under `src/` are bind-mounted and do not require rebuild.

## Development steps

1. Keep API changes in `src/soridormi_api`.
2. Implement sim behavior in `src/soridormi_sim`.
3. Keep runtime code in `src/soridormi_runtime`.
4. Keep hardware integration behind `HardwareRobot`.
5. Avoid importing MuJoCo in runtime code.

## GPU inference during PC simulation

The PC runtime-dev image uses the CUDA 13.1 cuDNN base image by default and is given GPU access in `compose.sim.yaml`. Policy loading prefers ONNX Runtime's `CUDAExecutionProvider` when available, then falls back to CPU.

## Config-driven MuJoCo backend

Robot/model-specific simulation settings live in YAML config files under:

```text
configs/robots/
```

The default Open Duck Mini v2 config is:

```text
configs/robots/open_duck_mini_v2.yaml
```

Daily fake-backend API testing uses:

```bash
SORIDORMI_SIM_BACKEND=fake ./scripts/run_sim_server.sh
```

To start the real MuJoCo backend with the configured Open Duck Mini v2 model:

```bash
SORIDORMI_SIM_BACKEND=mujoco ./scripts/run_sim_server.sh
```

To start the real MuJoCo backend with the passive viewer enabled:

```bash
xhost +local:docker
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
./scripts/run_sim_server.sh
```

Then start the runtime loop from another terminal:

```bash
./scripts/run_runtime_loop.sh
```

To use a different robot model later, create a new config file and point the sim server at it:

```bash
SORIDORMI_ROBOT_CONFIG=/app/configs/robots/my_robot.yaml \
SORIDORMI_SIM_BACKEND=mujoco \
./scripts/run_sim_server.sh
```

The backend code should stay generic. Robot structure, actuator names, base joint layout, control mode, and model path belong in config.

## MuJoCo viewer mode

The viewer is optional and is controlled by the robot config's `viewer.enabled_env` field. The default Open Duck Mini v2 config uses:

```text
SORIDORMI_MUJOCO_VIEWER
```

Set it to `1` only when you want a GUI window. Keep it unset for headless API tests and CI.
