# Soridormi

**Soridormi** is a sim-to-real humanoid robot development stack based on Open Duck Mini v2.

The project goal is to replace the original Raspberry Pi-style onboard runtime with an NVIDIA Jetson-class runtime, while keeping the same high-level robot API in simulation and on real hardware.

Project name note: the intended bronze dragon name is **Soridormi**.

## Design goals

```text
soridormi-runtime
  Robot-facing runtime.
  Runs policy inference, controller loop, and hardware/sim API client.
  No MuJoCo dependency.

soridormi-sim
  PC-side simulator.
  Runs MuJoCo, simulated motors, simulated IMU, simulated joint state, and API server.

soridormi-api
  Shared message/API package used by both runtime and simulator.
```

In simulation:

```text
soridormi-runtime  <---- same robot API ---->  soridormi-sim / MuJoCo
```

On real robot:

```text
soridormi-runtime  <---- same robot API ---->  motors / IMU / encoders / power
```

The runtime code should not need to know whether it talks to MuJoCo or real hardware.

## Target hardware

Long-term primary target:

- NVIDIA Jetson AGX Thor
- Ubuntu 24.04 / JetPack 7.x-class system
- CUDA 13.x-class runtime

Compatibility target:

- NVIDIA Jetson AGX Orin
- JetPack 6.x-class system, typically Ubuntu 22.04 today
- CUDA 12.x-class runtime

Development host:

- Ubuntu desktop PC
- NVIDIA GPU
- Docker + Docker Compose
- MuJoCo simulation

## Repository structure

```text
soridormi/
├── compose.sim.yaml             # PC simulation: runtime-dev + simulator
├── compose.orin.yaml            # Jetson AGX Orin runtime deployment template
├── compose.thor.yaml            # Jetson AGX Thor runtime deployment template
├── docker/
│   ├── runtime/Dockerfile       # Runtime image, parameterized by BASE_IMAGE
│   └── simulator/Dockerfile     # PC MuJoCo/CUDA simulation image
├── scripts/
│   ├── add_submodules.sh
│   ├── setup_env.sh
│   ├── build_sim.sh
│   ├── enter_runtime_dev.sh
│   ├── enter_sim.sh
│   ├── run_sim_server.sh
│   └── run_runtime_loop.sh
├── src/
│   ├── soridormi_api/
│   ├── soridormi_runtime/
│   └── soridormi_sim/
└── workspace/
    └── upstream Open Duck repos as git submodules
```


### Docker container user

Both Docker images create and use the same non-root user:

```text
chromie
```

The username is controlled by `CONTAINER_USER=chromie` in `.env`. The Dockerfiles also handle base images where UID/GID 1000 already exists, so builds should not fail with `GID '1000' already exists`.

## Quick start on PC

### 1. Clone this repo

```bash
git clone https://github.com/TimeTreker/soridormi.git
cd soridormi
```

If you started from the zip file, unzip it, then initialize git:

```bash
cd soridormi
git init
```

### 2. Add upstream submodules

```bash
./scripts/add_submodules.sh
```

This adds:

```text
workspace/Open_Duck_Mini
workspace/Open_Duck_Mini_Runtime
workspace/Open_Duck_Playground
```

### 3. Create local `.env`

```bash
./scripts/setup_env.sh
```

### 4. Build PC simulation stack

```bash
./scripts/build_sim.sh
```

This builds two logical images:

```text
soridormi-runtime:cuda13.1-cudnn-dev  GPU-capable runtime-dev image for PC simulation
soridormi-sim:cuda13.1-cudnn          heavy MuJoCo/CUDA/cuDNN simulation image
```

The real robot runtime uses the same runtime Dockerfile but a Jetson-specific base image.

By default, the PC-side runtime-dev and simulator images use:

```text
nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
```

The runtime-dev container is GPU-enabled so ONNX Runtime can prefer `CUDAExecutionProvider` during PC simulation. If `onnxruntime-gpu` is not compatible with your local driver/CUDA stack, set `RUNTIME_DEV_EXTRA=runtime` and `SORIDORMI_ONNXRUNTIME_GPU=0` in `.env`, then rebuild with `./scripts/build_sim.sh`.

Python dependencies are installed during `docker compose build`. Source code is still bind-mounted from `./src`, so normal code edits do not require rebuilding the image. Rebuild only when `pyproject.toml`, a Dockerfile, system dependencies, or base images change.

### 5. Enter simulator container

```bash
./scripts/enter_sim.sh
```

Inside the container:

```bash
whoami
sim
python - <<'PY'
import soridormi_api, soridormi_sim
print("Soridormi simulator imports OK")
PY
```

The simulator Python environment is now built into the Docker image at `/opt/venvs/sim`, so `bootstrap_simulator` is no longer required for normal daily use.

### 6. Start simulator API server

You can start it directly from the host:

```bash
./scripts/run_sim_server.sh
```

Or from inside the simulator container:

```bash
sim
python -m soridormi_sim.mujoco_server
```

At this stage the server includes a safe fake backend so the API can be tested immediately. Replace or extend `src/soridormi_sim/mujoco_backend.py` to connect the API to the real Open Duck MuJoCo model.

### 7. Start runtime container in another terminal

You can start the runtime loop directly from the host:

```bash
./scripts/run_runtime_loop.sh
```

Or enter the runtime-dev container:

```bash
./scripts/enter_runtime_dev.sh
```

Inside the runtime-dev container:

```bash
runtime
python -m soridormi_runtime.main
```

The runtime Python environment is built into the Docker image at `/opt/venvs/runtime`. The runtime will connect to the simulator using the same API shape that the real hardware backend should implement.

## Common commands

Build simulation images:

```bash
./scripts/build_sim.sh
```

Enter runtime-dev container:

```bash
./scripts/enter_runtime_dev.sh
```

Enter simulator container:

```bash
./scripts/enter_sim.sh
```

Run simulator server from host through Compose:

```bash
./scripts/run_sim_server.sh
```

Run runtime loop from host through Compose:

```bash
./scripts/run_runtime_loop.sh
```

## Runtime backend selection

The runtime uses `SORIDORMI_BACKEND`:

```bash
SORIDORMI_BACKEND=sim
```

or later:

```bash
SORIDORMI_BACKEND=hardware
```

Your controller should keep calling the same interface:

```python
state = robot.read_state()
robot.send_motor_command(command)
```

## Jetson notes

For Jetson AGX Orin:

```bash
docker compose -f compose.orin.yaml build runtime
```

For Jetson AGX Thor:

```bash
docker compose -f compose.thor.yaml build runtime
```

The exact Jetson base image depends on your installed JetPack/L4T version. Edit `.env` after `scripts/setup_env.sh` if needed:

```env
RUNTIME_DEV_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
ORIN_RUNTIME_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
THOR_RUNTIME_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
SIM_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
SORIDORMI_ONNXRUNTIME_GPU=1
SORIDORMI_USE_CUDA_PROVIDER=1
```

On Jetson, build the runtime image on the Jetson itself unless you already have a cross-build setup.

For a real Jetson deployment, verify the base image against the JetPack/L4T release installed on the board. The CUDA 13.1 cuDNN image is a good PC development default, but Jetson images may require NVIDIA's Jetson/L4T container images instead of normal desktop CUDA images.

## Important warning

Do not try to make the simulator image identical to the robot image. The simulator PC is usually x86_64 with a desktop NVIDIA GPU, while the robot computer is ARM64 Jetson hardware. Instead, keep the **API identical** and keep the runtime code portable.

## Upstream repositories

This project uses upstream repos as submodules rather than copying their code:

- `apirrone/Open_Duck_Mini`, branch `v2`
- `apirrone/Open_Duck_Mini_Runtime`, branch `v2`
- `apirrone/Open_Duck_Playground`, default branch

## License

This repository skeleton is MIT licensed. Upstream repositories keep their own licenses.
