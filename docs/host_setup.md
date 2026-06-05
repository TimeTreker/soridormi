# Host setup

## PC simulation host

Install:

- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

Test GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04 nvidia-smi
```

The default PC development base image includes CUDA, development headers, and cuDNN:

```text
nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
```

Use the same image for the runtime-dev container and the simulator container on your PC.

Pull your simulator/runtime-dev base image:

```bash
docker pull nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
```

Generate local settings before Compose commands:

```bash
./scripts/setup_env.sh
```

This automatically detects host UID/GID and GPU device group IDs such as `video` and `render` and writes them into `.env`.

For X11 GUI:

```bash
xhost +local:docker
```

If the MuJoCo viewer does not open, first request it explicitly:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

Then confirm the host exposes X11 to Docker:

```bash
echo "$DISPLAY"
ls /tmp/.X11-unix
```

On Wayland desktops, make sure XWayland is available or run from an X11
session.

## Common host fixes

If Compose reports `Unable to find group render`, regenerate `.env` so the
numeric device group IDs match your host:

```bash
./scripts/setup_env.sh
```

If ONNX Runtime GPU providers fail on your PC driver/CUDA stack, use CPU ONNX
Runtime in the runtime-dev image:

```env
RUNTIME_DEV_EXTRA=runtime
SORIDORMI_ONNXRUNTIME_GPU=0
SORIDORMI_USE_CUDA_PROVIDER=0
```

Then rebuild:

```bash
./scripts/build_sim.sh
```

If Python dependency changes are not visible in a container, rebuild the image:

```bash
./scripts/build_sim.sh
```

For a full clean rebuild:

```bash
docker compose -f compose.sim.yaml build --no-cache runtime sim
```

Runtime tools connect to the simulator through host networking by default. If a
runtime command cannot connect, start the MuJoCo server explicitly first:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

## Jetson host

Build runtime images on the Jetson itself first. Cross-building is possible but adds complexity.

Match `ORIN_RUNTIME_BASE` or `THOR_RUNTIME_BASE` to the JetPack/L4T release installed on the board.

If a Jetson base image fails, verify that the base image matches the installed
JetPack/L4T release, update `.env`, and rebuild the runtime image.
