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

## Jetson host

Build runtime images on the Jetson itself first. Cross-building is possible but adds complexity.

Match `ORIN_RUNTIME_BASE` or `THOR_RUNTIME_BASE` to the JetPack/L4T release installed on the board.
