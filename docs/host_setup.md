# Host setup

## PC simulation host

Install:

- Docker Engine
- Docker Compose v2
- NVIDIA driver
- NVIDIA Container Toolkit

Test GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu24.04 nvidia-smi
```

Pull your simulator base image:

```bash
docker pull nvidia/cuda:12.8.1-devel-ubuntu24.04
```

For X11 GUI:

```bash
xhost +local:docker
```

## Jetson host

Build runtime images on the Jetson itself first. Cross-building is possible but adds complexity.

Match `ORIN_RUNTIME_BASE` or `THOR_RUNTIME_BASE` to the JetPack/L4T release installed on the board.
