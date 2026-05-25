# Troubleshooting

## MuJoCo GUI does not open

On host:

```bash
xhost +local:docker
```

Inside sim container:

```bash
echo $DISPLAY
ls /tmp/.X11-unix
```

## JAX or CUDA grabs all GPU memory

The Compose file sets:

```env
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Runtime cannot connect to simulator

Make sure simulator server is running:

```bash
./scripts/run_sim_server.sh
```

Then start runtime loop:

```bash
./scripts/run_runtime_loop.sh
```

Both containers use host networking by default.

## Jetson base image fails

Check that the base image matches your JetPack/L4T release. Edit `.env` and rebuild.

## Docker build fails: `GID '1000' already exists`

This project intentionally maps the container user to your host UID/GID. Some base images already contain a group or user with UID/GID 1000. The current Dockerfiles handle that case by reusing the existing numeric group or renaming the existing UID owner to `chromie`.

Regenerate `.env` if needed:

```bash
./scripts/setup_env.sh
```

Then rebuild:

```bash
docker compose -f compose.sim.yaml build --no-cache
```


## ONNX Runtime GPU install/provider issues

The PC runtime-dev container defaults to `SORIDORMI_ONNXRUNTIME_GPU=1`, which installs the `runtime-gpu` extra and asks ONNX Runtime to prefer `CUDAExecutionProvider`. If that package or provider is incompatible with your host driver/CUDA stack, set this in `.env`:

```env
SORIDORMI_ONNXRUNTIME_GPU=0
SORIDORMI_USE_CUDA_PROVIDER=0
```

Then remove the runtime venv volume or rerun `bootstrap_runtime` in a fresh runtime container.
