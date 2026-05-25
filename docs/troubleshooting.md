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

## Docker error: `Unable to find group render`

Do not use group names in Compose. This project uses numeric group IDs instead:

```yaml
group_add:
  - "${VIDEO_GID:-44}"
  - "${RENDER_GID:-109}"
```

Regenerate `.env` so the values match your host:

```bash
./scripts/setup_env.sh
```

Then retry:

```bash
./scripts/enter_sim.sh
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

The PC runtime-dev image defaults to the `runtime-gpu` extra and asks ONNX Runtime to prefer `CUDAExecutionProvider`. If that package or provider is incompatible with your host driver/CUDA stack, set this in `.env`:

```env
RUNTIME_DEV_EXTRA=runtime
SORIDORMI_ONNXRUNTIME_GPU=0
SORIDORMI_USE_CUDA_PROVIDER=0
```

Then rebuild:

```bash
./scripts/build_sim.sh
```

## I changed Python dependencies but the container still uses the old environment

The Python environment is built into the image. Rebuild after dependency changes:

```bash
./scripts/build_sim.sh
```

For a full clean rebuild:

```bash
docker compose -f compose.sim.yaml build --no-cache runtime sim
```
