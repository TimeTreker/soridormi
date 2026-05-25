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
