# Workflow

## PC simulation loop

Terminal 1:

```bash
./scripts/run_sim_server.sh
```

Terminal 2:

```bash
./scripts/run_runtime_loop.sh
```

## Development steps

1. Keep API changes in `src/soridormi_api`.
2. Implement sim behavior in `src/soridormi_sim`.
3. Keep runtime code in `src/soridormi_runtime`.
4. Keep hardware integration behind `HardwareRobot`.
5. Avoid importing MuJoCo in runtime code.


## GPU inference during PC simulation

The PC runtime-dev image uses the CUDA 13.1 cuDNN base image by default and is given GPU access in `compose.sim.yaml`. Policy loading prefers ONNX Runtime's `CUDAExecutionProvider` when available, then falls back to CPU.
