# M4.10 Official Motor-Target Replay

M4.10 isolates the remaining Open Duck compatibility problem.

The official baseline already proves that `BEST_WALK_ONNX_2.onnx` can walk forward
in the Docker/MuJoCo environment. Soridormi's ONNX runtime now matches command
and phase, but it still produces less forward displacement.

The replay tool answers one question:

> If Soridormi's MuJoCo backend receives the exact motor targets produced by the
official runner, does it reproduce the official motion?

Run:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/replay_latest_official_targets.sh
./scripts/compare_official_replay_trace.sh
```

Interpretation:

- If replay displacement is close to the official displacement, the MuJoCo backend
  dynamics are compatible and the mismatch is in Soridormi's runtime
  observation/action generation.
- If replay displacement is still much smaller, the mismatch is in the simulator
  backend/reset/control stepping path, not in ONNX inference.

This is an isolation test. It does not replace the ONNX policy runtime and does
not introduce a new controller.
