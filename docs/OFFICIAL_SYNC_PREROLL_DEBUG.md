# Official synchronous pre-roll parity debug

The official Open Duck `MjInfer` loop and the Soridormi runtime both use the
same ONNX policy for `open_duck_forward`, but they do not enter the first policy
step at the same instant unless Soridormi pre-rolls the synchronous simulator.

Official startup order:

1. Create MuJoCo data.
2. Run one `mj_step` from default reset data.
3. Copy `keyframe("home").qpos` and `keyframe("home").ctrl` into MuJoCo data.
4. In the viewer/run loop, step MuJoCo for one policy-control interval
   (`decimation=10`, `sim_dt=0.002`) before the first ONNX observation.

Soridormi sync-step startup previously read `RobotState` immediately after reset
and fed that into `ObservationBuilder`. That can give the policy stale or too-early
sensor/contact data on policy step 0, which is enough to start a divergent
"wiggle but no forward displacement" rollout even though the official baseline
walks.

`open_duck_forward` now exports:

```text
SORIDORMI_SIM_SYNC_STEP=1
SORIDORMI_SIM_PREROLL_STEPS=1
```

The runtime implements this by sending a hold-current command using
`RobotState.actuator_ctrl` when available, falling back to current joint positions,
then stepping the simulator once before the first policy inference.

## Validation

Start MuJoCo with explicit backend/profile and optional viewer:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

Run the Soridormi smoke rollout. For parity debugging, force JSONL logging from
the smoke wrapper because the `open_duck_forward` policy profile defaults to
MCAP logging:

```bash
./scripts/run_policy_rollout_smoke.sh open_duck_forward \
  --steps 1000 \
  --log-format jsonl \
  --log-prefix parity_open_duck_forward \
  --log-dir /data/logs
```

The wrapper must print `Runtime log format override: jsonl`, and the runtime
inside the container must print:

```text
Runtime log: enabled=1 format=jsonl dir=/data/logs prefix=parity_open_duck_forward every_n=1
Soridormi JSONL runtime logger: /data/logs/parity_open_duck_forward_...jsonl
```

Then the host-side log should exist under `data/logs/`.

The runtime should also print:

```text
Sync step: 1
Sync pre-roll steps: 1
Simulator sync pre-roll API steps: 1
```

If the robot still only wiggles, do not collect teacher data yet. Run an
official-vs-Soridormi trace comparison and fix the largest observation/control
parity error first:

```bash
SORIDORMI_LOG="$(find data/logs -type f -name 'parity_open_duck_forward*.jsonl' | sort | tail -1)"
test -n "${SORIDORMI_LOG}"

PYTHONPATH=src python -m soridormi_runtime.compare_official_soridormi_trace \
  --official data/official_baseline/latest_official_baseline.json \
  --soridormi "${SORIDORMI_LOG}" \
  --steps 100
```
