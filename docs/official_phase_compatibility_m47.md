# M4.7 Official Open Duck phase compatibility

The M4.6 trace comparison showed the largest Soridormi-vs-official mismatch in the ONNX observation was the imitation phase segment.

Open Duck's official MuJoCo inference loop advances `imitation_i` before building the policy observation for a policy step, then computes:

```text
phase = [
  cos(imitation_i / PRM.nb_steps_in_period * 2*pi),
  sin(imitation_i / PRM.nb_steps_in_period * 2*pi),
]
```

`PRM.nb_steps_in_period` is loaded from `polynomial_coefficients.pkl` as `int(period * fps)`.

M4.7 changes Soridormi's phase generator to match that behavior:

- step mode now advances before returning the phase vector;
- policy profiles can set `phase.period_steps: auto`;
- runtime reads `phase.reference_data` and extracts `period * fps` directly from the Open Duck reference pickle;
- logs include `period_source` so experiments show whether the period came from reference data or fallback.

## Expected runtime log

When using `open_duck_forward`, the controller description should show something like:

```text
phase: {
  mode: step,
  period_steps: <loaded from reference data>,
  period_source: reference_data,
  reference_data: /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl
}
```

## Test sequence

```bash
./scripts/run_official_forward_baseline.sh
./scripts/run_official_compatible_policy_server.sh open_duck_forward
./scripts/run_policy_experiment.sh open_duck_forward
./scripts/compare_latest_official_soridormi_trace.sh
```

The imitation phase mean error should drop compared with M4.6. If phase improves but the robot still does not step forward, the next mismatch to port is foot-contact timing/extraction.


## M4.8 note: runtime reference-data mount

M4.7 introduced automatic Open Duck phase-period loading from
`polynomial_coefficients.pkl`, but the runtime container must be able to read
`/workspaces/Open_Duck_Playground`. M4.8 mounts `./workspace/Open_Duck_Playground`
into the runtime service as well as the simulator service.

For official policy profiles, `phase.require_reference_data: true` is enabled so
Soridormi fails fast instead of silently falling back to `period_steps=50`. If
the controller description shows `period_source: fallback_50`, the runtime is
not official-compatible. It should show `period_source: reference_data`.
