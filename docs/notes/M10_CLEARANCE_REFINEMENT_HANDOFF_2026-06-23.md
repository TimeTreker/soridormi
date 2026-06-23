# M10 Clearance Refinement Handoff - 2026-06-23

Current best retained blocked profile:

```text
clearance_liftscale_stack_s143_step090_offset005
```

It reuses `/data/rl_finetune/clearance_liftscale_stack_s127/residual_policy.onnx`
with:

```text
phase.step_increment: 0.9
phase.offset: 0.05
residual_scale: 0.16
```

Full required suite:

```text
result: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance ~= 2.150m
flat:       p50 ~= 0.01858m, low-clearance ratio ~= 0.268
start-stop: p50 ~= 0.01861m, low-clearance ratio ~= 0.257
curve:      p50 ~= 0.01781m, low-clearance ratio ~= 0.308
falls:      none
```

Readiness was regenerated with:

```bash
./scripts/analyze_clearance_readiness.sh \
  --profile-name clearance_liftscale_stack_s143_step090_offset005 \
  --suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 \
  --output-dir artifacts/clearance_readiness/clearance_liftscale_stack_s143_step090_offset005 \
  --json
```

Do not promote:

```text
clearance_cmdtail_stack_s133
clearance_liftscale_stack_s135_scale018
clearance_liftscale_stack_s137_step090
clearance_liftscale_stack_s139_step080
clearance_liftscale_stack_s141_step090_scale018
clearance_liftscale_stack_s145_step090_offset010
clearance_liftscale_stack_s147_step090_offset005_scale018
clearance_liftscale_stack_s149_step085_offset005
clearance_liftscale_stack_s151_step090_offset004
clearance_liftscale_stack_s153_step090_offset006
clearance_liftscale_stack_s155_step092_offset005
clearance_liftscale_stack_s157_step095_offset005
clearance_curve_tail_stack_s159
clearance_curve_direct_stack_s161
clearance_cmdcurve_direct_stack_s163
clearance_liftscale_stack_s165_step090_offset005_scale015
clearance_liftscale_stack_s167_step090_offset005_scale014
clearance_liftscale_stack_s169_step090_offset005_kneegain
clearance_harmonic_direct_stack_s171
clearance_harmonic_aggressive_stack_s173
```

Useful findings:

- `phase.step_increment=0.9` was the major improvement over `s127`.
- `phase.offset=0.05` improved curve versus offset `0.0`; offsets `0.04`,
  `0.06`, and `0.10` regressed.
- `step_increment=0.8`, `0.85`, `0.92`, and `0.95` regressed.
- Lowering deployed residual scale to `0.15` or `0.14` regressed curve.
- A small hip/knee postprocess gain regressed curve and should not be used as
  a promotion path.
- Small contact/phase, command-contact/phase, and harmonic continuations on
  top of `s143` did not break the `~0.31` curve low-ratio plateau.

Next best work:

1. Use a broader clearance redesign or acquire a higher-clearance teacher.
2. Keep `s143` as the current metric-grounded retained blocked reference.
3. Do not claim human visual review until a direct viewer/follow-camera pass is
   actually performed.
