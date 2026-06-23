# M10 Clearance Refinement Handoff - 2026-06-23

Transfer checkpoint:

```text
branch: main
latest pushed commit: b1a1a9d Add M10 low-clearance guard training evidence
status: M10/G10 still blocked by low-clearance-ratio gate
hardware: excluded for now
human visual review: pending; do not claim without viewer/follow-camera pass
```

Read first in a new workspace:

```text
AGENTS.md
LLM_CONTEXT.md
docs/SORIDORMI_EXECUTION_ROADMAP.md
docs/SORIDORMI_TARGET_AND_ROADMAP.md
docs/M6_SIM_TRAINING_LOOP.md
docs/notes/M10_CLEARANCE_REFINEMENT_HANDOFF_2026-06-23.md
```

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
clearance_cmdmlp_tail_stack_s175_quick
clearance_actionscale_stack_s175_scale026
clearance_actionscale_stack_s177_scale0262
clearance_actionscale_stack_s181_scale0248
clearance_reflex_stack_s183_swinglift
clearance_reflex_stack_s185_earlysoft
clearance_actionscale_stack_s187_scale02618
clearance_reflex_stack_s189_swinggain
clearance_actionscale_ramp_stack_s191_scale0262_ramp05
clearance_startup_tail_stack_s193
clearance_actionscale_preroll_stack_s195_scale0262_preroll25
clearance_actionscale_preroll_stack_s197_scale0263_preroll25
clearance_actionscale_stack_s199_scale026205
clearance_s177_tail_stack_s201
clearance_s177_tail_stack_s203_scale026215
clearance_cmdmlp_lowtail_s205
clearance_s201_microreflex_s207
clearance_s143_refguard_stack_s215
clearance_s143_gateguard_stack_s217
clearance_s143_curvegateguard_stack_s219
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
- `clearance_actionscale_stack_s177_scale0262` improved curve ratio to
  `~0.263` and total distance to `~2.297 m`, but worsened flat/start-stop
  ratios to `~0.319`/`~0.315`; it remained 0/3 and is not the retained best.
- Low-clearance misses are strongly startup/tail clustered. For the `s177`
  action-scale probe, the curve rollout after the initial second is under the
  `0.25` ratio target, while the full rollout still misses at `~0.263`.
- Profile command ramp now survives skill execution, but
  `clearance_actionscale_ramp_stack_s191_scale0262_ramp05` collapsed movement
  in curve (`low ratio 1.0`, distance `~0.019 m`).
- Runtime `swing_clearance_reflex` postprocessing is available as an opt-in
  diagnostic/probe mode, but fixed/sagittal reflex probes through `s189`
  regressed curve ratio and must not be promoted.
- `clearance_s177_tail_stack_s201` trained cleanly from the `s177` near miss,
  but live curve still failed at low-clearance ratio `~0.257` with p50
  `~0.01846 m`, distance `~0.717 m`, and no fall. The tiny `s203` action-scale
  nudge regressed curve to `~0.294`.
- `clearance_cmdmlp_lowtail_s205` used the broader `command_state_mlp` actor and
  a stricter `0.017 m` training target to pressure startup/lower-tail clearance,
  but live curve regressed to low-clearance ratio `~0.338`, p50 `~0.01712 m`,
  distance `~0.606 m`, and no fall.
- `clearance_s201_microreflex_s207` applied a much smaller reflex to the `s201`
  near miss. It also regressed curve to low-clearance ratio `~0.282`, p50
  `~0.01882 m`, distance `~0.694 m`, and no fall.
- `clearance_s143_cmdtail_stack_s211` trained directly on `s143` with a
  command-contact/phase lift actor and aggressive lower-tail penalties. It
  preserved no-fall behavior and total distance, but regressed the full-suite
  max low-clearance ratio to `~0.391`, so reject it.
- `clearance_s143_scenariogate_stack_s213` used scenario-shaped training
  lengths/commands against `s143`. It passed start-stop (`~0.249`
  low-clearance ratio) and improved total distance to `~2.171 m`, but regressed
  flat to `~0.295` and curve to `~0.318`, so reject it. This run tightened the
  reference-comparison helper: a candidate cannot be retained if any required
  scenario regresses low-clearance ratio versus `s143`.
- `clearance_s143_refguard_stack_s215` added training-time low-clearance
  reference penalties against the retained `s143` ratios. It preserved no-fall
  behavior and total distance (`~2.159 m`), but full-suite low-clearance ratios
  were flat `~0.295`, start-stop `~0.271`, and curve `~0.327`; reject it
  because all three regressed against `s143`.
- `clearance_s143_gateguard_stack_s217` used the same trainer guard with all
  references set to the G10 `0.25` gate. It passed start-stop (`~0.245`) with
  no fall and total distance `~2.167 m`, but flat `~0.271` and curve `~0.325`
  failed and regressed against `s143`; reject it.
- `clearance_s143_curvegateguard_stack_s219` corrected the curve training
  sequence to constant yaw from step one. It passed start-stop (`~0.241`) with
  no fall and total distance `~2.166 m`, but flat `~0.275` and curve `~0.340`
  failed and regressed against `s143`; reject it.

Next best work:

1. Use a broader clearance redesign or acquire a higher-clearance teacher.
2. Keep `s143` as the current metric-grounded retained blocked reference.
3. Do not claim human visual review until a direct viewer/follow-camera pass is
   actually performed.

Resume checklist:

```bash
git checkout main
git pull
bash -n scripts/train_residual_policy.sh
docker compose -f compose.sim.yaml run --rm runtime bash -lc 'source /opt/venvs/runtime/bin/activate && pytest -q tests/test_residual_policy_m619_m621.py tests/test_residual_scripts_m621.py tests/test_m10_clearance_readiness.py'
./scripts/validate_policy_profiles.sh configs/policies/clearance_liftscale_stack_s143_step090_offset005.yaml configs/policies/clearance_s143_refguard_stack_s215.yaml configs/policies/clearance_s143_gateguard_stack_s217.yaml configs/policies/clearance_s143_curvegateguard_stack_s219.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
```

Before any new live training or suite evaluation, start MuJoCo explicitly:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile context_stage1_three_scenario_10ep_e80 --no-viewer
```

Use the retained-reference comparator for every exploratory candidate before
retaining it:

```bash
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile <candidate_profile> --output-dir artifacts/scenario_eval/<candidate_profile> --json
./scripts/analyze_clearance_readiness.sh \
  --profile-name <candidate_profile> \
  --suite-dir artifacts/scenario_eval/<candidate_profile> \
  --reference-profile-name clearance_liftscale_stack_s143_step090_offset005 \
  --reference-suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 \
  --output-dir artifacts/clearance_readiness/<candidate_profile> \
  --json \
  --require-reference-improvement
```

Optional human visual pass, only after a candidate clears the metric gates:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile <candidate_profile> --viewer --follow-camera
```
