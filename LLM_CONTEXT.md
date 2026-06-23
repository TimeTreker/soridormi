# LLM_CONTEXT.md

Compact handoff for starting a new LLM session on Soridormi.

## Project Summary

Soridormi is a sim-to-real humanoid robot stack for Open Duck Mini v2. It
separates runtime, simulator, and shared API so one policy runtime can talk to
MuJoCo now and hardware later. The current project direction is
scenario-aware, context-conditioned locomotion data and behavior cloning in
MuJoCo.

Soridormi is the robot cerebellum/body runtime. Chromie is the robot brain in
`TimeTreker/chromie.git` on `main`. Chromie talks with people, understands
intent, plans high-level behavior, and chooses skills. Soridormi validates and
executes body skills safely in MuJoCo or hardware.

## Current Focus

Active direction: M10 runtime context policy plumbing after M9 Stage 1 offline
context BC.

Low-level policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Do not pass raw natural language into the low-level policy. A planner or skill
router may choose structured fields such as `skill_id`, velocity command,
terrain type, or obstacle context.

Current offline Stage 1 trainer input mode:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
```

Use:

```bash
./scripts/train_behavior_clone.sh PREPARED_CONTEXT_DATASET --input-mode context_stage1_command
./scripts/train_neural_behavior_clone.sh PREPARED_CONTEXT_DATASET --input-mode context_stage1_command --profile-name context_stage1_candidate --force-profile
```

Context-mode neural training can now share the same Stage 1 feature definition
with runtime policy input plumbing. A context model can be exported as a
`[1, 104]` ONNX/profile, but promotion still requires model validation and
MuJoCo rollout evidence.

Current M10 candidates:

```text
dataset: /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1_10ep/prepared_manifest.json

profile: context_stage1_flat_walk_v1_10ep
model: /data/training_runs/context_stage1_flat_walk_v1_10ep_neural_bc_m10/neural_behavior_clone.onnx

profile: context_stage1_flat_walk_v1_10ep_e80
model: /data/training_runs/context_stage1_flat_walk_v1_10ep_neural_bc_m10_e80/neural_behavior_clone.onnx

profile: context_stage1_three_scenario_10ep_e80
model: /data/training_runs/context_stage1_three_scenario_10ep_neural_bc_m10_e80/neural_behavior_clone.onnx
```

Current retained clearance-refinement candidate:

```text
profile: clearance_liftscale_stack_s143_step090_offset005
status: 0/3 scenarios pass, BLOCKED_BY_CLEARANCE_GATE
total distance ~= 2.150m
flat:       p50 ~= 0.01858m, low-clearance ratio ~= 0.268
start-stop: p50 ~= 0.01861m, low-clearance ratio ~= 0.257
curve:      p50 ~= 0.01781m, low-clearance ratio ~= 0.308
falls:      none
```

Initial candidate checkpoint:

```text
profile: context_stage1_flat_walk_v1_10ep
check_policy_model: OK
offline evaluation: test MAE ~= 0.00953, val MAE ~= 0.0286
200-step MuJoCo smoke: OK, 104D policy input, no resets, forward_x ~= 0.259 m
```

Scenario-suite checkpoint:

```text
context_stage1_flat_walk_v1_10ep:
  suite: flat_walk_varied_speed_v1, start_stop_velocity_ramp_v1, curve_turn_walk_v1
  result: FAIL, 0/3 scenarios accepted
  total_forward_distance_m ~= 0.0419
  mean_forward_speed_mps ~= 0.00251
  fallen_count: 0
open_duck_forward teacher baseline:
  result: PASS, 3/3 scenarios accepted
  total_forward_distance_m ~= 0.728
  mean_forward_speed_mps ~= 0.0435
```

Interpretation: the context candidate is runnable and remains upright, but it
is not promotable. It mostly stands still at the scenario nominal commands even
though the official teacher passes the same suite.

E80 candidate checkpoint:

```text
profile: context_stage1_flat_walk_v1_10ep_e80
check_policy_model: OK
offline evaluation: test MAE ~= 0.00596, val MAE ~= 0.01709
200-step MuJoCo command-response smoke:
  vx=0.125 -> forward_x ~= 0.226 m, mean speed ~= 0.0568 m/s
  vx=0.140 -> forward_x ~= 0.308 m, mean speed ~= 0.0774 m/s
  vx=0.150 -> forward_x ~= 0.348 m, mean speed ~= 0.0874 m/s
scenario suite: FAIL, 2/3 scenarios accepted
  flat_walk_varied_speed_v1: PASS
  start_stop_velocity_ramp_v1: PASS
  curve_turn_walk_v1: FAIL
  total_forward_distance_m ~= 0.502
  mean_forward_speed_mps ~= 0.0307
  fallen_count: 0
```

Interpretation: the E80 context candidate is a better runnable experimental
profile and fixes the low-speed command threshold seen in the initial model,
but it is still not promotable because the curve/turning scenario gets stuck.
That recommendation was superseded by the three-scenario candidate below,
which added curve/yaw coverage and passed the initial flat/start-stop/curve
suite before the remaining clearance blocker was identified.

Historical three-scenario candidate checkpoint:

```text
profile: context_stage1_three_scenario_10ep_e80
dataset: /data/training_datasets/context_bc/prepared/context_stage1_three_scenario_10ep/prepared_manifest.json
raw data:
  flat_walk_varied_speed_v1_10ep: 3000 samples
  start_stop_velocity_ramp_v1_10ep: 3000 samples
  curve_turn_walk_v1_10ep: 3000 samples
prepared splits: train 7200, val 900, test 900
check_policy_model: OK
offline evaluation:
  train MAE ~= 0.00720
  val MAE ~= 0.01101
  test MAE ~= 0.01244
scenario suite: PASS, 3/3 scenarios accepted
  flat_walk_varied_speed_v1: forward_distance ~= 0.312 m, mean speed ~= 0.0627 m/s
  start_stop_velocity_ramp_v1: forward_distance ~= 0.267 m, mean speed ~= 0.0411 m/s
  curve_turn_walk_v1: forward_distance ~= 0.155 m, mean speed ~= 0.0282 m/s
  total_forward_distance_m ~= 0.733
  mean_forward_speed_mps ~= 0.0440
  fallen_count: 0
```

Interpretation: this historical three-scenario candidate was the best retained
MuJoCo candidate. It passed the same flat/start-stop/curve suite that the
flat-only models failed, including the curve case. Do not call it
hardware-ready: all three scenario reports still warned about low swing
clearance, so the next checkpoint was visual/follow-camera inspection and
clearance-focused refinement.

Local 2026-06-22 regeneration from the retained M9 dataset:

```text
profile: context_stage1_three_scenario_10ep_e80
prepared dataset: /data/training_datasets/context_bc/prepared/context_stage1_three_scenario_10ep/prepared_manifest.json
training output: /data/training_runs/context_stage1_three_scenario_10ep_neural_bc_m10_e80/neural_behavior_clone.onnx
check_policy_model: OK
offline evaluation:
  train MAE ~= 0.01088
  val MAE ~= 0.01179
  test MAE ~= 0.01167
scenario suite: FAIL, 0/3 scenarios accepted
  flat_walk_varied_speed_v1: forward_distance ~= 0.318 m, p50 clearance ~= 0.0101 m, stuck ~= 0.016
  start_stop_velocity_ramp_v1: forward_distance ~= 0.322 m, p50 clearance ~= 0.0086 m, stuck ~= 0.012
  curve_turn_walk_v1: forward_distance ~= 0.152 m, p50 clearance ~= 0.0065 m, stuck ~= 0.277
  fallen_count: 0
```

Interpretation: the regenerated ONNX is runnable but is not equivalent to the
historical retained candidate. It remains blocked before promotion and before
visual-review claims.

2026-06-22 artifact restoration:

```text
canonical profile: context_stage1_three_scenario_10ep_e80
restored ONNX sha256: 2a7e41afe855702638aed56ec32e0f5e067a6b76fdcd76af4d43a101191730b7
preserved regenerated ONNX: data/training_runs/context_stage1_three_scenario_10ep_neural_bc_m10_e80_regenerated_20260622/
preserved regenerated suite: artifacts/scenario_eval/context_stage1_three_scenario_10ep_e80_suite_regenerated_20260622/
restored warm-start residual: /data/rl_finetune/m10_command_state_mlp_cem4x14_s79
```

The restored historical E80 reproduces the old movement metrics, but the
current acceptance gate now includes absolute clearance checks. Under the
current gate it is still `0/3`:

```text
flat:       distance ~= 0.312m, p50 clearance ~= 0.01023m, low-clearance ratio 1.0
start-stop: distance ~= 0.267m, p50 clearance ~= 0.00759m, low-clearance ratio 1.0
curve:      distance ~= 0.155m, p50 clearance ~= 0.00632m, low-clearance ratio 1.0
```

2026-06-22 visual-review status: follow-camera commands and visual-review
templates exist, and a metric-grounded Codex review artifact records the
clearance failure, but a direct human GUI visual pass is still pending before
promotion. The rollouts stayed upright but reproduced the clearance failure:

```text
flat:       p50 clearance 0.01006m, low-clearance ratio 1.0
start-stop: p50 clearance 0.00855m, low-clearance ratio 1.0
curve:      p50 clearance 0.00649m, low-clearance ratio 1.0
```

The rebuilt clearance package includes
`artifacts/clearance_evidence/context_stage1_three_scenario_10ep_e80/visual_review.json`
and remains `BLOCKED_BY_CLEARANCE_READINESS`.

2026-06-22 clearance-refinement probe:
`/data/rl_finetune/clearance_gap_probe_s91` trained successfully from the local
E80 teacher without the missing historical warm-start checkpoint, but selected
the zero residual (`Best parameter abs max: 0`). Treat it as a workflow probe,
not a candidate.

2026-06-22 restored warm-start clearance candidate:
`clearance_gap_sequence_restored_s83` was trained from the restored historical
E80 teacher with the restored `m10_command_state_mlp_cem4x14_s79` checkpoint.
Its profile/model contract passes, but the required scenario suite remains
`0/3` because all scenarios fail clearance:

```text
flat:       distance ~= 0.471m, p50 clearance ~= 0.01344m, low-clearance ratio ~= 0.752
start-stop: distance ~= 0.501m, p50 clearance ~= 0.01148m, low-clearance ratio ~= 0.985
curve:      distance ~= 0.236m, p50 clearance ~= 0.00863m, low-clearance ratio ~= 1.000
```

Do not promote `clearance_gap_sequence_restored_s83`. It improves movement over
restored E80 but does not beat the older retained
`m10_command_state_mlp_cem4x14_s79` residual, which remains the best retained
clearance-refinement reference:

```text
m10_command_state_mlp_cem4x14_s79:
  flat p50 clearance ~= 0.01471m, low-clearance ratio ~= 0.528
  start-stop p50 clearance ~= 0.01152m, low-clearance ratio ~= 0.971
  curve p50 clearance ~= 0.01025m, low-clearance ratio ~= 0.973
  required suite: 0/3 under current G10 clearance gate
```

Important correction found after `s83`: the clearance wrapper was intended to
preserve the retained warm-start candidate, but it hardcoded
`--residual-scale 0.05` while `m10_command_state_mlp_cem4x14_s79` was trained
and evaluated with `residual_scale 0.1`. The wrapper now defaults to `0.1` and
exposes `--residual-scale` so future warm starts preserve the checkpoint unless
the caller deliberately changes scale.

2026-06-22 scale-preserved and low-ratio continuation candidates:

```text
clearance_gap_sequence_scale_preserved_s89:
  initial checkpoint: m10_command_state_mlp_cem4x14_s79
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.673m
  flat:       p50 ~= 0.01532m, low-clearance ratio ~= 0.445
  start-stop: p50 ~= 0.01481m, low-clearance ratio ~= 0.514
  curve:      p50 ~= 0.01192m, low-clearance ratio ~= 0.992
```

`clearance_lowratio_sequence_s97` warm-started from `s89` with stronger
low-clearance pressure. It improved total distance, stuck ratio, and curve
clearance over `s79`, `s83`, and `s89`, but still failed the low-clearance-ratio
gate in all three scenarios and missed the start/stop and curve p50 gates.

```text
clearance_lowratio_sequence_s97:
  initial checkpoint: clearance_gap_sequence_scale_preserved_s89
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.729m
  flat:       p50 ~= 0.01560m, low-clearance ratio ~= 0.437
  start-stop: p50 ~= 0.01466m, low-clearance ratio ~= 0.520
  curve:      p50 ~= 0.01341m, low-clearance ratio ~= 0.754
```

`clearance_lowratio_turnfocus_s101` was a stricter turn-focused continuation
from `s97`, but the exported ONNX was byte-identical to `s97`; treat it as a
failed probe, not a retained candidate.

`clearance_lowratio_refine_s103` warm-started from `s97` with the retained
`residual_scale 0.1`, tighter CEM search (`initial_std 0.12`), and moderately
stronger clearance/low-ratio pressure. It improved total distance, start-stop
p50, curve p50, and max low-clearance ratio over `s97`, but remained blocked
by the G10 low-clearance-ratio gate in all three scenarios and missed curve p50
clearance.

```text
clearance_lowratio_refine_s103:
  initial checkpoint: clearance_lowratio_sequence_s97
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.838m
  flat:       p50 ~= 0.01634m, low-clearance ratio ~= 0.438
  start-stop: p50 ~= 0.01559m, low-clearance ratio ~= 0.456
  curve:      p50 ~= 0.01451m, low-clearance ratio ~= 0.662
```

`clearance_lowratio_multicmd_s107` warm-started from `s103` with multiple
fixed-command objectives and stronger low-clearance pressure. It improves total
distance, start-stop low-clearance ratio, curve p50 clearance, and max
low-clearance ratio over `s103`, but still fails the G10 low-clearance ratio
gate in all three scenarios and misses curve p50 clearance.

```text
clearance_lowratio_multicmd_s107:
  initial checkpoint: clearance_lowratio_refine_s103
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.903m
  flat:       p50 ~= 0.01614m, low-clearance ratio ~= 0.420
  start-stop: p50 ~= 0.01578m, low-clearance ratio ~= 0.425
  curve:      p50 ~= 0.01456m, low-clearance ratio ~= 0.519
```

`clearance_lowratio_curvepush_s109` was a curve-focused continuation from
`s107`. It slightly improved curve p50 but worsened total distance and max
low-clearance ratio, so treat it as a rejected probe.

`clearance_lowratio_gatepush_s111` warm-started from `s107` with stronger
low-clearance pressure and additional ramp/turn objectives. Treat `s111` as the
current best retained clearance-refinement candidate, not promotable. It
improves total distance and reduces low-clearance ratio in all three scenarios
relative to `s107`; p50 clearance is now above `0.015 m` in all three scenarios.
It remains blocked because all scenarios still exceed the `0.25` low-clearance
ratio gate.

```text
clearance_lowratio_gatepush_s111:
  initial checkpoint: clearance_lowratio_multicmd_s107
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.928m
  flat:       p50 ~= 0.01679m, low-clearance ratio ~= 0.388
  start-stop: p50 ~= 0.01626m, low-clearance ratio ~= 0.423
  curve:      p50 ~= 0.01506m, low-clearance ratio ~= 0.496
```

`clearance_lowratio_targetlift_s113` raised the training clearance target to
`0.0165 m`, but exported an ONNX byte-identical to `s111`; treat it as a
rejected probe.

The residual trainer now exposes `--no-zero-candidate`, which disables forcing
the current CEM mean into each generation when probing beyond a warm-start
checkpoint. `clearance_lowratio_forced_s115` used that flag through the
clearance wrapper and produced a different ONNX, but regressed training
clearance metrics versus `s111`; reject it. `clearance_lowratio_suitecmd_s117`
used the same flag with direct three-command suite objectives. It reduced curve
low-clearance ratio slightly in the full suite, but regressed flat/start
low-clearance ratio and total distance, so reject it.

The residual trainer also exposes a lower-tail clearance objective:
`--episodic-clearance-quantile` and
`--episodic-clearance-quantile-gap-weight`. It penalizes normalized shortfall at
the chosen clearance quantile, intended to attack low-clearance ratio after p50
clearance is already above target. `clearance_lowratio_quantile_s119`
warm-started from `s111` with quantile `0.25` and gap weight `10`. It is a
distinct ONNX and remains blocked by the G10 clearance gate. Treat it as a
metric-only blocked candidate, not a promotion. It improves worst-case
low-clearance ratio and total distance versus `s111`, but worsens flat
low-clearance ratio.

```text
clearance_lowratio_quantile_s119:
  initial checkpoint: clearance_lowratio_gatepush_s111
  residual_scale: 0.1
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 1.938m
  flat:       p50 ~= 0.01658m, low-clearance ratio ~= 0.408
  start-stop: p50 ~= 0.01632m, low-clearance ratio ~= 0.421
  curve:      p50 ~= 0.01516m, low-clearance ratio ~= 0.473
  falls:      none
```

2026-06-23 stacked clearance refinement:
the residual runtime now supports stacked residual teachers and compact
swing-lift actors. `clearance_contactlift_stack_s121` first stacked a
contact/phase lift residual on `s111`; it remained blocked but improved total
distance to about `2.028 m`, kept all p50 clearances above `0.015 m`, and
reduced max low-clearance ratio to about `0.460`. `clearance_liftscale_stack_s127`
then improved total distance to about `2.189 m` and reduced max low-clearance
ratio to about `0.409`.

```text
clearance_liftscale_stack_s127:
  teacher profile: clearance_contactlift_stack_s121
  actor kind: contact_phase_lift
  residual_scale: 0.16
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 2.189m
  flat:       p50 ~= 0.01742m, low-clearance ratio ~= 0.353
  start-stop: p50 ~= 0.01716m, low-clearance ratio ~= 0.345
  curve:      p50 ~= 0.01616m, low-clearance ratio ~= 0.409
  falls:      none
```

2026-06-23 phase-timing continuation:
profile-level phase timing moved the best blocked candidate to
`clearance_liftscale_stack_s143_step090_offset005`. It reuses the `s127` ONNX
with `step_increment=0.9`, `offset=0.05`, and `residual_scale=0.16`.

```text
clearance_liftscale_stack_s143_step090_offset005:
  suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
  total distance ~= 2.150m
  flat:       p50 ~= 0.01858m, low-clearance ratio ~= 0.268
  start-stop: p50 ~= 0.01861m, low-clearance ratio ~= 0.257
  curve:      p50 ~= 0.01781m, low-clearance ratio ~= 0.308
  falls:      none
```

`s143` improves the maximum low-clearance ratio from `s127`'s `~0.409` to
`~0.308`; p50 clearance and no-fall behavior are healthy, but G10 still fails
because all three scenarios exceed the `0.25` low-clearance-ratio limit.

2026-06-23 post-`s143` low-ratio probes:
profile-level action-scale, pre-roll, command-ramp, startup-tail training, and
opt-in clearance-reflex probes did not pass G10. `clearance_actionscale_stack_s177_scale0262`
is the best full-suite action-scale near miss, with total distance `~2.297 m`,
no falls, and curve low-clearance ratio `~0.263`, but it worsens flat/start-stop
to `~0.319`/`~0.315`, so it is not the retained best. A learned continuation
from that profile, `clearance_s177_tail_stack_s201`, trained cleanly but live
curve still failed at low-clearance ratio `~0.257`; the tiny
`clearance_s177_tail_stack_s203_scale026215` action-scale nudge regressed curve
to `~0.294`. A broader `command_state_mlp` lower-tail run,
`clearance_cmdmlp_lowtail_s205`, also regressed live curve to `~0.338`; a tiny
micro-reflex on the `s201` near miss, `clearance_s201_microreflex_s207`,
regressed curve to `~0.282`.

Two implementation findings are now preserved for later M10 work:
`skill_execution.plan_shell_exports()` no longer forces
`SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE=0`, so profile-level command ramp can
be exercised in scenario rollouts; and `ActionPostprocessor` has an opt-in
`swing_clearance_reflex` diagnostic mode that uses feet contact/position state.
The ramp and reflex profiles tried so far were metric-grounded rejects, not
promotion candidates.

Do not promote the intermediate or rejected probes:
`clearance_contactlift_stack_s121`, `clearance_contactlift_stack_s123`,
`clearance_cmdlift_stack_s125`, `clearance_liftscale_stack_s129`, or
`clearance_harmonic_stack_s131`, plus the phase/training probes
`clearance_cmdtail_stack_s133`, `clearance_liftscale_stack_s135_scale018`,
`clearance_liftscale_stack_s137_step090`, `clearance_liftscale_stack_s139_step080`,
`clearance_liftscale_stack_s141_step090_scale018`,
`clearance_liftscale_stack_s145_step090_offset010`,
`clearance_liftscale_stack_s147_step090_offset005_scale018`,
`clearance_liftscale_stack_s149_step085_offset005`,
`clearance_liftscale_stack_s151_step090_offset004`,
`clearance_liftscale_stack_s153_step090_offset006`,
`clearance_liftscale_stack_s155_step092_offset005`,
`clearance_liftscale_stack_s157_step095_offset005`,
`clearance_curve_tail_stack_s159`, `clearance_curve_direct_stack_s161`,
`clearance_cmdcurve_direct_stack_s163`,
`clearance_liftscale_stack_s165_step090_offset005_scale015`,
`clearance_liftscale_stack_s167_step090_offset005_scale014`,
`clearance_liftscale_stack_s169_step090_offset005_kneegain`,
`clearance_harmonic_direct_stack_s171`, and
`clearance_harmonic_aggressive_stack_s173`; plus post-`s143` probes
`clearance_cmdmlp_tail_stack_s175_quick`,
`clearance_actionscale_stack_s175_scale026`,
`clearance_actionscale_stack_s177_scale0262`,
`clearance_actionscale_stack_s181_scale0248`,
`clearance_reflex_stack_s183_swinglift`,
`clearance_reflex_stack_s185_earlysoft`,
`clearance_actionscale_stack_s187_scale02618`,
`clearance_reflex_stack_s189_swinggain`,
`clearance_actionscale_ramp_stack_s191_scale0262_ramp05`,
`clearance_startup_tail_stack_s193`,
`clearance_actionscale_preroll_stack_s195_scale0262_preroll25`,
`clearance_actionscale_preroll_stack_s197_scale0263_preroll25`,
`clearance_actionscale_stack_s199_scale026205`,
`clearance_s177_tail_stack_s201`, and
`clearance_s177_tail_stack_s203_scale026215`,
`clearance_cmdmlp_lowtail_s205`, and
`clearance_s201_microreflex_s207`.

Current best retained candidate: `clearance_liftscale_stack_s143_step090_offset005`.
It is still blocked by the G10 low-clearance-ratio gate in all three scenarios;
p50 clearance and no-fall behavior are no longer the bottleneck. The remaining
failure is lower-tail/startup clearance, especially in turning. Next M10 work
should use a more substantial training redesign or a higher-clearance teacher to
drive low-clearance ratio below `0.25` while preserving strong movement.

clearance evidence commands:

```bash
./scripts/analyze_clearance_readiness.sh \
  --profile-name context_stage1_three_scenario_10ep_e80 \
  --output-dir artifacts/clearance_readiness/context_stage1_three_scenario_10ep_e80
./scripts/plan_policy_visual_inspection.sh \
  --profile-name context_stage1_three_scenario_10ep_e80 \
  --output-dir artifacts/policy_visual_inspection/context_stage1_three_scenario_10ep_e80
./scripts/build_clearance_evidence_package.sh \
  --profile-name context_stage1_three_scenario_10ep_e80 \
  --output-dir artifacts/clearance_evidence/context_stage1_three_scenario_10ep_e80
./scripts/compare_policy_teacher_suite.sh \
  artifacts/scenario_eval/open_duck_forward_m10_baseline_suite/suite_summary.json \
  artifacts/scenario_eval/context_stage1_three_scenario_10ep_e80_suite/suite_summary.json \
  --output-dir artifacts/policy_teacher_comparison/context_stage1_three_scenario_10ep_e80 \
  --strict
```

Teacher comparison is a relative behavior check only. It does not replace the
absolute `0.015 m` swing-clearance gate.

Current stored teacher comparison:

```text
status: TEACHER_COMPARISON_PASS
flat distance/speed ratio:       0.978
start-stop distance/speed ratio: 0.939
curve distance/speed ratio:      1.242
candidate fallen_count:          0
```

The candidate remains blocked because all three absolute clearance checks fail
and follow-camera visual review is still pending.

The official teacher also fails the same absolute clearance target. Its stored
scenario p50 clearances are approximately `0.0100`, `0.0079`, and `0.0056 m`.
Therefore, more BC collection from the unchanged teacher is not the primary
clearance solution. The residual/RL path now supports explicit swing-clearance
reward terms through `--swing-clearance-weight`,
`--low-clearance-penalty-weight`, and `--target-swing-clearance`.

The residual path now inherits the teacher policy input width, so both the
official 101D policy and the M10 104D context policy are supported. A live
zero-residual smoke with the context candidate completed 40/40 steps and
observed 9 swing samples with mean clearance about `0.00919 m`; all 9 were
below `0.015 m`, confirming that the reward sees the M10 failure.

Residual training supports optional per-command emphasis with
`--training-command VX,VY,YAW,WEIGHT`. Use this to focus the next run on
start/stop and turning clearance while still retaining a lower-weight flat-walk
command.

First bounded clearance experiment:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --no-viewer

./scripts/train_residual_policy.sh \
  context_stage1_three_scenario_10ep_e80 \
  --output-dir /data/rl_finetune/m10_clearance_residual_smoke \
  --profile-name m10_clearance_residual_smoke \
  --iterations 2 \
  --population 8 \
  --steps-per-episode 150 \
  --residual-scale 0.03 \
  --swing-clearance-weight 0.75 \
  --low-clearance-penalty-weight 1.0 \
  --target-swing-clearance 0.015 \
  --force-profile
```

This remains an experiment until the generated profile passes the original
three-scenario suite, absolute clearance readiness, and teacher comparison.

Constant-residual experiment result:

```text
profile: m10_clearance_residual_cem2x8_s31
search: 2 iterations x 8 candidates x 100 steps
residual scale: 0.03
result: REJECTED

baseline -> residual swing-clearance p50:
flat:       0.01023m -> 0.00943m
start-stop: 0.00759m -> 0.00759m
curve:      0.00632m -> 0.00599m

falls: 0
clearance gate: 0/3 PASS
```

Conclusion: a constant 14D residual bias cannot target the swing phase without
also perturbing stance behavior. Do not spend more search budget on this policy
class. That follow-up has already been implemented: phase/contact,
command/state, and warm-started nonlinear residual candidates were trained and
evaluated. The strongest retained experimental candidate is
`m10_command_state_mlp_cem4x14_s79`; it improves distance, stuck ratio, and
clearance relative to the context candidate, but still fails the absolute G10
`0.015 m` clearance threshold. The rejected constant-bias model, metrics, and
rollout evidence remain under `/data/rl_finetune/` and `artifacts/`; its
generated runtime profile was removed.

## Read First

```text
README.md
docs/README.md
docs/PROJECT_SOP.md
docs/PATCH_DELIVERY_AND_VALIDATION.md
docs/SORIDORMI_TARGET_AND_ROADMAP.md
docs/SORIDORMI_EXECUTION_ROADMAP.md
docs/SORIDORMI_NEURAL_PARAMETER_ADAPTIVE_MPC_WBC.md
docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md
docs/SORIDORMI_BC_TRAINING_CONTRACT.md
docs/SORIDORMI_DATA_PIPELINE_M9.md
docs/SORIDORMI_SCENARIO_CURRICULUM.md
```

## Simulator Ownership

External-sim tools need a separately running MuJoCo server:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Visual inspection:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

`collect_random_teacher_dataset.sh` is different: it owns its temporary MuJoCo
server. Do not start a second sim server for it. Pass `--viewer` and usually
`--follow-camera` to the collector itself when needed.

## Current M9 Pipeline

1. Collect scenario-aware teacher rows:

```bash
./scripts/collect_random_teacher_dataset.sh \
  --backend mujoco \
  --scenario flat_walk_varied_speed_v1 \
  --profile open_duck_forward \
  --episodes 10 \
  --steps-per-episode 300 \
  --command-ramp-steps 20 \
  --reset-attempts 10 \
  --reset-retry-sleep 0.5 \
  --seed 7 \
  --output /data/training_datasets/flat_walk_varied_speed_v1_10ep.jsonl \
  --json
```

2. Gate/report/export/prepare:

```bash
./scripts/report_dataset_coverage.sh RAW.jsonl --json
./scripts/gate_dataset_scenario_coverage.sh RAW.jsonl --require-scenario flat_walk_varied_speed_v1 --json
./scripts/export_context_bc_dataset.sh RAW.jsonl --output CONTEXT.jsonl --json
./scripts/validate_bc_training_contract.sh --sample-jsonl CONTEXT.jsonl --json
./scripts/prepare_context_bc_dataset.sh CONTEXT.jsonl --output-dir PREPARED_DIR --json
./scripts/gate_context_bc_prepared_dataset.sh PREPARED_DIR/prepared_manifest.json --require-scenario flat_walk_varied_speed_v1 --json
```

3. Train offline BC smoke models:

```bash
./scripts/train_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_stage1_command --json
./scripts/train_neural_behavior_clone.sh PREPARED_DIR/prepared_manifest.json --input-mode context_stage1_command --skip-onnx --no-profile --json
```

## Validation Policy

Preferred validation:

```bash
pytest -q
python -m compileall -q src
```

For JSON-producing shell wrappers, stdout must stay machine-readable when
`--json`; Docker/Compose/CUDA status belongs on stderr.

## Boundaries

- MuJoCo before hardware.
- Chromie is brain; Soridormi is cerebellum/body runtime.
- Chromie calls structured skills/context, never raw joint actions or low-level
  `action_14d` policy outputs.
- Hardware commands require explicit user intent; otherwise dry-run/read-only.
- BC copies the teacher distribution. Do not claim it improves stride,
  clearance, obstacles, or recovery beyond teacher behavior without rollout
  evidence.
- Generated reports belong under `artifacts/` and should not be committed.
