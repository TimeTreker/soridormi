# Soridormi execution roadmap and milestone plan

> Principle: **Every milestone must produce verifiable artifacts and pass an
> explicit acceptance gate.**
>
> Work remains MuJoCo-first. A policy that trains successfully or has low
> offline error is not accepted until it demonstrates the required closed-loop
> behavior.

This document defines execution order, milestone gates, major risks, and the
immediate work queue. See `docs/SORIDORMI_TARGET_AND_ROADMAP.md` for the system
target, brain/cerebellum boundary, and detailed evidence for current policy
candidates.

## Overview

```text
M0-M3 Runtime foundations
      |
      v
M4 Official parity
      |
      +--> M5 Replaceable policy interface
      |       |
      |       v
      +--> M6 Training/evaluation backbone
              |
              v
M7 Structured body skills --> M8 Scenarios and social behavior
              |
              v
M9 Context BC data pipeline --> M10 Runtime context policy
                                      |
                                      v
                               M11 Generalization
                                      |
                                      v
                               M12 Targeted BC/RL improvement
                                      |
                                      v
                               M13 Hardware bridge
                                      |
                                      v
                               M14 Chromie integration
                                      |
                                      v
                               M15 Navigation goal pipeline
```

Current position:

```text
M4-M9: substantially complete
M10: functional Stage 1 candidate available
Current blocker: low swing-foot clearance and limited held-out generalization
M11A task-agent contract foundation: no-motion gate available
M11B task event cursor: monitorable event stream available
M11C retry-safe task refs: client idempotency available
M11D task timeout expiry: planning-hold timeout available
Hardware: intentionally not started
```

---

## M0-M3 - Runtime foundations [Completed]

**Goal:** Establish stable robot, API, simulator, observation, action, and
controller contracts.

- [x] Define `RobotState` and `MotorCommand`
- [x] Load Open Duck Mini v2 in MuJoCo
- [x] Implement simulator API and runtime loop
- [x] Define the 101D official-policy observation
- [x] Map 14D policy actions to actuator commands
- [x] Add reset, fixed-base, logging, and debugging modes
- [x] Preserve backend separation for future hardware

**Gate G3:** The same runtime interfaces can operate against MuJoCo and support
a future hardware backend without changing policy semantics.

---

## M4 - Official Open Duck policy parity [Completed]

**Goal:** Reproduce the official Open Duck locomotion policy through Soridormi.

- [x] Port official ONNX inference behavior
- [x] Match observation construction
- [x] Match phase and command handling
- [x] Match action-to-motor mapping
- [x] Add official baseline, replay, and trace comparison
- [x] Add first-divergence analysis
- [x] Validate synchronous policy stepping

**Gate G4:**

- Official reference files are required and checked.
- Observation, action, loop ordering, and reset behavior show no unexplained
  meaningful divergence.
- Official baseline and comparison tools remain available.

> Official parity is the permanent trusted baseline. Later models may replace
> the policy, but not the parity evidence.

---

## M5 - Replaceable policy interface [Completed]

**Goal:** Replace compatible models without rewriting runtime code.

- [x] Define policy profiles
- [x] Validate input/output shapes
- [x] Validate ONNX providers
- [x] Export policy contracts and manifests
- [x] Package, install, restore, and promote candidates
- [x] Add profile acceptance checks
- [x] Preserve the official model as a fallback

**Gate G5:** A compatible policy package can be validated and loaded through
the existing runtime using only its declared profile and contract.

---

## M6 - Training and evaluation backbone [Completed]

**Goal:** Build a reproducible policy-improvement loop.

- [x] Export policy-supervision datasets
- [x] Validate observations and actions
- [x] Produce deterministic train/validation/test splits
- [x] Train linear BC baselines
- [x] Train neural BC candidates
- [x] Evaluate candidates offline
- [x] Evaluate candidates in MuJoCo
- [x] Compare candidates with the official teacher
- [x] Support relabeling and iterative retraining

**Gate G6:**

```text
dataset valid
training reproducible
model/profile contract valid
bounded MuJoCo rollout completed
candidate compared with teacher
```

> Offline action error alone is never a promotion criterion.

---

## M7 - Structured body-skill interface [Completed]

**Goal:** Expose bounded body capabilities to higher-level systems.

- [x] Define named skills
- [x] Validate skill arguments and availability
- [x] Translate locomotion skills into structured commands
- [x] Reject unsupported or unsafe requests
- [x] Prevent planners from sending raw joint actions
- [x] Export a capability manifest

**Gate G7:** Every externally callable body action is represented by a bounded
skill or structured context request.

---

## M8 - Scenario and interaction layer [Completed]

**Goal:** Test body behavior in meaningful scenarios rather than isolated
commands.

- [x] Define a scenario registry and curriculum
- [x] Add single-scenario rollout evaluation
- [x] Add batch scenario-suite evaluation
- [x] Add scenario-specific acceptance thresholds
- [x] Implement bounded head and social skills
- [x] Add social-skill readiness and acceptance reports
- [x] Add structured look-target input

**Gate G8:** A scenario produces reproducible evidence containing commands,
outcomes, failures, and acceptance results.

---

## M9 - Context BC data pipeline [Current: core complete]

**Goal:** Produce auditable, scenario-aware training datasets.

Policy stage:

```text
observation[101]
+ desired_command(vx_mps, vy_mps, yaw_radps)
-> action_14d
```

- [x] Collect scenario-aware teacher rows
- [x] Report dataset coverage
- [x] Gate required scenarios and command ranges
- [x] Export context BC rows
- [x] Validate the BC training contract
- [x] Split by rollout group to prevent leakage
- [x] Gate prepared train/validation/test datasets
- [x] Reject empty or invalid upstream outputs
- [x] Add a training-ready manifest and report command
- [x] Generate and retain a training-ready report for the current
  three-scenario dataset

**Gate G9:**

- All required scenarios meet coverage thresholds.
- Train, validation, and test splits are non-empty.
- No rollout group appears in multiple splits.
- Contracts and file hashes are recorded.
- Both dataset gates pass.
- A training-ready manifest is generated before training.

> **Retained local evidence:** Regenerated on 2026-06-22 under
> `data/training_datasets/` and `artifacts/`, which are intentionally ignored
> runtime-output directories. The final report is
> `artifacts/training/context_bc/training_ready/context_stage1_three_scenario_10ep/training_ready_report.md`.
> It records `9000` total samples, balanced `7200/900/900` splits,
> `3000` samples each for `flat_walk_varied_speed_v1`,
> `start_stop_velocity_ramp_v1`, and `curve_turn_walk_v1`, passing scenario and
> prepared gates, no rollout-group leakage, and file hashes for the prepared
> manifest, gates, contract, and splits.

---

## M10 - Runtime context policy [Current: clearance blocker identified]

**Goal:** Run context-conditioned policies through the production runtime path.

- [x] Define `context_stage1_command`
- [x] Build 104D policy inputs
- [x] Require `[1, 104]` profile/model contracts
- [x] Export runnable context-policy ONNX profiles
- [x] Train flat-only candidates
- [x] Diagnose low-speed and curve failures
- [x] Train a three-scenario candidate
- [x] Pass the flat/start-stop/curve suite
- [x] Diagnose low swing-foot clearance ✗ FAILS
  - flat_walk_varied_speed_v1: 0.0102m (need +0.0048m)
  - start_stop_velocity_ramp_v1: 0.0076m (need +0.0074m)
  - curve_turn_walk_v1: 0.0063m (need +0.0087m)
- [ ] Perform human follow-camera visual inspection before promotion
- [x] Define clearance-focused promotion thresholds
- [x] Add threshold-aligned clearance readiness report
- [x] Add an clearance evidence package and visual-review template
- [x] Add scenario-suite comparison against the official teacher
- [x] Re-evaluate the current candidate against the official teacher
  - flat distance/speed ratio: 0.978
  - start-stop distance/speed ratio: 0.939
  - curve distance/speed ratio: 1.242
  - no falls; stuck-ratio regression within 0.10
- [x] Test clearance-aware constant residual bias
  - result: rejected; clearance regressed or remained unchanged
  - flat: 0.01023m -> 0.00943m
  - start-stop: 0.00759m -> 0.00759m
  - curve: 0.00632m -> 0.00599m
- [x] Implement a bounded phase/state-conditioned residual policy
  - actor inputs: bias + left/right foot contact + cosine/sine gait phase
  - bounded by `tanh` plus the existing residual scale and clipping envelope
  - supports repeated `--training-command VX,VY,YAW` conditions
  - supports optional command weights as `--training-command VX,VY,YAW,WEIGHT`
    for start/stop and turning emphasis
  - 104D ONNX contract and 100-step MuJoCo deployment smoke: PASS
- [x] Train and evaluate a phase/contact candidate across the three M10 scenarios
  - candidate: `m10_phase_contact_clearance_cem3x8_s53`
  - no falls; distance and stuck ratio improved across all three scenarios
  - clearance improved over the context candidate:
    - flat: 0.01023m -> 0.01134m
    - start-stop: 0.00759m -> 0.00973m
    - curve: 0.00632m -> 0.00724m
  - result: useful experimental improvement, but rejected for promotion
  - all scenarios remain below 0.015m and low-clearance ratio remains 1.0
- [x] Add gate-aligned episodic clearance scoring
  - rewards episode median clearance relative to the target
  - penalizes the fraction of swing samples below the target
- [x] Add compact command/state/history residual actor
  - desired velocity + contacts + phase + sagittal joint offsets/history
  - bounded output to six sagittal leg joints only
- [x] Train and evaluate gate-aligned command/state candidate
  - candidate: `m10_command_state_gate_cem4x12_s67`
  - flat: 0.01023m -> 0.01314m
  - start-stop: 0.00759m -> 0.01080m
  - curve: 0.00632m -> 0.00759m
  - total distance: 0.733m -> 1.070m
  - no falls; maximum stuck ratio improved to 0.084
  - result: best candidate so far, but still blocked by G10
- [x] Add warm-started nonlinear command/state residual actor
  - four hidden units plus a linear skip path
  - warm-start preserves the best linear candidate exactly
  - same bounded six-joint sagittal output contract
- [x] Train and evaluate nonlinear residual candidate
  - candidate: `m10_command_state_mlp_cem4x14_s79`
  - flat: 0.01023m -> 0.01471m
  - start-stop: 0.00759m -> 0.01152m
  - curve: 0.00632m -> 0.01025m
  - total distance: 0.733m -> 1.275m
  - no falls; maximum stuck ratio: 0.0365
  - result: strongest candidate, but all scenarios remain below G10
- [x] Regenerate local `context_stage1_three_scenario_10ep_e80` ONNX from the
  retained M9 dataset
  - model/profile contract: PASS
  - offline MAE: train 0.01088, val 0.01179, test 0.01167
  - required scenario suite: FAIL, 0/3 scenarios accepted
  - no falls; all three scenarios fail clearance
  - flat: distance 0.31755m, p50 clearance 0.01006m, stuck 0.016
  - start-stop: distance 0.32191m, p50 clearance 0.00855m, stuck 0.012
  - curve: distance 0.15236m, p50 clearance 0.00649m, stuck 0.277
  - result: runnable but not equivalent to the historical retained candidate
- [x] Restore the historical `context_stage1_three_scenario_10ep_e80` ONNX from
  local backup and preserve the regenerated ONNX separately
  - restored ONNX sha256:
    `2a7e41afe855702638aed56ec32e0f5e067a6b76fdcd76af4d43a101191730b7`
  - regenerated ONNX preserved under
    `data/training_runs/context_stage1_three_scenario_10ep_neural_bc_m10_e80_regenerated_20260622/`
  - restored live suite under the current clearance gate: FAIL, 0/3
  - flat: distance 0.31202m, p50 clearance 0.01023m
  - start-stop: distance 0.26651m, p50 clearance 0.00759m
  - curve: distance 0.15465m, p50 clearance 0.00632m
- [x] Rebuild clearance evidence and keep the follow-camera review path ready
  - command for human visual pass: `./scripts/run_sim_server.sh --backend mujoco --profile context_stage1_three_scenario_10ep_e80 --viewer --follow-camera`
  - metric-grounded review artifact records no falls, but all three scenarios
    failed clearance
  - filled review:
    `artifacts/clearance_evidence/context_stage1_three_scenario_10ep_e80/visual_review.json`
  - evidence package status: `BLOCKED_BY_CLEARANCE_READINESS`
- [x] Run a bounded fresh clearance-residual probe without the missing
  historical warm-start checkpoint
  - output: `/data/rl_finetune/clearance_gap_probe_s91`
  - best score: `0.26074`
  - result: zero residual selected; not a promotion candidate
- [x] Restore the strongest retained nonlinear residual checkpoint
  - output: `/data/rl_finetune/m10_command_state_mlp_cem4x14_s79`
  - suite remains blocked by current clearance gate, but it is still the best
    retained residual reference
  - flat: p50 clearance 0.01471m, low-clearance ratio 0.528
  - start-stop: p50 clearance 0.01152m, low-clearance ratio 0.971
  - curve: p50 clearance 0.01025m, low-clearance ratio 0.973
- [x] Train and evaluate a restored warm-start clearance sequence candidate
  - candidate: `clearance_gap_sequence_restored_s83`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.47062m, p50 clearance 0.01344m, low-clearance ratio 0.752
  - start-stop: distance 0.50116m, p50 clearance 0.01148m, low-clearance ratio 0.985
  - curve: distance 0.23559m, p50 clearance 0.00863m, low-clearance ratio 1.000
  - result: reproducible failed candidate; do not promote
  - follow-up finding: wrapper used `residual_scale 0.05`, so the `s79`
    warm-start behavior was not actually preserved
- [x] Fix the clearance wrapper to preserve retained warm-start scale
  - default `--residual-scale`: `0.1`, matching
    `m10_command_state_mlp_cem4x14_s79`
  - `--residual-scale` is now an explicit wrapper option
- [x] Train and evaluate a scale-preserved sequence candidate
  - candidate: `clearance_gap_sequence_scale_preserved_s89`
  - initial checkpoint: `m10_command_state_mlp_cem4x14_s79`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.58371m, p50 clearance 0.01532m, low-clearance ratio 0.445
  - start-stop: distance 0.69515m, p50 clearance 0.01481m, low-clearance ratio 0.514
  - curve: distance 0.39385m, p50 clearance 0.01192m, low-clearance ratio 0.992
  - result: strong movement and flat p50 improvement; curve low-clearance ratio
    remained saturated
- [x] Train and evaluate a low-clearance-ratio continuation
  - candidate: `clearance_lowratio_sequence_s97`
  - initial checkpoint: `clearance_gap_sequence_scale_preserved_s89`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.59257m, p50 clearance 0.01560m, low-clearance ratio 0.437
  - start-stop: distance 0.68191m, p50 clearance 0.01466m, low-clearance ratio 0.520
  - curve: distance 0.45424m, p50 clearance 0.01341m, low-clearance ratio 0.754
  - result: best retained candidate so far, but still blocked by low-clearance
    ratio and start/stop plus curve p50 clearance
- [x] Train and evaluate a tighter low-ratio continuation
  - rejected probe: `clearance_lowratio_turnfocus_s101` re-exported the same
    ONNX as `clearance_lowratio_sequence_s97`
  - candidate: `clearance_lowratio_refine_s103`
  - initial checkpoint: `clearance_lowratio_sequence_s97`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.62978m, p50 clearance 0.01634m, low-clearance ratio 0.438
  - start-stop: distance 0.74535m, p50 clearance 0.01559m, low-clearance ratio 0.456
  - curve: distance 0.46317m, p50 clearance 0.01451m, low-clearance ratio 0.662
  - result: best retained candidate so far, but still blocked by low-clearance
    ratio in all scenarios and curve p50 clearance
- [x] Train and evaluate a multi-command low-ratio continuation
  - candidate: `clearance_lowratio_multicmd_s107`
  - initial checkpoint: `clearance_lowratio_refine_s103`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.63908m, p50 clearance 0.01614m, low-clearance ratio 0.420
  - start-stop: distance 0.78692m, p50 clearance 0.01578m, low-clearance ratio 0.425
  - curve: distance 0.47653m, p50 clearance 0.01456m, low-clearance ratio 0.519
  - rejected probe: `clearance_lowratio_curvepush_s109` slightly improved curve
    p50 but worsened total distance and max low-clearance ratio
  - result: best retained candidate so far, but still blocked by low-clearance
    ratio in all scenarios and curve p50 clearance
- [x] Train and evaluate a gate-push low-ratio continuation
  - candidate: `clearance_lowratio_gatepush_s111`
  - initial checkpoint: `clearance_lowratio_multicmd_s107`
  - model/profile contract: PASS
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.65596m, p50 clearance 0.01679m, low-clearance ratio 0.388
  - start-stop: distance 0.78588m, p50 clearance 0.01626m, low-clearance ratio 0.423
  - curve: distance 0.48593m, p50 clearance 0.01506m, low-clearance ratio 0.496
  - rejected probe: `clearance_lowratio_targetlift_s113` exported the same
    ONNX as `clearance_lowratio_gatepush_s111`
  - rejected probe: `clearance_lowratio_forced_s115` disabled zero-candidate
    retention and produced a different ONNX, but regressed training clearance
    metrics
  - rejected probe: `clearance_lowratio_suitecmd_s117` directly optimized the
    three suite commands and reduced curve low-clearance ratio slightly, but
    regressed flat/start low-clearance ratio and total distance
  - result: best retained candidate so far, but still blocked by low-clearance
    ratio in all scenarios
- [x] Add stacked residual-teacher support and compact swing-lift actors
  - `residual_onnx` teachers now compose through `make_runtime_policy`, so a
    residual profile can safely train on top of another residual profile.
  - added `contact_phase_lift`, `command_contact_phase_lift`, and
    `contact_phase_harmonic_lift` residual actor families for bounded swing-leg
    clearance refinement.
- [x] Train and evaluate stacked contact/phase lift candidates
  - `clearance_contactlift_stack_s121`: blocked, but improved total distance to
    `~2.028 m`, kept all p50 clearances above `0.015 m`, and reduced max
    low-clearance ratio to `~0.460`.
  - `clearance_contactlift_stack_s123`: blocked; improved flat/start ratios and
    total distance, but regressed curve to `~0.480`; do not promote.
  - `clearance_cmdlift_stack_s125`: command-conditioned training probe; did not
    beat `s121`/`s127` on curve; do not promote.
- [x] Train and evaluate explicit residual-scale lift candidates
  - `clearance_liftscale_stack_s127`: previously best retained blocked candidate.
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.72664m, p50 clearance 0.01742m, low-clearance ratio 0.353
  - start-stop: distance 0.88341m, p50 clearance 0.01716m, low-clearance ratio
    0.345
  - curve: distance 0.57878m, p50 clearance 0.01616m, low-clearance ratio 0.409
  - total distance: 2.18883m; no falls
  - `clearance_liftscale_stack_s129`: blocked; improved flat ratio to `~0.318`
    but regressed start-stop and curve versus `s127`; do not promote.
- [x] Add and smoke-test a harmonic swing-lift actor
  - `clearance_harmonic_stack_s131`: blocked training probe; did not beat `s127`
    in training lower-tail metrics, so it did not receive full G10 evidence.
  - result: keep `s127` as the best retained blocked candidate before phase
    timing probes.
- [x] Run phase-timing and continuation probes from `s127`
  - `clearance_liftscale_stack_s137_step090`: phase-rate probe; reduced
    low-clearance ratios to flat `~0.298`, start-stop `~0.264`, curve `~0.346`,
    but remained 0/3 under G10.
  - `clearance_liftscale_stack_s143_step090_offset005`: current best retained
    blocked candidate after adding `phase.offset=0.05`.
  - required scenario suite: FAIL, 0/3 under current clearance gate
  - flat: distance 0.70473m, p50 clearance 0.01858m, low-clearance ratio 0.268
  - start-stop: distance 0.84432m, p50 clearance 0.01861m, low-clearance ratio
    0.257
  - curve: distance 0.60072m, p50 clearance 0.01781m, low-clearance ratio 0.308
  - total distance: 2.14977m; no falls
  - rejected probes: `clearance_cmdtail_stack_s133`,
    `clearance_liftscale_stack_s135_scale018`,
    `clearance_liftscale_stack_s139_step080`,
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
    `clearance_harmonic_aggressive_stack_s173`.
- [x] Run post-`s143` low-ratio and startup-tail probes
  - `clearance_actionscale_stack_s177_scale0262`: full-suite 0/3; curve nearly
    passed at low-clearance ratio `~0.263`, but flat/start-stop regressed to
    `~0.319`/`~0.315`, so it is not the retained best.
  - `clearance_actionscale_stack_s175_scale026`,
    `clearance_actionscale_stack_s181_scale0248`,
    `clearance_actionscale_stack_s187_scale02618`,
    `clearance_actionscale_stack_s199_scale026205`,
    `clearance_actionscale_preroll_stack_s195_scale0262_preroll25`, and
    `clearance_actionscale_preroll_stack_s197_scale0263_preroll25`: profile
    brackets remained blocked by the curve low-clearance-ratio gate.
  - `clearance_actionscale_ramp_stack_s191_scale0262_ramp05`: confirmed
    profile-level command ramp can now reach the controller, but the rollout
    collapsed in curve (`low ratio 1.0`, distance `~0.019 m`).
  - `clearance_reflex_stack_s183_swinglift`,
    `clearance_reflex_stack_s185_earlysoft`, and
    `clearance_reflex_stack_s189_swinggain`: opt-in clearance-reflex runtime
    probes were metric-grounded rejects; do not promote.
  - `clearance_startup_tail_stack_s193`: learned startup-tail continuation
    trained cleanly but retained poor startup lower-tail metrics.
  - `clearance_s177_tail_stack_s201`: trained from the `s177` action-scale near
    miss; live curve still failed at low-clearance ratio `~0.257`, with p50
    `~0.01846 m`, distance `~0.717 m`, and no fall.
  - `clearance_s177_tail_stack_s203_scale026215`: tiny action-scale nudge on
    `s201`; curve regressed to low-clearance ratio `~0.294`.
  - `clearance_cmdmlp_lowtail_s205`: broader `command_state_mlp` lower-tail
    run with a stricter `0.017 m` training target; live curve regressed to
    low-clearance ratio `~0.338`, p50 `~0.01712 m`, distance `~0.606 m`, and no
    fall.
  - `clearance_s201_microreflex_s207`: tiny swing-clearance reflex on the
    `s201` near miss; live curve regressed to low-clearance ratio `~0.282`,
    p50 `~0.01882 m`, distance `~0.694 m`, and no fall.
  - `clearance_s143_cmdtail_stack_s211`: direct `s143` command-contact/phase
    lift continuation with aggressive lower-tail penalties; live full suite
    regressed max low-clearance ratio to `~0.391`.
  - `clearance_s143_scenariogate_stack_s213`: scenario-shaped direct `s143`
    continuation; start-stop passed (`~0.249` low-clearance ratio), but flat
    regressed to `~0.295` and curve regressed to `~0.318`, so it is not retained.
  - `clearance_s143_refguard_stack_s215`: direct `s143` continuation with
    per-objective reference low-clearance-ratio penalties set to the retained
    `s143` scenario ratios; full suite remained 0/3 and regressed all three
    low-clearance ratios (`flat ~0.295`, `start-stop ~0.271`,
    `curve ~0.327`).
  - `clearance_s143_gateguard_stack_s217`: same direct `s143` continuation, but
    with all per-objective low-clearance references set to the G10 `0.25` gate;
    start-stop passed at `~0.245`, but flat `~0.271` and curve `~0.325`
    failed and regressed against `s143`.
  - `clearance_s143_curvegateguard_stack_s219`: corrected the curve training
    objective to constant yaw (`0.09,0,0.12`) matching the suite command;
    start-stop passed at `~0.241`, but flat `~0.275` and curve `~0.340`
    failed and regressed against `s143`.
  - result: keep `clearance_liftscale_stack_s143_step090_offset005` as the best
    balanced retained full-suite reference. The remaining blocker is lower-tail
    startup/turning clearance, not p50 clearance or falls.
- [ ] Focus the next training stage on a broader clearance redesign, or
  acquire a higher-clearance teacher; do not continue narrow scalar/reflex/guard
  penalty retunes as the primary M10 path
- [ ] **DECISION REQUIRED:** Clearance refinement or experimental M10.0?

**Gate G10:**

```text
profile/model contract: PASS
bounded rollout: PASS
required scenario suite: FAIL under current clearance gate
fall/reset limits: PASS
foot-clearance threshold: ✗ FAIL (low-clearance ratio remains too high)
human visual inspection: PENDING
metric-grounded review: ✗ FAIL
teacher comparison: PASS (relative behavior only; does not replace clearance)
```

> **Current status:** Historical retained evidence for
> `context_stage1_three_scenario_10ep_e80` passed the three-scenario suite but
> failed G10 due to swing clearance deficit. The historical ONNX was restored
> locally on 2026-06-22 and the regenerated ONNX was preserved separately. Under
> the current clearance-aware gate, restored E80 still fails all three scenario
> acceptance gates because p50 swing clearance is below `0.015 m` and
> low-clearance ratio remains too high.
> The clearance evidence package includes a filled metric-grounded review. It is
> intentionally blocked, not a human visual PASS: all scenarios remained upright
> but failed the `0.015 m` swing-clearance gate. A direct human follow-camera
> inspection remains pending before any promotion.
> The current best retained residual candidate is
> `clearance_liftscale_stack_s143_step090_offset005`: it keeps all three
> scenario p50 clearances over the `0.015 m` threshold, has no falls, and
> reduces max low-clearance ratio to `0.308`, but it remains blocked because
> flat (`0.268`), start-stop (`0.257`), and curve (`0.308`) still exceed the
> `0.25` low-clearance-ratio limit.
> Post-`s143` action-scale, pre-roll, command-ramp, reflex, and startup-tail
> probes through `clearance_s201_microreflex_s207` did not pass G10. The
> closest live curve-only probe was `clearance_s177_tail_stack_s201` (`~0.257`
> low-clearance ratio), but it still failed the curve gate and did not justify a
> new retained promotion.
>
> **Clearance readiness:** Current best generated with
> `./scripts/analyze_clearance_readiness.sh --profile-name clearance_liftscale_stack_s143_step090_offset005 --suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 --output-dir artifacts/clearance_readiness/clearance_liftscale_stack_s143_step090_offset005 --json`.
>
> **Decision path:**
> - **Option 1 (Recommended):** Continue clearance-focused refinement from the
>   best retained residual reference before M10.0 release
> - **Option 2:** Accept as experimental M10.0, document limitation, refine in M11

---

## M11A - Task-agent contract foundation [Current: gate ready]

**Goal:** Make the Chromie-to-Soridormi embodied task boundary executable as a
validated contract before adding real navigation, perception, manipulation, or
hardware execution.

- [x] Add task-level MCP tools:
  `soridormi.task.get_capabilities`, `preview`, `submit`, `status`, `events`,
  and `cancel`
- [x] Keep the task API no-motion while the body task executor is still
  contract-first
- [x] Store Soridormi-owned task readiness in
  `configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`
- [x] Add structured plan steps, blocked subsystems, and
  `recommended_next_actions`
- [x] Add a derived Soridormi `task_graph` with stable node IDs and
  `raw_control_allowed=false`
- [x] Add no-motion acceptance cases for successful skill dry-runs,
  fail-closed navigation/manipulation/perception paths, stop redirection, and
  unsafe physical requests
- [x] Add navigation-goal training and contract cases without claiming that
  navigation is executable
- [x] Add the host validation gate:
  `./scripts/validate_task_agent_contract.sh`

**Gate G11A:**

```text
./scripts/validate_task_agent_contract.sh
```

This gate passes when:

- the MCP task capability manifest exports successfully;
- the task capability readiness table validates;
- task preview/submit/status/events/cancel tests pass;
- acceptance cases preserve `no_motion=true`;
- unsupported or unsafe embodied tasks fail closed;
- task responses expose `task_graph` without low-level control fields;
- training cases and navigation-goal contracts remain schema-valid;
- task-agent docs contain the required contract references and balanced
  Markdown fences.

> **Status:** G11A proves the Chromie-facing embodied task contract, not robot
> autonomy. It does not claim that Soridormi can route to a house, fetch water,
> manipulate objects, or execute task-level goals physically in MuJoCo or on
> hardware. Those remain later milestones behind explicit simulator gates.

---

## M11B - Task event cursor and monitoring contract [Current: gate ready]

**Goal:** Give Chromie a stable polling loop for Soridormi-owned embodied tasks
before task-level physical execution exists.

- [x] Add a versioned `soridormi.task_events.v1` response from
  `soridormi.task.events`
- [x] Report `status`, `phase`, `terminal`, and `safe_idle` directly in the
  event response
- [x] Preserve the cursor contract with `after_sequence`,
  `next_after_sequence`, and `latest_sequence`
- [x] Add `returned_count` and `has_more` for future pagination compatibility
- [x] Add `poll_recommendation` so Chromie can continue polling, cancel, or
  stop polling after terminal states
- [x] Reject invalid negative cursors
- [x] Declare the monitoring cursor fields in the MCP capability manifest

**Gate G11B:**

```text
./scripts/validate_task_agent_contract.sh
```

This gate passes when task event cursor behavior works for active planning-hold
tasks, terminal dry-run tasks, and terminal safe-idle tasks, and the manifest
exports the monitoring fields Chromie needs.

> **Status:** G11B improves observability only. It does not add real-time
> asynchronous execution, physical task execution, or navigation autonomy.

---

## M11C - Retry-safe task identity [Current: gate ready]

**Goal:** Let Chromie safely retry task submission and monitor/cancel by its own
task reference without creating duplicate Soridormi records.

- [x] Add optional `client_task_ref` to task preview/submit payloads
- [x] Index submitted tasks by `client_task_ref`
- [x] Return the original Soridormi `task_id` when the same
  `client_task_ref` and payload are submitted again
- [x] Mark duplicate retries with `idempotent_replay=true`
- [x] Reject reuse of the same `client_task_ref` with a different payload
- [x] Allow `task.status`, `task.events`, and `task.cancel` lookup by either
  `task_id` or `client_task_ref`

**Gate G11C:**

```text
./scripts/validate_task_agent_contract.sh
```

This gate passes when duplicate submits do not duplicate records, conflicting
payloads fail closed, and the MCP manifest exposes the retry identity fields.

---

## M11D - Task timeout expiry [Current: gate ready]

**Goal:** Prevent no-motion planning-hold tasks from remaining active forever
after Chromie has stopped waiting.

- [x] Add `deadline_at`, `expired`, and `timeout_elapsed_s` to task status
  payloads
- [x] Expire non-terminal tasks during `task.status`, `task.events`, or
  `task.cancel` reads after `timeout_s`
- [x] Transition expired tasks to terminal `failed`
- [x] Emit a `task_timed_out` lifecycle event
- [x] Return timeout-specific recommended actions, including an emergency-stop
  hint when the task requested `emergency_stop_on_timeout`
- [x] Keep already-terminal dry-run and refused tasks unchanged

**Gate G11D:**

```text
./scripts/validate_task_agent_contract.sh
```

This gate passes when planning-hold tasks expire deterministically, timeout
events are cursor-visible, and terminal timeout status does not emit duplicate
timeout events on repeated reads.

> **Status:** G11C/G11D improve reliability and lifecycle hygiene only. They do
> not add asynchronous execution, real navigation, manipulation, or physical
> task motion.

---

## M11 - Broader locomotion generalization

**Goal:** Generalize beyond the initial three training scenarios.

- [ ] Add lateral walking
- [ ] Expand turn and curve ranges
- [ ] Add randomized start/stop transitions
- [ ] Add rough-ground scenarios
- [ ] Add slopes
- [ ] Add low obstacles
- [ ] Add command-transition stress tests
- [ ] Create held-out randomized evaluation suites
- [ ] Add bounded task context
- [ ] Add bounded environment context
- [ ] Evaluate short-history input only when evidence supports it

**Gate G11:**

- Training and held-out scenarios are explicitly separated.
- Required held-out suites pass across multiple fixed seeds.
- No major regression occurs on the original three-scenario suite.
- The candidate remains within fall, drift, yaw, and clearance limits.

---

## M12 - Targeted policy improvement

**Goal:** Improve measured weaknesses without hiding failures behind tuning.

Possible methods:

```text
targeted recollection
teacher relabeling
dataset balancing
DAgger-style recovery data
residual policy
RL fine-tuning
```

- [ ] Rank failure modes from M11 evidence
- [ ] Select one measurable weakness
- [ ] Establish a frozen baseline
- [ ] Collect targeted data
- [ ] Train a BC or residual candidate
- [ ] Compare with the frozen baseline
- [ ] Reject improvements that damage safety or generalization

**Gate G12:** The candidate improves its declared target metric and does not
regress required safety or held-out suites.

> RL is introduced only after the BC baseline and failure metric are clear.

---

## M13 - Hardware bridge

**Goal:** Transfer the validated runtime to the physical robot in controlled
stages.

```text
H0 read-only state
 -> H1 motor-command dry-run
 -> H2 safety/watchdog
 -> H3 low-power single joint
 -> H4 standing pose
 -> H5 tethered low-speed locomotion
```

- [ ] Implement hardware state streaming
- [ ] Verify joint ordering, signs, limits, and units
- [ ] Implement command dry-run logging
- [ ] Add emergency stop
- [ ] Add command timeout and watchdog
- [ ] Add voltage, thermal, and communication checks
- [ ] Perform low-power single-joint validation
- [ ] Perform tethered standing
- [ ] Perform tethered low-speed walking

**Gate G13:**

- Every earlier hardware phase passes before actuator authority increases.
- Commands remain inside validated limits.
- Loss of communication leads to a safe stop.
- Logs are retained for every actuator test.
- MuJoCo acceptance remains green.

> No actuator command is permitted during read-only or dry-run phases.

---

## M14 - Chromie integration [Current: contract surface complete]

**Goal:** Connect the Chromie brain to Soridormi through structured skills.

- [x] Publish Soridormi capability and availability data
- [x] Define versioned request/response schemas
- [x] Add execution status and failure reasons
- [x] Add cancellation and stop semantics
- [x] Validate named-skill integration in MuJoCo
- [x] Add bounded locomotion requests
- [x] Add bounded social and attention requests
- [x] Add safety refusal tests
- [x] Keep raw joint and `action_14d` interfaces inaccessible

**Gate G14:** Chromie can select, monitor, cancel, and recover from body skills
without producing low-level motor or policy actions.

> **Contract evidence:** Soridormi now exports task-level capability readiness,
> versioned preview/submit/status/events/cancel schemas, structured
> `plan_steps`, `blocked_subsystems`, `recommended_next_actions`, and a
> `raw_control_allowed=false` body-task graph. The M14 contract surface is
> validated by `./scripts/validate_task_agent_contract.sh`,
> `tests/test_mcp_capability_manifest.py`,
> `tests/test_task_acceptance_cases_m11.py`,
> `tests/test_mcp_local_tools.py`, and `tests/test_mcp_runtime_tools.py`.
> Broad natural-language routing into those declared task types is Chromie-side
> follow-up; missing navigation, approach, and manipulation goals remain
> structured refusals until M15 proves the required simulator pipelines.

---

## M15 - Navigation goal pipeline

**Goal:** Convert destination requests such as "walk forward to the house" into
safe, bounded local motion plans without feeding raw language or raw perception
into the low-level policy.

- [x] Declare `navigate_to_target` as a future, non-executable skill
- [x] Define the navigation goal contract and refusal conditions
- [ ] Add target-resolution adapters for structured place/object/person refs
- [ ] Add MuJoCo localization and route fixtures
- [ ] Add body-frame waypoint and short-route evaluators
- [ ] Add stop-before-obstacle and lost-target acceptance gates
- [ ] Promote `trajectory_follow` only after route tracking gates pass
- [ ] Promote `navigate_to_target` only after target resolution, routing, local
  obstacle checks, cancellation, timeout, and safe-idle evidence pass

**Gate G15:** Soridormi either refuses unresolved destination language before
motion, or executes a structured navigation goal through target resolution,
localization, routing, local planning, monitored execution, and safe-idle
verification in MuJoCo.

---

## Critical path

```text
M9 training-ready evidence
  -> clearance refinement
  -> M11 held-out generalization
  -> M12 targeted improvement
  -> M13 staged hardware transfer
  -> M14 physical Chromie integration
  -> M15 navigation goal pipeline
```

M14 API work may begin in MuJoCo before M13 finishes, but physical integration
depends on the hardware safety gate.

## Highest risks

| Risk | Impact | Mitigation |
|---|---|---|
| Low foot clearance | Trips and obstacle failures | Add clearance metrics, visual inspection, targeted data, and scenarios |
| Dataset leakage | Misleading validation results | Preserve rollout-group splits and enforce the prepared dataset gate |
| Narrow scenario coverage | Policy overfits nominal walking | Require held-out randomized suites before promotion |
| Offline/online mismatch | Low-MAE model fails in closed loop | Make MuJoCo rollout gates mandatory |
| Simulator contention | Empty or corrupted datasets | Preserve collector-owned simulator lifecycle |
| Premature hardware testing | Damage or unsafe motion | Enforce staged H0-H5 hardware gates |
| Brain/body boundary erosion | Unsafe planner-generated control | Accept only structured skills and bounded context |
| Unresolved navigation goals | Robot walks without knowing target, route, or obstacles | Refuse raw destination language until target resolution, localization, route, and local safety checks exist |

## Milestone-to-gate map

| Milestone | Gate | Primary evidence |
|---|---|---|
| M4 | G4 | Official/Soridormi parity traces and first-divergence reports |
| M5 | G5 | Policy contract, profile validation, package checks |
| M6 | G6 | Dataset, training, offline evaluation, MuJoCo rollout |
| M7 | G7 | Skill manifest, validation, refusal tests |
| M8 | G8 | Scenario and social-skill acceptance reports |
| M9 | G9 | Coverage gate, prepared gate, training-ready manifest |
| M10 | G10 | Context profile contract, scenario suite, clearance evidence |
| M11 | G11 | Held-out multi-seed scenario suites |
| M12 | G12 | Frozen-baseline comparison and non-regression evidence |
| M13 | G13 | H0-H5 hardware safety and execution logs |
| M14 | G14 | Structured integration and safety-refusal tests |
| M15 | G15 | Navigation contract, target-resolution refusal, route/local-planning evidence |

## Immediate execution plan

1. Use `clearance_liftscale_stack_s143_step090_offset005` as the retained
   residual reference. Reject the post-`s143` scalar, pre-roll, MLP, reflex, and
   startup-tail probes through `clearance_s201_microreflex_s207` for promotion.
2. Before launching another training run, summarize the existing candidate
   history:

```bash
./scripts/report_clearance_candidate_history.sh
```

   The current artifact history reports no ready candidate and no
   reference-beating blocked candidate, so keep `s143` as the retained blocked
   reference and move to a broader clearance redesign or a higher-clearance
   teacher instead of another narrow scalar/reflex/guard retune.
3. Close the M10 engineering-process section with the dry/offline validation
   gate:

```bash
./scripts/validate_m10_engineering_process.sh
```

   This gate validates reporting, readiness analysis, follow-camera planning,
   evidence packaging, docs, and focused tests. It intentionally does not train,
   launch MuJoCo, or send actuator commands.
4. Train the next clearance candidate to beat `s143` on low-clearance ratio in
   all three scenarios while preserving p50 clearance, no-fall behavior, and
   movement distance. Screen candidates with:

```bash
./scripts/analyze_clearance_readiness.sh \
  --profile-name <candidate_profile> \
  --suite-dir artifacts/scenario_eval/<candidate_profile> \
  --reference-profile-name clearance_liftscale_stack_s143_step090_offset005 \
  --reference-suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 \
  --output-dir artifacts/clearance_readiness/<candidate_profile> \
  --json \
  --require-reference-improvement
```

5. Run the candidate with `--viewer --follow-camera` for human visual review.
6. Record swing-clearance evidence for all three scenarios.
7. Fill the visual-review template and rebuild the evidence package.
8. Focus training commands/objective on start-stop and turning clearance while
   preserving the near-passing flat result, or acquire a higher-clearance
   teacher.
9. Pass the quantitative clearance readiness gate without regressing the
   original scenario suite.
10. Compare the new candidate suite against the official teacher with
   `compare_policy_teacher_suite.sh`.
11. Begin M11 held-out scenario development only after G10 passes.

## Project success criteria

Soridormi succeeds when:

- policies are replaceable through stable contracts;
- datasets and training runs are reproducible;
- locomotion works across held-out scenarios;
- failures are visible rather than hidden by tuning;
- simulation and hardware share the same runtime concepts;
- hardware rollout is staged and reversible; and
- Chromie controls the body exclusively through safe structured skills.

### M10 residual-training refinement notes

- [x] Add weighted fixed-command residual training commands.
- [x] Add single-reset training sequences for ramped start/stop and curve episodes.
- [x] Add worst-case score blending so CEM cannot hide a scenario regression behind
  a weighted-average improvement.
- [x] Add optional final score breakdown for the best residual, so each run can
  identify whether flat, start/stop, or curve-style training objectives remain
  the limiting score before full scenario evaluation.
- [x] Add per-objective episode diagnostics to the final breakdown, including
  completed steps, termination state, median/min swing clearance, and
  low-clearance ratio, so clearance failures are visible without rerunning a full
  scenario-suite report.
- [x] Add per-segment diagnostics for sequence objectives, so start, cruise,
  turn, and stop segments can be compared directly when a sequence-level score
  is low or has no swing-clearance samples.
- [x] Add optional per-step score normalization for mixed-length objectives, so
  worst-case selection is based on comparable objective scores instead of raw
  total reward length.
- [x] Add an episodic clearance-gap penalty so optimization can distinguish
  shallow below-target swings from deep below-target swings when
  `low_clearance_ratio` is saturated at 1.0.
- [x] Add `--no-zero-candidate` so warm-start probes can force CEM to evaluate
  actual parameter moves instead of re-exporting the unchanged checkpoint.
- [x] Add an episodic lower-quantile clearance-gap penalty
  (`--episodic-clearance-quantile` and
  `--episodic-clearance-quantile-gap-weight`) so optimization can target the
  lower tail after p50 clearance already exceeds the gate threshold.

Recommended next M10 experiment: warm-start from
`m10_command_state_mlp_cem4x14_s79`, keep one fixed flat-walk command, add
weighted start/stop and curve sequences, use a nonzero worst-case score weight,
set `--score-normalization per_step`, add a small nonzero
`--episodic-clearance-gap-weight` when the turn objective has saturated
low-clearance ratio, and enable `--final-score-breakdown` before accepting any
clearance improvement for G10 evidence.

2026-06-22 update: `clearance_gap_sequence_restored_s83` followed this
warm-start path and produced a valid runtime profile, but failed the
three-scenario clearance gate and did not improve on
`m10_command_state_mlp_cem4x14_s79` because the wrapper used `residual_scale
0.05` instead of the retained checkpoint's `0.1`. The wrapper now defaults to
`0.1`. The corrected `clearance_gap_sequence_scale_preserved_s89` improved
movement and flat p50 clearance. The first low-ratio continuation,
`clearance_lowratio_sequence_s97`, improved curve clearance and total distance.
The stricter turn-focused `clearance_lowratio_turnfocus_s101` re-exported the
same ONNX as `s97`, so it is a rejected probe. The tighter-search continuation
`clearance_lowratio_refine_s103` improved distance and max low-clearance ratio.
The multi-command continuation `clearance_lowratio_multicmd_s107` improved
distance, curve p50, and max low-clearance ratio. The curve-focused
`clearance_lowratio_curvepush_s109` slightly improved curve p50 but regressed
distance and max low-clearance ratio, so it is a rejected probe. The gate-push
continuation `clearance_lowratio_gatepush_s111` is the best retained blocked
candidate so far. The target-lift probe `clearance_lowratio_targetlift_s113`
exported the same ONNX as `s111`, so it is rejected. The no-zero probe
`clearance_lowratio_forced_s115` moved away from `s111` but regressed training
clearance metrics. The suite-command probe `clearance_lowratio_suitecmd_s117`
reduced curve low-clearance ratio slightly but regressed flat/start
low-clearance ratio and total distance, so it is rejected. The quantile-tail
probe `clearance_lowratio_quantile_s119` added a lower-tail clearance objective
at q=0.25 and produced a distinct ONNX from `s111`. It remains `0/3` and
`BLOCKED_BY_CLEARANCE_GATE`, with no falls. It improves total distance
(`~1.938 m`) and worst-case low-clearance ratio (`~0.473`) versus `s111`, but
regresses flat low-clearance ratio (`~0.408` versus `~0.388`), so it is not a
clean replacement. The next run should compare against both `s111` and `s119`
before consuming full G10 evidence time.

2026-06-23 update: stacked residual teachers are now supported, so residual
profiles can train on top of prior residual profiles. The best retained blocked
candidate is `clearance_liftscale_stack_s127`, trained from
`clearance_contactlift_stack_s121` with actor kind `contact_phase_lift` and
explicit `residual_scale=0.16`. It remains `0/3` and
`BLOCKED_BY_CLEARANCE_GATE`, with no falls, but improves total distance to
`~2.189 m`, all p50 clearances above `0.015 m`, and worst-case low-clearance
ratio to `~0.409`. Reject `clearance_contactlift_stack_s123`,
`clearance_cmdlift_stack_s125`, `clearance_liftscale_stack_s129`, and
`clearance_harmonic_stack_s131` for promotion. The next run should beat `s127`
on low-clearance ratio in all three scenarios while preserving its movement
distance.

Host wrapper:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/train_clearance_residual_policy.sh --dry-run
./scripts/train_clearance_residual_policy.sh
```

The wrapper is a reproducible experiment launcher only. Promotion still requires
the scenario-suite, clearance-readiness, visual inspection, and teacher
comparison gates above.
