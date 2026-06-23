# M10 Clearance Refinement Handoff - 2026-06-22

This handoff preserves the incoming development state for continuing M10 MuJoCo
clearance refinement from another machine. It records the user-provided handoff
verbatim in spirit, then adds the metric-only continuation produced in this
session.

Hardware is excluded. Do not claim human visual review unless a direct
viewer/follow-camera pass was actually performed. Metric-grounded review is
separate.

## Incoming Handoff

Working tree:

```text
/home/chromie/github/soridormi
branch: main
latest pushed Soridormi commit: d05edd4 Preserve residual scale in clearance refinement
```

Read first:

```text
AGENTS.md
LLM_CONTEXT.md
docs/SORIDORMI_EXECUTION_ROADMAP.md
docs/SORIDORMI_TARGET_AND_ROADMAP.md
docs/M6_SIM_TRAINING_LOOP.md
```

Current focus:

```text
M10 MuJoCo clearance refinement.
Hardware is excluded for now.
Do not claim human visual review unless actually done with viewer/follow-camera.
Metric-grounded review is separate.
```

Current best retained candidate at handoff start:

```text
clearance_lowratio_sequence_s97
```

Status at handoff start:

```text
Still blocked by G10 clearance gate, but best so far.
```

Metrics for `clearance_lowratio_sequence_s97`:

```text
total distance: ~1.729m
flat:       p50 clearance ~0.01560m, low-clearance ratio ~0.437
start-stop: p50 ~0.01466m, low-clearance ratio ~0.520
curve:      p50 ~0.01341m, low-clearance ratio ~0.754
result:     0/3 scenarios pass, BLOCKED_BY_CLEARANCE_GATE
falls:      none
```

Important recent fix:

```text
scripts/train_clearance_residual_policy.sh now defaults to residual_scale=0.1
and exposes --residual-scale. This fixed the earlier s83 issue where the
wrapper changed the retained warm-start scale from 0.1 to 0.05.
```

Do not promote:

```text
context_stage1_three_scenario_10ep_e80
clearance_gap_sequence_restored_s83
clearance_gap_sequence_scale_preserved_s89
clearance_lowratio_sequence_s97
```

Next best work from the incoming handoff:

1. Train the next clearance candidate against `s97`.
2. Reduce low-clearance ratio below `0.25` in all scenarios.
3. Lift start-stop and curve p50 clearance above `0.015m`.
4. Preserve no-fall behavior and strong movement distance.

Validation commands used recently:

```bash
bash -n scripts/train_clearance_residual_policy.sh
docker compose -f compose.sim.yaml run --rm runtime bash -lc 'source /opt/venvs/runtime/bin/activate && pytest -q tests/test_residual_scripts_m621.py tests/test_policy_input_features_m10.py'
./scripts/validate_policy_profiles.sh configs/policies/clearance_gap_sequence_scale_preserved_s89.yaml configs/policies/clearance_lowratio_sequence_s97.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile clearance_lowratio_sequence_s97 --output-dir artifacts/scenario_eval/clearance_lowratio_sequence_s97 --json
./scripts/analyze_clearance_readiness.sh --profile-name clearance_lowratio_sequence_s97 --suite-dir artifacts/scenario_eval/clearance_lowratio_sequence_s97 --output-dir artifacts/clearance_readiness/clearance_lowratio_sequence_s97 --json
```

Before training, start sim explicitly:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile context_stage1_three_scenario_10ep_e80 --no-viewer
```

Optional human visual pass later:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile <candidate> --viewer --follow-camera
```

## Same-Session Continuation

Two continuations were run from `clearance_lowratio_sequence_s97` with the
required MuJoCo backend started headless using:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile context_stage1_three_scenario_10ep_e80 --no-viewer
```

`clearance_lowratio_turnfocus_s101`:

```text
status: rejected probe
reason: exported ONNX was byte-identical to s97
```

`clearance_lowratio_refine_s103`:

```text
initial checkpoint: clearance_lowratio_sequence_s97
residual_scale: 0.1
profile contract: PASS
suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance: ~1.838m
flat:       p50 ~0.01634m, low-clearance ratio ~0.438
start-stop: p50 ~0.01559m, low-clearance ratio ~0.456
curve:      p50 ~0.01451m, low-clearance ratio ~0.662
falls:      none
```

Interpretation:

```text
clearance_lowratio_refine_s103 became the best retained blocked candidate at
that point.
It improves total distance, start-stop p50 clearance, curve p50 clearance, and
max low-clearance ratio over s97. It is not promotable: all scenarios still fail
the low-clearance-ratio gate, and curve p50 remains below 0.015m.
```

Metric evidence generated:

```text
artifacts/scenario_eval/clearance_lowratio_refine_s103/suite_summary.json
artifacts/clearance_readiness/clearance_lowratio_refine_s103/clearance_readiness.json
data/rl_finetune/clearance_lowratio_refine_s103/residual_train_report.md
```

Validation commands run for the continuation:

```bash
./scripts/validate_policy_profiles.sh configs/policies/clearance_lowratio_refine_s103.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile clearance_lowratio_refine_s103 --output-dir artifacts/scenario_eval/clearance_lowratio_refine_s103 --json
./scripts/analyze_clearance_readiness.sh --profile-name clearance_lowratio_refine_s103 --suite-dir artifacts/scenario_eval/clearance_lowratio_refine_s103 --output-dir artifacts/clearance_readiness/clearance_lowratio_refine_s103 --json
```

No human viewer/follow-camera visual review was performed in this continuation.

## 2026-06-23 No-Zero Probe Continuation

The residual trainer now exposes `--no-zero-candidate`, which prevents CEM from
forcing the current warm-start mean into every generation. This was added after
`clearance_lowratio_targetlift_s113` exported an ONNX byte-identical to `s111`.

`clearance_lowratio_forced_s115`:

```text
status: rejected probe
initial checkpoint: clearance_lowratio_gatepush_s111
training mode: clearance wrapper with --no-zero-candidate
profile contract: PASS
reason: produced a different ONNX but regressed training clearance metrics
```

`clearance_lowratio_suitecmd_s117`:

```text
status: rejected probe
initial checkpoint: clearance_lowratio_gatepush_s111
training mode: direct train_residual_policy.sh suite-command objectives with
--no-zero-candidate
profile contract: PASS
suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance: ~1.892m
flat:       p50 ~0.01597m, low-clearance ratio ~0.430
start-stop: p50 ~0.01603m, low-clearance ratio ~0.446
curve:      p50 ~0.01522m, low-clearance ratio ~0.489
falls:      none
reason: slightly improved curve low-clearance ratio but regressed flat/start
low-clearance ratio and total distance relative to s111
```

Metric evidence generated:

```text
artifacts/scenario_eval/clearance_lowratio_suitecmd_s117/suite_summary.json
artifacts/clearance_readiness/clearance_lowratio_suitecmd_s117/clearance_readiness.json
data/rl_finetune/clearance_lowratio_forced_s115/residual_train_report.md
data/rl_finetune/clearance_lowratio_suitecmd_s117/residual_train_report.md
```

Validation commands run:

```bash
docker compose -f compose.sim.yaml run --rm runtime bash -lc 'source /opt/venvs/runtime/bin/activate && pytest -q tests/test_residual_policy_m619_m621.py tests/test_residual_scripts_m621.py'
./scripts/validate_policy_profiles.sh configs/policies/clearance_lowratio_forced_s115.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/validate_policy_profiles.sh configs/policies/clearance_lowratio_suitecmd_s117.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile clearance_lowratio_suitecmd_s117 --output-dir artifacts/scenario_eval/clearance_lowratio_suitecmd_s117 --json
./scripts/analyze_clearance_readiness.sh --profile-name clearance_lowratio_suitecmd_s117 --suite-dir artifacts/scenario_eval/clearance_lowratio_suitecmd_s117 --output-dir artifacts/clearance_readiness/clearance_lowratio_suitecmd_s117 --json
```

No human viewer/follow-camera visual review was performed in this continuation.

## 2026-06-23 Continuation

The next continuation started MuJoCo explicitly with the same headless command:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile context_stage1_three_scenario_10ep_e80 --no-viewer
```

`clearance_lowratio_multicmd_s107`:

```text
initial checkpoint: clearance_lowratio_refine_s103
residual_scale: 0.1
profile contract: PASS
suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance: ~1.903m
flat:       p50 ~0.01614m, low-clearance ratio ~0.420
start-stop: p50 ~0.01578m, low-clearance ratio ~0.425
curve:      p50 ~0.01456m, low-clearance ratio ~0.519
falls:      none
```

Interpretation:

```text
clearance_lowratio_multicmd_s107 became the best retained blocked candidate at
that point. It improves total distance, start-stop low-clearance ratio, curve
p50 clearance, and max low-clearance ratio over s103. It is not promotable: all
scenarios still fail the low-clearance-ratio gate, and curve p50 remains below
0.015m.
```

`clearance_lowratio_curvepush_s109`:

```text
status: rejected probe
reason: slightly improved curve p50 but worsened total distance and max
low-clearance ratio relative to s107
suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance: ~1.893m
flat:       p50 ~0.01625m, low-clearance ratio ~0.426
start-stop: p50 ~0.01589m, low-clearance ratio ~0.454
curve:      p50 ~0.01464m, low-clearance ratio ~0.541
falls:      none
```

Metric evidence generated:

```text
artifacts/scenario_eval/clearance_lowratio_multicmd_s107/suite_summary.json
artifacts/clearance_readiness/clearance_lowratio_multicmd_s107/clearance_readiness.json
data/rl_finetune/clearance_lowratio_multicmd_s107/residual_train_report.md
artifacts/scenario_eval/clearance_lowratio_curvepush_s109/suite_summary.json
artifacts/clearance_readiness/clearance_lowratio_curvepush_s109/clearance_readiness.json
data/rl_finetune/clearance_lowratio_curvepush_s109/residual_train_report.md
```

Validation commands run for the retained continuation:

```bash
./scripts/validate_policy_profiles.sh configs/policies/clearance_lowratio_multicmd_s107.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile clearance_lowratio_multicmd_s107 --output-dir artifacts/scenario_eval/clearance_lowratio_multicmd_s107 --json
./scripts/analyze_clearance_readiness.sh --profile-name clearance_lowratio_multicmd_s107 --suite-dir artifacts/scenario_eval/clearance_lowratio_multicmd_s107 --output-dir artifacts/clearance_readiness/clearance_lowratio_multicmd_s107 --json
```

No human viewer/follow-camera visual review was performed in this continuation.

## 2026-06-23 Gate-Push Continuation

The retained continuation started from `clearance_lowratio_multicmd_s107`:

```text
clearance_lowratio_gatepush_s111:
initial checkpoint: clearance_lowratio_multicmd_s107
residual_scale: 0.1
profile contract: PASS
suite: 0/3, BLOCKED_BY_CLEARANCE_GATE
total distance: ~1.928m
flat:       p50 ~0.01679m, low-clearance ratio ~0.388
start-stop: p50 ~0.01626m, low-clearance ratio ~0.423
curve:      p50 ~0.01506m, low-clearance ratio ~0.496
falls:      none
```

Interpretation:

```text
clearance_lowratio_gatepush_s111 is the current best retained blocked
candidate. It improves total distance and low-clearance ratio in all three
scenarios relative to s107. It also brings curve p50 clearance above 0.015m, so
all three p50 clearance checks now meet target. It is not promotable: all
scenarios still fail the low-clearance-ratio gate.
```

The target-lift probe raised the training target to `0.0165m`:

```text
clearance_lowratio_targetlift_s113:
status: rejected probe
reason: exported ONNX was byte-identical to s111
```

Metric evidence generated:

```text
artifacts/scenario_eval/clearance_lowratio_gatepush_s111/suite_summary.json
artifacts/clearance_readiness/clearance_lowratio_gatepush_s111/clearance_readiness.json
data/rl_finetune/clearance_lowratio_gatepush_s111/residual_train_report.md
data/rl_finetune/clearance_lowratio_targetlift_s113/residual_train_report.md
```

Validation commands run for the retained continuation:

```bash
./scripts/validate_policy_profiles.sh configs/policies/clearance_lowratio_gatepush_s111.yaml --robot-config configs/robots/open_duck_mini_v2.yaml
./scripts/evaluate_scenario_suite.sh --backend mujoco --profile clearance_lowratio_gatepush_s111 --output-dir artifacts/scenario_eval/clearance_lowratio_gatepush_s111 --json
./scripts/analyze_clearance_readiness.sh --profile-name clearance_lowratio_gatepush_s111 --suite-dir artifacts/scenario_eval/clearance_lowratio_gatepush_s111 --output-dir artifacts/clearance_readiness/clearance_lowratio_gatepush_s111 --json
```

No human viewer/follow-camera visual review was performed in this continuation.
