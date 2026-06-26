# Soridormi WBC clearance-control contract

This document starts the motion-control section after the M10 engineering
process gate. The goal is to make WBC/body-control work testable before tuning
or learning any model.

The first stage is deliberately sim-only and parameter-bounded. It does not
create hardware commands, does not let Chromie send raw joint or `action_14d`
commands, and does not replace the M10 scenario evidence path.

## Contract

The first contract is:

```text
configs/wbc/open_duck_mini_v2_clearance_contract.json
```

It defines:

- required WBC/gait clearance parameters;
- allowed min/default/max values;
- sim-only safety constraints;
- the six-scenario MuJoCo evaluation surface used for WBC clearance work;
- candidate parameter sets for startup/tail and turning clearance;
- post-implementation evaluation commands.

The contract is a planning and validation surface. Because the WBC runtime
backend is not implemented yet, candidates are marked
`WAITING_FOR_WBC_RUNTIME_BACKEND`.

The WBC clearance scenario surface is deliberately small and flat-ground:

- `flat_walk_varied_speed_v1`
- `start_stop_velocity_ramp_v1`
- `curve_turn_walk_v1`
- `startup_tail_clearance_v1`
- `s_turn_reversal_v1`
- `turn_stop_settle_v1`

The first three remain the M10 promotion core. The last three enrich the suite
before WBC tuning by stressing startup/stop tails, turn reversal, and
turn-to-stop settling.

Validate that surface before tuning:

```bash
./scripts/validate_pre_wbc_scenario_surface.sh
```

This is a dry/offline gate. It checks the WBC contract, the default ready
locomotion suite, run-plan generation, and the M10 core/enrichment split; it
does not launch MuJoCo or create candidate profiles.

## Planning Harness

Generate the bounded experiment plan:

```bash
./scripts/plan_wbc_clearance_experiment.sh
```

This writes:

```text
artifacts/wbc_clearance_experiments/open_duck_mini_v2_v0/wbc_clearance_experiment_plan.json
artifacts/wbc_clearance_experiments/open_duck_mini_v2_v0/wbc_clearance_experiment_plan.md
```

The command is dry/offline. It does not train, launch MuJoCo, create runtime
profiles, or send actuator commands.

## Validation

Run the WBC clearance contract gate:

```bash
./scripts/validate_wbc_clearance_contract.sh
```

This validates the contract, planning harness, docs, and focused tests. It is
also dry/offline.

## Promotion Rule

A WBC clearance candidate cannot be promoted from this contract alone. After a
runtime backend exists, each candidate must still pass:

```bash
./scripts/validate_m10_engineering_process.sh
./scripts/evaluate_scenario_suite.sh \
  --backend mujoco \
  --profile <candidate_profile> \
  --output-dir artifacts/scenario_eval/<candidate_profile> \
  --json
./scripts/analyze_clearance_readiness.sh \
  --profile-name <candidate_profile> \
  --suite-dir artifacts/scenario_eval/<candidate_profile> \
  --reference-profile-name clearance_liftscale_stack_s143_step090_offset005 \
  --reference-suite-dir artifacts/scenario_eval/clearance_liftscale_stack_s143_step090_offset005 \
  --output-dir artifacts/clearance_readiness/<candidate_profile> \
  --json \
  --require-reference-improvement
```

Only after quantitative readiness passes should a human run follow-camera
visual review and teacher comparison.
