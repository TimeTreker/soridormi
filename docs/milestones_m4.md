# Soridormi M4 milestones

## M4.0 First walk integration

- Real MuJoCo foot-contact observation path.
- Open Duck-style accelerometer bias.
- First-walk server/runtime scripts.
- MuJoCo home/default export helper.

## M4.1 ONNX forward policy compatibility

- ONNX-only forward policy entrypoint.
- Step-based imitation phase.
- First-state default actuator bootstrap.
- Forward command ramp.

## M4.2 Runnable policy engine and model replacement foundation

- Policy profiles in `configs/policies/*.yaml`.
- One-command runner: `scripts/run_policy_experiment.sh`.
- Model checker: `scripts/check_policy_model.sh`.
- Explicit ONNX input/output name/shape/type contract.
- Base pose logging and forward-displacement analysis.

Success for M4.2 means the system can run a named policy profile, validate its
ONNX model contract, log state/action/debug data, survive/reset/restart in sim,
and report whether the base moved forward before falling.

## M4.3: ONNX stepping compatibility

Goal: keep the ONNX policy path, but diagnose/fix the current symptom where the
robot wiggles its body while both feet remain planted.

Added:

- optional ONNX action postprocessor
- leg-action gain and head/neck damping controls
- `/soridormi/policy_raw_action` logging
- foot world-position logging in `RobotState.feet_position_xyz`
- `open_duck_stepping_debug` profile
- `scripts/run_stepping_policy_experiment.sh`
- analyzer output for foot lift and leg/head action magnitude

This is still ONNX-controlled. It is not an open-loop gait controller.


## M4.4 — Official Open Duck baseline reproduction

Goal: run the official Open Duck Mini v2 MuJoCo inference path inside the Soridormi Docker stack before tuning Soridormi's wrapper further.

Added:

- `src/soridormi_sim/official_open_duck_baseline.py`
- `scripts/run_official_forward_baseline.sh`
- `scripts/run_official_keyboard_baseline.sh`
- `scripts/show_official_baseline_summary.sh`
- `docs/official_open_duck_baseline_m4.md`

Success condition:

```text
The official baseline runs from one command and writes data/official_baseline/latest_official_baseline.json.
```

Decision rule:

```text
If the official baseline steps forward, port the exact mismatch into Soridormi.
If the official baseline also only wiggles, first fix model/assets/dependency/command baseline before training.
```


## M4.5 Official inference behavior in Soridormi runtime

- Add MuJoCo `home` keyframe startup/reset mode.
- Expose backend `actuator_ctrl` in `RobotState`.
- Bootstrap ONNX policy defaults from `actuator_ctrl` when available.
- Add an official-compatible server launcher for normal Soridormi policy experiments.

## M4.6 Official vs Soridormi trace comparison

Status: implemented.

Adds per-step official trace export, raw Soridormi policy-observation logging, reset-at-experiment-start support, and trace comparison scripts. The goal is to identify the exact observation/action/control mismatch between the official walking baseline and Soridormi's policy runtime.

## M4.7 Official phase compatibility

M4.7 aligns Soridormi's imitation phase with Open Duck's official MuJoCo inference loop.

- step phase advances before observation construction;
- phase period can be loaded from Open Duck `polynomial_coefficients.pkl`;
- policy profiles use `period_steps: auto` and `reference_data`;
- runtime logs describe the phase period source.


## M4.8 — Runtime reference-data mount for official phase

- Mount `workspace/Open_Duck_Playground` into the runtime container.
- Export `SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA` from policy profiles.
- Fail fast when Open Duck phase reference data is unavailable.
- Official-compatible profiles should report `period_source=reference_data`, not `fallback_50`.


## M4.9: Official sensor/contact compatibility

Status: implemented.

M4.9 aligns the Soridormi MuJoCo backend more closely with the official Open Duck inference runner after M4.8 fixed phase:

- official reset/startup sequence,
- required MuJoCo gyro/accelerometer sensors in official mode,
- body-only official foot contact checks,
- policy profiles and compose forwarding for the new MuJoCo compatibility flags.

Goal: reduce remaining `accelerometer_xyz`, `gyro_xyz`, and `feet_contacts` trace error, and improve forward displacement in the Soridormi ONNX runtime.

## M4.11: Official observation/action parity checker

M4.10 proved Soridormi's MuJoCo backend reproduces official dynamics exactly when replaying official motor targets. M4.11 adds a checker for the remaining policy-side mismatch:

```text
official obs[101] -> ONNX action[14]
soridormi obs[101] -> ONNX action[14]
```

Run:

```bash
./scripts/check_latest_observation_action_parity.sh
```

The checker re-runs the ONNX model on both official and Soridormi observations and reports whether the ONNX wrapper is compatible and which observation segments still diverge.

## M4.12: synchronous simulator policy stepping

- Add `step_command` API request.
- Add `RobotApiClient.step_motor_command()`.
- Add runtime `SORIDORMI_SIM_SYNC_STEP` mode.
- Enable sync stepping for official Open Duck policy profiles.
- Preserve old async `get_state` / `send_command` behavior for generic runtime and future hardware backend.
