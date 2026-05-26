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
