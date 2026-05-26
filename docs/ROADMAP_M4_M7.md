# Soridormi Roadmap M4-M7

## Principle

Engineering first, algorithm second.

Use the official Open Duck policy as a known-good reference. Build a reliable runtime/policy/sim/logging platform first. Then replace the model, train new policies, and later move to real hardware.

## M4: Runnable ONNX policy system

Goal: official ONNX policy runs in Soridormi and produces forward motion in MuJoCo.

Milestone checklist:

- Official Open Duck baseline runs.
- Soridormi ONNX profile runner works.
- Policy model checker validates model input/output.
- State/action/debug logging works.
- Official-vs-Soridormi trace comparison works.
- Official motor target replay works.
- Observation/action parity checker works.
- Soridormi policy loop reproduces official walking sufficiently.

Current status:

- Baseline/replay/ONNX parity are good.
- Soridormi closed-loop still diverges after step 0.
- M4.13 should implement exact loop-order parity and first divergence diagnostics.

## M5: Model replacement interface

Goal: new compatible ONNX policies can be dropped in without code changes.

Expected features:

- `configs/policies/*.yaml` as source of truth.
- Model path, input/output names, shapes, action scale, max motor velocity, phase config externalized.
- `check_policy_model.sh --profile NAME` validates model contract.
- `run_policy_experiment.sh PROFILE` runs the selected policy.
- Observation/action contract documented.
- Multiple profiles supported.

## M6: Training pipeline

Goal: train new policies and run them through the same runtime.

Expected features:

- Data collection from simulator.
- Dataset schema tied to `RobotState`, `MotorCommand`, observation vectors, and actions.
- Reward/task definitions.
- Training entrypoints.
- ONNX export pipeline.
- Automated compatibility validation before runtime.

## M7: Transfer to real robot

Goal: run the same runtime contracts against hardware backend.

Expected phases:

- M7.1 Jetson runtime image.
- M7.2 hardware backend skeleton.
- M7.3 read-only hardware state streaming.
- M7.4 motor command dry-run mode.
- M7.5 torque/position/current limits.
- M7.6 emergency stop/watchdog.
- M7.7 low-power single-joint test.
- M7.8 standing pose on real robot.
- M7.9 tethered first walking test.

Core invariant:

```text
Same runtime.
Same policy interface.
Same RobotState.
Same MotorCommand.
Different backend.
```
