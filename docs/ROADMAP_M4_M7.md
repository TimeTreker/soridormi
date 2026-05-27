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

- M4 closed-loop parity work is validated through the first-divergence analyzer.
- Official and Soridormi compared policy traces match for the checked window at strict thresholds after fixing policy-default/motor-target aliasing.
- M5.1 exports the static model replacement contract.
- M5.2 makes `check_policy_model.sh --profile NAME` a combined static-contract plus ONNX-file preflight gate.
- M5.3 centralizes ONNX Runtime execution provider selection and adds CUDA/provider preflight enforcement.
- M5.4 adds a profile scaffolder for drop-in replacement ONNX profiles.
- M5.5-M5.9 add suite validation, CI/static checks, manifest export, acceptance artifacts, and verifiable package handoff.
- M5.5 adds a policy profile-suite validator for static CI checks and optional ONNX/provider preflight.
- M5.6 adds a shared local/GitHub CI static-check gate for replacement profile workflows.
- M5.7 adds a replacement manifest exporter with profile, contract, optional ONNX check, and model SHA256 metadata.
- M5.8 adds an acceptance artifact gate for contract/manifest/profile-suite reports.
- M5.9 adds a policy package exporter and verifier for handoff/release tarballs.
- M5.10-M5.11 complete package install/restore and package inventory/index workflows.
- M6.1 starts the training pipeline by exporting supervised observation/action datasets from Soridormi runtime logs.

## M5: Model replacement interface

Goal: new compatible ONNX policies can be dropped in without code changes.

Expected features:

- `configs/policies/*.yaml` as source of truth.
- Model path, input/output names, shapes, action scale, max motor velocity, phase config externalized.
- `check_policy_model.sh --profile NAME` validates model contract and active ONNX execution providers.
- `run_policy_experiment.sh PROFILE` runs the selected policy.
- Observation/action contract documented.
- Multiple profiles supported.
- Replacement profiles can be accepted into a reproducible contract/manifest/report bundle before runtime.
- Suite-level profile validation available for CI and release preflights.
- Local and GitHub CI static-check entrypoint exercises contract export, profile validation, scaffolding, and M5 unit tests.
- Replacement manifest export records profile/contract/model artifact metadata for handoff and release preflights.
- Accepted profiles can be packaged and verified as tar.gz handoff artifacts with optional embedded model bytes.

## M6: Training pipeline

Goal: train new policies and run them through the same runtime.

Current status:

- M6.1 adds `./scripts/export_training_dataset.sh`, which converts Soridormi JSONL/MCAP runtime logs into supervised `observation -> action` JSONL datasets plus a manifest.

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


## M5.10 policy package install/restore

Adds `./scripts/install_policy_package.sh PACKAGE.policy.tar.gz`, the inverse of the M5.9 package command. It verifies the tarball, restores `profile.yaml` into `configs/policies/`, optionally copies embedded ONNX model bytes into `data/policy_models/<profile>/`, and rewrites `model.path` to the runtime-visible `/data/policy_models/...` location. Use `--force` to overwrite an existing installed profile/model.

## M5.11 policy package index

Adds `./scripts/list_policy_packages.sh`, which scans generated replacement-policy packages, verifies package hashes by default, and reports profile name, model inclusion, timestamp, and package digest for install/release workflows.


## M6.1 training dataset export

Adds `./scripts/export_training_dataset.sh LOG...`, which reads Soridormi runtime `.jsonl` or `.mcap` logs and writes a supervised `soridormi.policy_supervision.v1` dataset under `/data/training_datasets` by default. Each sample preserves the 101D policy observation, 14D policy action, raw action when logged, motor command, compact state summary, next-state summary, policy command, and debug metadata. A sidecar manifest records source logs, sample count, skipped records, expected vector sizes, and dataset SHA256.
