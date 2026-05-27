# Next Session Prompt

Copy this into a new ChatGPT/Claude session:

```text
We are working on Soridormi: https://github.com/TimeTreker/soridormi.git main.

Please first read these project-local files:
- CLAUDE.md
- LLM_CONTEXT.md
- AGENTS.md
- docs/LLM_HANDOFF_M4.md
- docs/ROADMAP_M4_M7.md

Context summary:
Soridormi is a sim-to-real Open Duck Mini v2 engineering platform. M4 made the official Open Duck ONNX walking policy reproducible through Soridormi's own runtime. Official Open Duck baseline walks forward. Official motor-target replay through Soridormi backend matches official trajectory exactly. Soridormi ONNX wrapper reproduces official actions exactly when given official observations. M4.13 added first-divergence diagnostics and fixed a policy-default/motor-target aliasing bug; official and Soridormi compared traces now match for the checked window at strict thresholds.

Please continue with M5: model replacement interface. Do not train a new model yet. M5.1 added static observation/action contract export, M5.2 made `check_policy_model.sh --profile NAME` validate both the static runtime contract and the actual ONNX IO metadata before runtime, M5.3 centralized ONNX Runtime provider selection so CUDA/CPU/TensorRT choices can be inspected and enforced by preflight checks, and M5.4 added a profile scaffolder for drop-in replacement ONNX profiles, M5.5 added a suite validator for all policy profiles with optional ONNX/provider checks, M5.6 added a shared local/GitHub CI static-check gate for replacement profile workflows, M5.7 added a replacement manifest exporter for reproducible profile/model handoffs, M5.8 added acceptance artifact bundles, and M5.9 added verifiable policy package tarballs for handoff/release. M5.10 added policy package install/restore, M5.11 added package inventory/indexing, and M6.1 started the training pipeline with a runtime-log-to-supervised-dataset exporter, and M6.2 added dataset validation plus deterministic train/val/test split preparation, and M6.3 added dataset statistics plus train-only normalization artifacts. Continue toward M6 training data/reward/export workflows while preserving the same runtime/API/backend invariants.
```


## M5.10 policy package install/restore

Adds `./scripts/install_policy_package.sh PACKAGE.policy.tar.gz`, the inverse of the M5.9 package command. It verifies the tarball, restores `profile.yaml` into `configs/policies/`, optionally copies embedded ONNX model bytes into `data/policy_models/<profile>/`, and rewrites `model.path` to the runtime-visible `/data/policy_models/...` location. Use `--force` to overwrite an existing installed profile/model.

## M5.11 policy package index

Adds `./scripts/list_policy_packages.sh` and `python -m soridormi_runtime.policy_package index` to list generated package tarballs with verification status, profile names, model inclusion, timestamps, and SHA256 digests.


## M6.1 training dataset export

Adds `./scripts/export_training_dataset.sh LOG...` and `python -m soridormi_runtime.training_dataset` to export Soridormi runtime `.jsonl`/`.mcap` logs into supervised policy datasets with observation/action samples and a manifest.


## M6.2 training dataset preparation

Use `./scripts/prepare_training_dataset.sh DATASET.jsonl` after exporting supervised logs. It validates the exported JSONL and creates deterministic train/val/test split files with a manifest and split hashes. This is still offline data prep; do not change runtime/backend behavior for training experiments.

## M6.3 training dataset statistics/normalization

Use `./scripts/summarize_training_dataset.sh PREPARED_DATASET` after `prepare_training_dataset.sh`. It computes split-level observation/action statistics, train-only normalization vectors, SHA256 metadata, and a Markdown report. This is still offline training data plumbing; do not alter runtime/backend behavior.

## M6.4 behavior cloning baseline trainer

Use `./scripts/train_behavior_clone.sh PREPARED_DATASET` after `summarize_training_dataset.sh`. It trains a deterministic NumPy linear behavior-cloning baseline and writes model/metrics/report artifacts.

M6.5 linear behavior-clone runtime profile

Use `./scripts/create_linear_bc_profile.sh NAME --model data/training_runs/.../linear_behavior_clone.npz --template open_duck_forward` to scaffold a runtime profile for the baseline NPZ artifact. The generated profile uses `model.kind: linear_behavior_clone`, and the policy controller selects `LinearBehaviorClonePolicy` through `SORIDORMI_POLICY_BACKEND`. This is a rollout smoke-test path for offline training artifacts; full neural/ONNX export remains later M6 work.

## M6.6 offline policy evaluation

Use `./scripts/evaluate_policy_profile.sh PROFILE PREPARED_DATASET` after creating a runtime profile. It evaluates ONNX or `linear_behavior_clone` policies against prepared train/val/test supervised splits, writes `evaluation.json` and `evaluation_report.md`, optionally writes prediction JSONL files, and supports threshold flags such as `--max-test-mae` for offline acceptance before MuJoCo rollout.
