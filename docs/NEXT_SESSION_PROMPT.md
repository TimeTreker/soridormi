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

Please continue with M5: model replacement interface. Do not train a new model yet. M5.1 added static observation/action contract export, M5.2 made `check_policy_model.sh --profile NAME` validate both the static runtime contract and the actual ONNX IO metadata before runtime, M5.3 centralized ONNX Runtime provider selection so CUDA/CPU/TensorRT choices can be inspected and enforced by preflight checks, and M5.4 added a profile scaffolder for drop-in replacement ONNX profiles, M5.5 added a suite validator for all policy profiles with optional ONNX/provider checks, M5.6 added a shared local/GitHub CI static-check gate for replacement profile workflows, M5.7 added a replacement manifest exporter for reproducible profile/model handoffs, M5.8 added acceptance artifact bundles, and M5.9 added verifiable policy package tarballs for handoff/release. Continue hardening compatible ONNX replacement workflows while preserving the same runtime/API/backend invariants.
```


## M5.10 policy package install/restore

Adds `./scripts/install_policy_package.sh PACKAGE.policy.tar.gz`, the inverse of the M5.9 package command. It verifies the tarball, restores `profile.yaml` into `configs/policies/`, optionally copies embedded ONNX model bytes into `data/policy_models/<profile>/`, and rewrites `model.path` to the runtime-visible `/data/policy_models/...` location. Use `--force` to overwrite an existing installed profile/model.
