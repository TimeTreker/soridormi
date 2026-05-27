# M5.1 Model replacement interface

M5 starts after the official Open Duck policy can be reproduced through the
Soridormi runtime. The goal is to make ONNX model replacement boring: a new model
should be selectable by profile, validated before runtime, and documented by a
single observation/action contract.

## Contract export

Use the contract exporter to inspect the runtime interface for a profile:

```bash
./scripts/export_policy_contract.sh open_duck_forward
```

For machine-readable output:

```bash
./scripts/export_policy_contract.sh open_duck_forward --json
```

The exporter does not load the ONNX file and does not start MuJoCo. It statically
combines:

- the selected `configs/policies/*.yaml` profile;
- the robot actuator/default-pose/action-mapping config;
- the canonical 101D Open Duck observation layout;
- the 14D action-to-motor target contract.

It fails if the profile's declared model IO shape is incompatible with the
runtime observation/action sizes, or if an optional declared joint order does not
match the robot contract.


## M5.2 profile/model preflight gate

`check_policy_model.sh --profile NAME` is the runtime preflight gate for model
replacement. It now validates two layers before a policy is run:

1. the static Soridormi contract exported by `policy_contract.py`; and
2. the actual ONNX file input/output metadata.

This means a profile fails fast if it declares an observation size, action size,
model IO shape, dtype, or optional joint order that does not match Soridormi's
runtime interface, even before MuJoCo or the runtime loop start.

Run the gate directly:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward
```

Use JSON output for CI or release automation:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward --json
```

`run_policy_experiment.sh PROFILE` already calls this gate unless
`SORIDORMI_SKIP_POLICY_CHECK=1` is set. Keep the skip flag for emergency local
debugging only; replacement models should pass the preflight gate before runtime.


## M5.3 ONNX execution providers

Soridormi now uses one ONNX provider selection path for both runtime inference and
`check_policy_model.sh`. By default it prefers CUDA when ONNX Runtime reports
`CUDAExecutionProvider`, then keeps CPU as a fallback:

```text
CUDAExecutionProvider,CPUExecutionProvider
```

TensorRT is intentionally not selected by default even when available because it
can introduce engine-build/cache behavior during policy-debug runs. Request it
explicitly when you want to experiment with it.

Check which providers are available and active for a profile:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward
```

Force a provider order:

```bash
SORIDORMI_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
  ./scripts/check_policy_model.sh --profile open_duck_forward
```

Require GPU preflight failure if CUDA is not actually selected/activated:

```bash
./scripts/check_policy_model.sh \
  --profile open_duck_forward \
  --require-provider CUDAExecutionProvider
```

For CPU-only debugging:

```bash
./scripts/check_policy_model.sh --profile open_duck_forward --cpu
# or
SORIDORMI_USE_CUDA_PROVIDER=0 ./scripts/run_policy_experiment.sh open_duck_forward
```



## M5.4 profile scaffolding

Use the profile scaffolder when you have a new ONNX file that should follow the
same Soridormi/Open Duck runtime contract as an existing profile. It clones a
known-good profile, changes the model path, stamps the static observation/action
contract into the YAML, and updates the logging prefix.

```bash
./scripts/create_policy_profile.sh my_replacement \
  --model /models/my_replacement.onnx \
  --template open_duck_forward \
  --description "My replacement ONNX policy"
```

Then validate before runtime:

```bash
./scripts/export_policy_contract.sh my_replacement
./scripts/check_policy_model.sh --profile my_replacement
```

The scaffolder does not load the ONNX model. This is intentional: profile
creation is a static YAML operation, while `check_policy_model.sh` remains the
preflight gate for actual ONNX metadata and provider selection.

Implementation note: most Soridormi Docker commands mount `/app/configs` as
read-only. `create_policy_profile.sh` is the exception; it temporarily asks
Compose to mount `./configs` read-write so the generated YAML lands back in the
host repo under `configs/policies/`.

Useful options:

```bash
# Preview YAML without writing a file
./scripts/create_policy_profile.sh my_replacement \
  --model /models/my_replacement.onnx \
  --stdout

# Write to a custom path
./scripts/create_policy_profile.sh my_replacement \
  --model /models/my_replacement.onnx \
  --output configs/policies/my_replacement.yaml

# Override model IO names/shapes when the compatible ONNX uses different names
./scripts/create_policy_profile.sh my_replacement \
  --model /models/my_replacement.onnx \
  --input-name obs \
  --output-name continuous_actions \
  --input-shape '[1, 101]' \
  --output-shape '[1, 14]'
```


## M5.5 profile-suite validation

Use the suite validator before committing or running replacement experiments. By
default it scans every profile under `configs/policies/` and performs the static
contract checks from M5.1 without loading ONNX models:

```bash
./scripts/validate_policy_profiles.sh
```

For machine-readable output:

```bash
./scripts/validate_policy_profiles.sh --json
```

To validate only selected profiles:

```bash
./scripts/validate_policy_profiles.sh open_duck_forward my_replacement
```

For local/GPU preflight, also load each ONNX file and enforce provider selection:

```bash
./scripts/validate_policy_profiles.sh \
  --check-models \
  --require-provider CUDAExecutionProvider
```

The default static mode is intentionally CI-friendly: it catches bad profile YAML,
contract drift, shape mismatches, duplicate profile names, and joint-order
metadata problems even when the external ONNX artifact is not mounted. Use
`--check-models` in runtime containers or release jobs where model files are
available.


## M5.6 CI static-check gate

Use the local CI static-check script before opening a PR or pushing profile
changes:

```bash
./scripts/ci_static_check.sh
```

When Docker Compose is available, the script runs inside the runtime container by
default so host Python does not need project dependencies such as NumPy or
ONNX Runtime. Set `SORIDORMI_CI_STATIC_CHECK_USE_DOCKER=0` to force host mode,
which is useful in GitHub Actions after `python -m pip install -e '.[dev]'`.

The script intentionally stays artifact-free. It does not load ONNX model files
and does not start MuJoCo. It validates the canonical profile contract, validates
all policy profiles statically, creates a temporary replacement profile through
the M5.4 scaffolder, validates that generated profile, and runs the M5 unit test
set.

For a fast smoke run that skips pytest while still exercising the profile and
scaffolder commands:

```bash
SORIDORMI_CI_SKIP_PYTEST=1 ./scripts/ci_static_check.sh
```

GitHub Actions runs this same script in `.github/workflows/static-check.yml`, so
local and CI profile checks use the same entrypoint. Full ONNX/provider checks
remain a local or release preflight because model artifacts and GPU providers are
not guaranteed in the default static CI environment.


## M5.7 replacement manifest export

Use the manifest exporter when you want a reproducible record of a policy
replacement profile before sharing, releasing, or running longer experiments:

```bash
./scripts/export_policy_manifest.sh open_duck_forward
```

For machine-readable output suitable for release artifacts:

```bash
./scripts/export_policy_manifest.sh open_duck_forward --json
```

The default mode is static and CI-friendly. It includes the selected profile, the
runtime observation/action contract, the declared model path, and a SHA256 hash
when the ONNX file is mounted. Missing model files are warnings by default so the
command can still describe profiles in source-only environments.

For release or GPU preflight, require the model file and load the ONNX metadata:

```bash
./scripts/export_policy_manifest.sh my_replacement \
  --require-model \
  --check-model \
  --require-provider CUDAExecutionProvider \
  --json > data/my_replacement.manifest.json
```

Use `--no-hash` for quick local checks of very large model files when the exact
artifact hash is not needed.



## M5.8 acceptance gate

Use the acceptance gate when a replacement profile is ready for handoff from
configuration work to simulation/runtime experiments. It bundles the static
contract, replacement manifest, profile-suite validation, and a Markdown report
into one timestamped artifact directory.

Static acceptance does not require the ONNX file to be mounted:

```bash
./scripts/accept_policy_profile.sh my_replacement
```

Release/GPU acceptance can require the model artifact and enforce provider
selection:

```bash
./scripts/accept_policy_profile.sh my_replacement \
  --check-model \
  --require-model \
  --require-provider CUDAExecutionProvider
```

Artifacts are written under `data/policy_acceptance/` by default when using the
Docker wrapper:

```text
contract.json
manifest.json
profile_suite.json
acceptance.json
acceptance_report.md
```

The acceptance gate still does not start MuJoCo. It is the last cheap preflight
before running `run_policy_experiment.sh PROFILE`.


## M5.9 replacement package and verifier

Use the package workflow when a replacement profile is ready to move between
machines, releases, or review sessions. It wraps the M5.8 acceptance artifacts
and the profile YAML into a deterministic handoff tarball with file hashes. The
ONNX model is not embedded by default, so source-only packages remain small.

```bash
./scripts/package_policy_profile.sh my_replacement
```

Packages are written under `data/policy_packages/` by default when using the
Docker wrapper and have the suffix `.policy.tar.gz`. Each package contains:

```text
profile.yaml
package_manifest.json
artifacts/contract.json
artifacts/manifest.json
artifacts/profile_suite.json
artifacts/acceptance.json
artifacts/acceptance_report.md
```

Verify a package before sharing or importing it into another checkout:

```bash
./scripts/verify_policy_package.sh data/policy_packages/my_replacement_*.policy.tar.gz
```

For release packages where the ONNX file should travel with the profile, embed
and require the model artifact:

```bash
./scripts/package_policy_profile.sh my_replacement \
  --include-model \
  --require-model \
  --check-model \
  --require-provider CUDAExecutionProvider
```

The verifier checks the package manifest, required files, file sizes, and SHA256
hashes. It does not run MuJoCo or execute the policy.

## Runtime contract

Current Open Duck-compatible replacement models must use:

```text
input:  obs                  shape [1, 101] dtype tensor(float)
output: continuous_actions   shape [1, 14]  dtype tensor(float)
```

Observation segments:

| Segment | Size | Meaning |
|---|---:|---|
| `gyro_xyz` | 3 | IMU angular velocity |
| `accelerometer_xyz` | 3 | IMU acceleration after configured policy bias |
| `command` | 7 | `x`, `y`, `yaw`, `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` |
| `joint_offsets` | 14 | `joint_position - policy_default_position` |
| `joint_velocities_scaled` | 14 | joint velocity multiplied by `dof_vel_scale` |
| `last_action` | 14 | previous policy action |
| `last_last_action` | 14 | action from two inference steps ago |
| `last_last_last_action` | 14 | action from three inference steps ago |
| `motor_targets` | 14 | previous speed-limited motor targets |
| `feet_contacts` | 2 | left/right foot contacts |
| `imitation_phase` | 2 | phase reference |
| **Total** | **101** | |

Action mapping:

```text
raw_target = default_position + action_scale * action
target = speed_limit(raw_target, previous_target, max_motor_velocity * dt)
target = optional_ctrlrange_clip(target)
```

## Replacing a model

1. Copy an existing profile under `configs/policies/`.
2. Change `model.path` to the new ONNX file.
3. Update `model.input_name`, `model.output_name`, shapes, and dtypes if the
   exporter used different names.
4. Run:

```bash
./scripts/export_policy_contract.sh my_profile
./scripts/check_policy_model.sh --profile my_profile
```

Only run the policy after both checks pass.

## Optional profile contract metadata

Profiles may include a static declaration. This is useful when profiles are
shared with trained artifacts:

```yaml
metadata:
  format_version: 1
  policy_family: open_duck_mini_v2

contract:
  observation_size: 101
  action_size: 14
  joint_names:
    - left_hip_yaw
    - left_hip_roll
    # ... all 14 in action order
```

`contract.joint_names` is optional because the robot config remains the source of
truth for actuator order. If present, it must exactly match the runtime order.


## M5.10 policy package install/restore

Adds `./scripts/install_policy_package.sh PACKAGE.policy.tar.gz`, the inverse of the M5.9 package command. It verifies the tarball, restores `profile.yaml` into `configs/policies/`, optionally copies embedded ONNX model bytes into `data/policy_models/<profile>/`, and rewrites `model.path` to the runtime-visible `/data/policy_models/...` location. Use `--force` to overwrite an existing installed profile/model.

## M5.11 policy package index

Adds `./scripts/list_policy_packages.sh`, which scans `data/policy_packages/*.policy.tar.gz`, reads package manifests, verifies package hashes by default, and prints an install-ready package inventory. Use `--json` for automation and `--no-verify` when a fast manifest-only scan is enough.
