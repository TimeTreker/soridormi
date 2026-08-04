#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

use_docker="${SORIDORMI_CI_STATIC_CHECK_USE_DOCKER:-auto}"
inside_container="${SORIDORMI_CI_STATIC_CHECK_IN_CONTAINER:-0}"

if [ "${inside_container}" != "1" ] && [ "${use_docker}" != "0" ]; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "Soridormi CI static check"
    echo "========================="
    echo "Running inside the runtime container so local host Python dependencies are not required."
    echo "Set SORIDORMI_CI_STATIC_CHECK_USE_DOCKER=0 to force host mode."
    exec docker compose -f compose.sim.yaml run --rm \
      -e SORIDORMI_CI_STATIC_CHECK_IN_CONTAINER=1 \
      -e SORIDORMI_CI_SKIP_PYTEST="${SORIDORMI_CI_SKIP_PYTEST:-0}" \
      -e SORIDORMI_CI_ROBOT_CONFIG="${SORIDORMI_CI_ROBOT_CONFIG:-configs/robots/open_duck_mini_v2.yaml}" \
      -v "${repo_root}/scripts:/app/scripts:ro" \
      runtime bash -lc 'source /opt/venvs/runtime/bin/activate && /app/scripts/ci_static_check.sh'
  elif [ "${use_docker}" = "1" ]; then
    echo "ERROR: SORIDORMI_CI_STATIC_CHECK_USE_DOCKER=1 was requested, but Docker Compose is not available." >&2
    exit 2
  fi
fi

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${repo_root}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${repo_root}/src"
fi

missing_deps=0
python - <<'PY' || missing_deps=1
import importlib.util
missing = [name for name in ("numpy", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing Python dependencies: " + ", ".join(missing))
PY
if [ "${missing_deps}" = "1" ]; then
  cat >&2 <<'EOF_MISSING'
ERROR: host Python dependencies are missing.

Run this script with Docker enabled, or install the development dependencies first:
  SORIDORMI_CI_STATIC_CHECK_USE_DOCKER=1 ./scripts/ci_static_check.sh
  python -m pip install -e '.[dev]'
EOF_MISSING
  exit 2
fi

echo "Soridormi CI static check"
echo "========================="

echo "Checking repository governance..."
python scripts/validate_repository_governance.py

robot_config="${SORIDORMI_CI_ROBOT_CONFIG:-configs/robots/open_duck_mini_v2.yaml}"
echo "Checking canonical policy contract..."
python -m soridormi_runtime.policy_contract open_duck_forward --robot-config "${robot_config}" --validate-only

echo "Validating policy profile suite..."
python -m soridormi_runtime.validate_policy_profiles --robot-config "${robot_config}"

echo "Exporting canonical policy manifest..."
python -m soridormi_runtime.policy_manifest open_duck_forward --robot-config "${robot_config}" --json >/dev/null

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "Checking replacement-profile scaffold workflow..."
python -m soridormi_runtime.create_policy_profile ci_replacement \
  --model /tmp/ci_replacement.onnx \
  --template open_duck_forward \
  --description "CI replacement profile scaffold" \
  --output "${tmpdir}/ci_replacement.yaml" \
  --robot-config "${robot_config}"
python -m soridormi_runtime.validate_policy_profiles "${tmpdir}/ci_replacement.yaml" --robot-config "${robot_config}"
python -m soridormi_runtime.policy_manifest "${tmpdir}/ci_replacement.yaml" --robot-config "${robot_config}" --json >/dev/null
python -m soridormi_runtime.policy_acceptance "${tmpdir}/ci_replacement.yaml" --robot-config "${robot_config}" --profile-only --output-dir "${tmpdir}/acceptance" --json >/dev/null

echo "Checking replacement-policy package workflow..."
python -m soridormi_runtime.policy_package package "${tmpdir}/ci_replacement.yaml"   --robot-config "${robot_config}"   --output-dir "${tmpdir}/packages"   --json >/dev/null
package_path="$(find "${tmpdir}/packages" -name '*.policy.tar.gz' -print -quit)"
python -m soridormi_runtime.policy_package verify "${package_path}" --json >/dev/null
python -m soridormi_runtime.policy_package index --directory "${tmpdir}/packages" --json >/dev/null
python -m soridormi_runtime.policy_package install "${package_path}" \
  --profile-dir "${tmpdir}/installed_profiles" \
  --model-dir "${tmpdir}/installed_models" \
  --runtime-model-prefix /tmp/installed_models \
  --json >/dev/null


echo "Checking policy-training dataset export smoke workflow..."
SORIDORMI_TMPDIR="${tmpdir}" python - <<'PY_DATASET'
import json
import os
from pathlib import Path
root = Path(os.environ["SORIDORMI_TMPDIR"])
log = root / "runtime_training_smoke.jsonl"
observation = [0.0] * 101
action = [0.0] * 14
payloads = []
for step in range(2):
    payloads.append({
        "type": "runtime_step",
        "step_index": step,
        "time_wall_ns": 1_000_000_000 + step * 20_000_000,
        "robot_time": step * 0.02,
        "mode": "onnx_policy",
        "backend": "sim",
        "state": {"joints": {"names": ["j0"], "positions": [0.1 * step], "velocities": [0.0]}},
        "command": {"names": ["j0"], "positions": [0.0]},
        "policy_observation": observation,
        "policy_action": action,
        "policy_debug": {"command": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    })
log.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n", encoding="utf-8")
PY_DATASET
python -m soridormi_runtime.training_dataset "${tmpdir}/runtime_training_smoke.jsonl" \
  --output "${tmpdir}/training_dataset.jsonl" \
  --manifest "${tmpdir}/training_dataset.manifest.json" \
  --json >/dev/null
python -m soridormi_runtime.training_dataset_prepare "${tmpdir}/training_dataset.jsonl" \
  --output-dir "${tmpdir}/prepared_training_dataset" \
  --seed 123 \
  --json >/dev/null
python -m soridormi_runtime.training_dataset_stats "${tmpdir}/prepared_training_dataset" \
  --json >/dev/null
python -m soridormi_runtime.train_behavior_clone "${tmpdir}/prepared_training_dataset" \
  --output-dir "${tmpdir}/behavior_clone_baseline" \
  --json >/dev/null
python -m soridormi_runtime.create_linear_bc_profile ci_linear_bc \
  --model "${tmpdir}/behavior_clone_baseline/linear_behavior_clone.npz" \
  --template open_duck_forward \
  --description "CI linear behavior-clone profile" \
  --output "${tmpdir}/ci_linear_bc.yaml" \
  --robot-config "${robot_config}" >/dev/null
python -m soridormi_runtime.policy_contract "${tmpdir}/ci_linear_bc.yaml" --robot-config "${robot_config}" --validate-only
python -m soridormi_runtime.check_policy_model --profile "${tmpdir}/ci_linear_bc.yaml" --robot-config "${robot_config}" --json >/dev/null
python -m soridormi_runtime.evaluate_policy_profile "${tmpdir}/ci_linear_bc.yaml" "${tmpdir}/prepared_training_dataset"   --output-dir "${tmpdir}/policy_evaluation"   --splits train,val,test   --max-samples-per-split 2   --json >/dev/null
python -m soridormi_runtime.policy_candidate_leaderboard "${tmpdir}/policy_evaluation" \
  --output-dir "${tmpdir}/candidate_leaderboard" \
  --max-test-mae 1.0 \
  --require-promotable \
  --json >/dev/null
python -m soridormi_runtime.promote_policy_candidate "${tmpdir}/candidate_leaderboard" \
  --target-profile ci_promoted_linear_bc \
  --output-dir "${tmpdir}/promoted_profiles" \
  --records-dir "${tmpdir}/policy_promotions" \
  --robot-config "${robot_config}" \
  --json >/dev/null

echo "Checking bounded rollout smoke wrapper..."
./scripts/run_policy_rollout_smoke.sh --help >/dev/null
./scripts/accept_policy_rollout.sh --help >/dev/null
python -m soridormi_runtime.rollout_acceptance "${tmpdir}/runtime_training_smoke.jsonl" \
  --output-dir "${tmpdir}/rollout_acceptance" \
  --min-policy-records 2 \
  --min-robot-duration 0.01 \
  --json >/dev/null

if [ "${SORIDORMI_CI_SKIP_PYTEST:-0}" != "1" ]; then
  echo "Running replaceable policy interface unit tests..."
  pytest -q \
    tests/test_check_policy_model.py \
    tests/test_policy_contract.py \
    tests/test_policy_model_contract_gate.py \
    tests/test_onnx_providers.py \
    tests/test_create_policy_profile.py \
    tests/test_validate_policy_profiles.py \
    tests/test_ci_static_check.py \
    tests/test_policy_manifest.py \
    tests/test_policy_acceptance.py \
    tests/test_policy_package.py \
    tests/test_policy_package_index.py \
    tests/test_training_dataset.py \
    tests/test_training_dataset_prepare.py \
    tests/test_training_dataset_stats.py \
    tests/test_train_behavior_clone.py \
    tests/test_linear_behavior_clone_policy.py \
    tests/test_evaluate_policy_profile.py \
    tests/test_policy_candidate_leaderboard.py \
    tests/test_promote_policy_candidate.py \
    tests/test_runtime_limits.py \
    tests/test_rollout_acceptance.py
fi
