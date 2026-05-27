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

if [ "${SORIDORMI_CI_SKIP_PYTEST:-0}" != "1" ]; then
  echo "Running M5 unit tests..."
  pytest -q \
    tests/test_check_policy_model_m42.py \
    tests/test_policy_contract_m5.py \
    tests/test_policy_model_contract_gate_m52.py \
    tests/test_onnx_providers_m53.py \
    tests/test_create_policy_profile_m54.py \
    tests/test_validate_policy_profiles_m55.py \
    tests/test_ci_static_check_m56.py \
    tests/test_policy_manifest_m57.py     tests/test_policy_acceptance_m58.py     tests/test_policy_package_m59.py
fi
