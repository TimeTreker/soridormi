#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Soridormi body-concurrency validation"
echo "===================================="

echo "Checking repository governance..."
python scripts/validate_repository_governance.py

echo "Checking Python modules..."
python -m compileall -q \
  src/soridormi_runtime/mcp \
  src/soridormi_runtime/skill_execution.py \
  src/soridormi_runtime/skill_manifest.py \
  tests/test_body_activity_concurrency.py

echo "Validating skill and task manifests..."
python - <<'PY'
from soridormi_runtime.skill_manifest import load_skill_manifest, validate_skill_manifest
from soridormi_runtime.task_capabilities import load_task_capability_manifest, validate_task_capability_manifest

skill_result = validate_skill_manifest(load_skill_manifest())
assert skill_result.ok, skill_result.errors

task_result = validate_task_capability_manifest(load_task_capability_manifest())
assert task_result.ok, task_result.errors
PY

echo "Exporting MCP capabilities..."
python -m soridormi_runtime.mcp.export_capabilities --compact \
  >/tmp/soridormi_body_concurrency_capabilities.json
python -m json.tool /tmp/soridormi_body_concurrency_capabilities.json >/dev/null

echo "Running focused tests..."
pytest -q \
  tests/test_body_activity_concurrency.py \
  tests/test_skill_manifest.py \
  tests/test_skill_execution.py \
  tests/test_mcp_capability_manifest.py \
  tests/test_mcp_local_tools.py \
  tests/test_mcp_runtime_tools.py \
  tests/test_mcp_http_server.py \
  tests/test_task_semantic_integrity.py

echo "Body-concurrency validation: PASS"
