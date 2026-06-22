#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

use_docker="${SORIDORMI_TASK_AGENT_USE_DOCKER:-${SORIDORMI_M11_TASK_AGENT_USE_DOCKER:-auto}}"

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_task_agent_contract.sh [--help]

Validate Soridormi's task-agent contract surface with compile checks, manifest
checks, focused tests, and documentation guards.

Environment:
  SORIDORMI_TASK_AGENT_USE_DOCKER=auto|1|0
      auto: use Docker Compose when available, otherwise host Python
         1: require Docker Compose
         0: run on host Python
USAGE
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

run_python_gate() {
  echo "Checking Python compile targets..."
  python -m compileall -q src/soridormi_runtime/mcp src/soridormi_runtime/task_capabilities.py

  echo "Validating task capability manifest..."
  python -c 'from soridormi_runtime.task_capabilities import load_task_capability_manifest, validate_task_capability_manifest; result = validate_task_capability_manifest(load_task_capability_manifest()); assert result.ok, result.errors'

  echo "Exporting compact MCP capability manifest..."
  python -m soridormi_runtime.mcp.export_capabilities --compact >/tmp/soridormi_task_agent_manifest.json
  python -m json.tool /tmp/soridormi_task_agent_manifest.json >/dev/null

  echo "Running task-agent contract tests..."
  pytest -q \
    tests/test_task_capability_manifest_m11.py \
    tests/test_mcp_capability_manifest.py \
    tests/test_mcp_local_tools.py \
    tests/test_mcp_runtime_tools.py \
    tests/test_mcp_http_server.py \
    tests/test_task_acceptance_cases_m11.py \
    tests/test_training_cases_m11.py \
    tests/test_navigation_goal_contract_m11.py \
    tests/test_skill_manifest_m7.py
}

run_docs_gate() {
  echo "Checking task-agent docs..."
  rg -n "task_graph" \
    docs/SORIDORMI_MCP_SERVER.md \
    docs/mcp_capability_manifest.md \
    docs/mcp_dag_integration.md \
    docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md >/dev/null
  rg -n "configs/task_capabilities/open_duck_mini_v2_task_capabilities.json" \
    docs/README.md \
    docs/SORIDORMI_MCP_SERVER.md \
    docs/mcp_capability_manifest.md \
    docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md >/dev/null
  rg -n "soridormi.task_events.v1|poll_recommendation" \
    docs/SORIDORMI_MCP_SERVER.md \
    docs/mcp_capability_manifest.md \
    docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md >/dev/null
  rg -n "client_task_ref|idempotent_replay|task_timed_out" \
    docs/SORIDORMI_MCP_SERVER.md \
    docs/mcp_capability_manifest.md \
    docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md \
    docs/SORIDORMI_EXECUTION_ROADMAP.md >/dev/null
  python -c 'from pathlib import Path; paths = [Path(p) for p in ("docs/README.md", "docs/SORIDORMI_EXECUTION_ROADMAP.md", "docs/SORIDORMI_MCP_SERVER.md", "docs/mcp_capability_manifest.md", "docs/mcp_dag_integration.md", "docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md", "docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md")]; bad = [str(p) for p in paths if p.read_text(encoding="utf-8").count("```") % 2]; assert not bad, bad'
}

echo "Soridormi task-agent contract validation"
echo "========================================"

if [ "${use_docker}" != "0" ]; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if [ ! -f .env ]; then
      ./scripts/setup_env.sh >/dev/null
    fi
    echo "Running Python/test gate inside the runtime container."
    docker compose -f compose.sim.yaml run --rm runtime bash -lc '
      set -euo pipefail
      cd /app
      source /opt/venvs/runtime/bin/activate
      export PYTHONPATH=/app/src
      python -m compileall -q src/soridormi_runtime/mcp src/soridormi_runtime/task_capabilities.py
      python -c '"'"'from soridormi_runtime.task_capabilities import load_task_capability_manifest, validate_task_capability_manifest; result = validate_task_capability_manifest(load_task_capability_manifest()); assert result.ok, result.errors'"'"'
      python -m soridormi_runtime.mcp.export_capabilities --compact >/tmp/soridormi_task_agent_manifest.json
      python -m json.tool /tmp/soridormi_task_agent_manifest.json >/dev/null
      pytest -q \
        tests/test_task_capability_manifest_m11.py \
        tests/test_mcp_capability_manifest.py \
        tests/test_mcp_local_tools.py \
        tests/test_mcp_runtime_tools.py \
        tests/test_mcp_http_server.py \
        tests/test_task_acceptance_cases_m11.py \
        tests/test_training_cases_m11.py \
        tests/test_navigation_goal_contract_m11.py \
        tests/test_skill_manifest_m7.py
    '
  elif [ "${use_docker}" = "1" ]; then
    echo "ERROR: SORIDORMI_TASK_AGENT_USE_DOCKER=1 was requested, but Docker Compose is not available." >&2
    exit 2
  else
    echo "Docker Compose not available; falling back to host Python."
    export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
    run_python_gate
  fi
else
  echo "Running Python/test gate on host."
  export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
  run_python_gate
fi

run_docs_gate

echo "Task-agent contract validation: PASS"
