#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/x11_access.sh

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_scenario.sh [--scenario FILE] [--profile PROFILE] [--viewer|--no-viewer] [--validate]

Read a simulation scenario, build its world with MuJoCo MjSpec, start the
Soridormi simulator as a child process, and drive scenario events on simulation
time. The default scenario places a milk bottle on the table 10 m ahead.
If an earlier scenario from this checkout owns SIM_PORT, replace it before
starting. Stop the current run with Ctrl-C or docker stop CONTAINER_NAME.

The scenario file must be accessible on this host. --validate builds and
compiles the scene without starting Soridormi or sending robot commands.
--profile applies the same simulator compatibility settings as run_sim_server.sh.
USAGE
}

SCENARIO_PATH="$PWD/configs/simulation_scenarios/default.json"
VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"
SIM_POLICY_PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-}"
VALIDATE="0"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --scenario)
      SCENARIO_PATH="$(realpath "${2:?--scenario requires a file}")"
      shift 2
      ;;
    --viewer) VIEWER_ENABLED="1"; shift ;;
    --no-viewer) VIEWER_ENABLED="0"; shift ;;
    --profile) SIM_POLICY_PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --validate) VALIDATE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -f "$SCENARIO_PATH" ]; then
  echo "Scenario file not found: $SCENARIO_PATH" >&2
  exit 2
fi
if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

cleanup() {
  local rc=$?
  soridormi_x11_cleanup "$rc" || true
  return "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
soridormi_x11_acquire "$VIEWER_ENABLED"

sim_port="${SIM_PORT:-5555}"
tcp_port_active() {
  python3 - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.socket() as probe:
    probe.settimeout(0.5)
    sys.exit(0 if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

check_runtime_idle() {
  local runtime_port
  if [ "$(docker inspect -f '{{.State.Running}}' soridormi-runtime-mcp 2>/dev/null || true)" != "true" ]; then
    return 0
  fi
  runtime_port="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' soridormi-runtime-mcp \
    | sed -n 's/^SIM_PORT=//p' | head -n 1)"
  if [ "$runtime_port" != "$sim_port" ]; then
    return 0
  fi
  docker exec -i soridormi-runtime-mcp /opt/venvs/runtime/bin/python - <<'PY'
import asyncio
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def check():
    port = os.environ.get("SORIDORMI_MCP_PORT", "8000")
    path = os.environ.get("SORIDORMI_MCP_PATH", "/mcp")
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        async with streamable_http_client(f"http://127.0.0.1:{port}{path}", http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("soridormi.robot.get_status", {})
                status = result.structuredContent or {}
                if result.isError or status.get("safe_idle") is not True or status.get("active_task") is not None:
                    raise RuntimeError("Soridormi has active work or is not safely idle")

try:
    asyncio.run(check())
except Exception as error:
    print(f"[soridormi][error] Cannot restart scenario: {error}", file=sys.stderr)
    sys.exit(1)
PY
}

if [ "$VALIDATE" != "1" ] && { tcp_port_active "$sim_port" || tcp_port_active "$((sim_port + 1))"; }; then
  matching_containers=()
  while IFS= read -r container; do
    [ -n "$container" ] || continue
    command_json="$(docker inspect -f '{{json .Config.Cmd}}' "$container")"
    [[ "$command_json" == *'python -m scenario_runner'* ]] || continue
    container_env="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container")"
    [[ $'\n'"$container_env"$'\n' == *$'\n'"SIM_PORT=$sim_port"$'\n'* ]] || continue
    matching_containers+=("$container")
  done < <(docker ps -q \
    --filter "label=com.docker.compose.project.working_dir=$PWD" \
    --filter 'label=com.docker.compose.service=sim')
  if [ "${#matching_containers[@]}" -ne 1 ]; then
    echo "[soridormi][error] Simulator port $sim_port is busy; expected one scenario container from this checkout, found ${#matching_containers[@]}." >&2
    echo "[soridormi][hint] Inspect the port owner and stop it explicitly, or use another SIM_PORT." >&2
    exit 1
  fi
  check_runtime_idle
  echo "[soridormi] Replacing prior scenario container ${matching_containers[0]} on port $sim_port."
  docker stop --timeout 10 "${matching_containers[0]}" >/dev/null
  docker rm "${matching_containers[0]}" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! tcp_port_active "$sim_port" && ! tcp_port_active "$((sim_port + 1))"; then
      break
    fi
    sleep 0.2
  done
  if tcp_port_active "$sim_port" || tcp_port_active "$((sim_port + 1))"; then
    echo "[soridormi][error] Simulator port $sim_port is still busy after stopping the prior scenario." >&2
    exit 1
  fi
fi

RUNNER_ARGS=(--scenario /scenario/scenario.json)
if [ "$VALIDATE" = "1" ]; then
  RUNNER_ARGS+=(--validate)
fi

docker compose -f compose.sim.yaml run --rm \
  "${SORIDORMI_X11_DOCKER_ARGS[@]}" \
  -v "$SCENARIO_PATH:/scenario/scenario.json:ro" \
  -e SORIDORMI_MUJOCO_VIEWER="$VIEWER_ENABLED" \
  -e SORIDORMI_SIM_POLICY_PROFILE="$SIM_POLICY_PROFILE" \
  -e SIM_PORT="$sim_port" \
  sim bash -lc '
    source /opt/venvs/sim/bin/activate
    if [ -n "${SORIDORMI_SIM_POLICY_PROFILE:-}" ]; then
      eval "$(python -m soridormi_runtime.policy_profiles "${SORIDORMI_SIM_POLICY_PROFILE}" --shell)"
    fi
    python -m scenario_runner "$@"
  ' \
  scenario-runner "${RUNNER_ARGS[@]}"
