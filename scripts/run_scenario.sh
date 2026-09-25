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

RUNNER_ARGS=(--scenario /scenario/scenario.json)
if [ "$VALIDATE" = "1" ]; then
  RUNNER_ARGS+=(--validate)
fi

docker compose -f compose.sim.yaml run --rm \
  "${SORIDORMI_X11_DOCKER_ARGS[@]}" \
  -v "$SCENARIO_PATH:/scenario/scenario.json:ro" \
  -e SORIDORMI_MUJOCO_VIEWER="$VIEWER_ENABLED" \
  -e SORIDORMI_SIM_POLICY_PROFILE="$SIM_POLICY_PROFILE" \
  -e SIM_PORT="${SIM_PORT:-5555}" \
  sim bash -lc '
    source /opt/venvs/sim/bin/activate
    if [ -n "${SORIDORMI_SIM_POLICY_PROFILE:-}" ]; then
      eval "$(python -m soridormi_runtime.policy_profiles "${SORIDORMI_SIM_POLICY_PROFILE}" --shell)"
    fi
    python -m scenario_runner "$@"
  ' \
  scenario-runner "${RUNNER_ARGS[@]}"
