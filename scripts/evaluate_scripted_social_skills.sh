#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/evaluate_scripted_social_skills.sh [options]

Evaluate acceptance gates for safe scripted head/neck social skills. By default
this runs dry-run trajectory checks inside the runtime Docker container. Add
--execute to run against an already-running MuJoCo simulator and check observed
joint ranges plus base-height fall telemetry.

Options:
  --skill SKILL       Skill id to evaluate; repeatable. Defaults to all scripted social gates.
  --execute           Stream commands to an already-running MuJoCo simulator.
  --backend mujoco    Required sim backend selector (default: mujoco).
  --host HOST         Simulator API host (default: 127.0.0.1).
  --port PORT         Simulator API port (default: 5555).
  --control-hz HZ     Scripted control frequency (default: 50).
  --transition-fraction N
                      Fraction of each segment spent ramping before holding target pose (default: 0.40).
  --max-head-velocity-radps N
                      Planned head target speed limit in rad/s (default: 0.35; 0 disables).
  --no-auto-stretch-duration
                      Do not extend too-short gestures to satisfy the speed limit.
  --fall-height-m N   Base-height fall threshold for live telemetry (default: 0.14).
  --require-observed  In --execute mode, fail if observed joint range is below the gate threshold.
  --json              Print machine-readable JSON.
  -h, --help          Show this help.

Start MuJoCo before live execution:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

Dry-run example:
  ./scripts/evaluate_scripted_social_skills.sh --json

Live MuJoCo example:
  ./scripts/evaluate_scripted_social_skills.sh --execute --backend mujoco --require-observed
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

# Override the CUDA image entrypoint so --json stdout can be redirected into a
# valid JSON file without the NVIDIA container banner at the top.
docker compose -f compose.sim.yaml run --rm   --entrypoint bash   runtime -lc '
    set -euo pipefail
    source /opt/venvs/runtime/bin/activate
    python -m soridormi_runtime.scripted_social_acceptance "$@"
  ' _ "$@"
