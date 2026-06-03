#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_scripted_social_skill_in_sim.sh SKILL [options]

Execute a safe scripted head/neck social skill against an already-running MuJoCo
simulator from the runtime Docker container. Hardware execution is intentionally unavailable.

Options:
  --args JSON          Skill parameter JSON object.
  --backend mujoco     Required sim backend selector (default: mujoco).
  --host HOST          Simulator API host (default: 127.0.0.1).
  --port PORT          Simulator API port (default: 5555).
  --control-hz HZ      Scripted control frequency (default: 50).
  --transition-fraction N
                       Fraction of each segment spent ramping before holding target pose (default: 0.40).
  --max-head-velocity-radps N
                       Planned head target speed limit in rad/s (default: 0.35; 0 disables).
  --no-auto-stretch-duration
                       Do not extend too-short gestures to satisfy the speed limit.
  --fall-height-m N    Base-height fall threshold for live telemetry (default: 0.14).
  --kp KP              Position gain for scripted commands (default: 10).
  --kd KD              Velocity damping for scripted commands (default: 0.35).
  --dry-run            Validate and print without connecting to MuJoCo.
  --json               Print machine-readable JSON.
  -h, --help           Show this help.

Start MuJoCo first:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

Examples:
  ./scripts/run_scripted_social_skill_in_sim.sh look_direction \
    --args '{"head_yaw_rad":0.25,"head_pitch_rad":-0.08,"duration_s":1.6}' \
    --backend mujoco

  ./scripts/run_scripted_social_skill_in_sim.sh nod_yes \
    --args '{"count":2,"amplitude":"small","duration_s":4.0}' \
    --backend mujoco

  ./scripts/run_scripted_social_skill_in_sim.sh neutral_head \
    --args '{"duration_s":3.0}' \
    --backend mujoco

  ./scripts/run_scripted_social_skill_in_sim.sh shake_no \
    --args '{"count":2,"amplitude":"small","duration_s":4.0}' \
    --backend mujoco

  ./scripts/run_scripted_social_skill_in_sim.sh bow \
    --args '{"depth":"small","duration_s":5.0}' \
    --backend mujoco

This wrapper intentionally runs inside compose.sim.yaml service runtime so host
Python packages such as pyzmq are not required.
USAGE
}

skill="${1:-}"
if [ -z "${skill}" ] || [ "${skill}" = "-h" ] || [ "${skill}" = "--help" ]; then
  usage
  exit 0
fi
shift || true

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.scripted_head_skill "$@"
' _ "${skill}" "$@"
