#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_look_at_person_target.sh [target options] [execution options]

Resolve a structured person target into yaw/pitch, then run the existing safe
look_at_person scripted head trajectory from the runtime Docker container.
This is not camera/person detection; it is the target-provider boundary that
future perception or Chromie can call.

Target sources (provide exactly one):
  --target-json JSON_OR_PATH
                     JSON object/path with target_yaw_rad/target_pitch_rad or image_x_norm/image_y_norm.
  --target-yaw-rad R --target-pitch-rad R
                     Manual structured target offsets in radians.
  --image-x-norm X --image-y-norm Y
                     Stub image target center in [0,1], converted using camera FOV.

Target options:
  --target-ref REF    Target label (default: person).
  --confidence C      Structured target confidence in [0,1] (default: 1.0).
  --horizontal-fov-rad R
                     Horizontal FOV for image-point stub (default: 60 deg).
  --vertical-fov-rad R
                     Vertical FOV for image-point stub (default: 45 deg).
  --duration-s S      look_at_person requested duration (default: 4.0).
  --hold-fraction F   Fraction of duration holding target (default: 0.5).
  --end-mode MODE     hold_target (default) keeps gaze on the person; return_neutral returns straight.

Execution options:
  --backend mujoco    Required sim backend selector (default: mujoco).
  --host HOST         Simulator API host (default: 127.0.0.1).
  --port PORT         Simulator API port (default: 5555).
  --control-hz HZ     Scripted control frequency (default: 50).
  --transition-fraction N
                     Fraction of each segment spent ramping before hold (default: 0.40).
  --max-head-velocity-radps N
                     Planned head speed limit in rad/s (default: 0.35; 0 disables).
  --no-auto-stretch-duration
                     Do not extend too-short target motions.
  --fall-height-m N   Base-height fall threshold for live telemetry (default: 0.14).
  --kp KP             Position gain for scripted commands (default: 10).
  --kd KD             Velocity damping for scripted commands (default: 0.35).
  --dry-run           Resolve/validate without connecting to MuJoCo.
  --resolve-only      Only print target + generated look_at_person args.
  --json              Print machine-readable JSON.
  -h, --help          Show this help.

Start MuJoCo first for live execution:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

Examples:
  ./scripts/run_look_at_person_target.sh \
    --target-yaw-rad 0.30 --target-pitch-rad -0.06 \
    --duration-s 4.0 --backend mujoco --dry-run

  ./scripts/run_look_at_person_target.sh \
    --image-x-norm 0.75 --image-y-norm 0.45 \
    --duration-s 4.0 --backend mujoco
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

docker compose -f compose.sim.yaml run --rm runtime bash -lc '
  set -euo pipefail
  source /opt/venvs/runtime/bin/activate
  python -m soridormi_runtime.look_at_person_target "$@"
' _ "$@"
