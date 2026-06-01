#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_sim_server.sh [options]

Start the Soridormi simulator server. The normal locomotion/sim validation
backend is MuJoCo. The MuJoCo viewer is off by default; enable it explicitly
when you want a visual functional test.

Options:
  --backend NAME        Simulator backend to start. default: mujoco
                        Common values: mujoco, fake
  --profile PROFILE     Resolve simulator compatibility settings from a policy
                        profile before starting MuJoCo. Use this for walking
                        parity tests, for example open_duck_forward.
  --viewer              Enable the passive MuJoCo viewer.
  --no-viewer           Disable the passive MuJoCo viewer. default
  --follow-camera       Keep the MuJoCo viewer centered on the robot base.
  --no-follow-camera    Disable viewer follow camera. default
  --camera-distance N   Follow-camera distance. default: 1.4
  --camera-azimuth DEG  Follow-camera azimuth. default: 135
  --camera-elevation DEG
                        Follow-camera elevation. default: -20
  -h, --help            Show this help.

Examples:
  ./scripts/run_sim_server.sh --backend mujoco --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

For policy parity/teacher rollout tests, start the simulator with the same
policy profile used by the runtime so MuJoCo receives the profile's official
Open Duck compatibility flags:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
USAGE
}

SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"
VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"
FOLLOW_CAMERA="${SORIDORMI_MUJOCO_FOLLOW_CAMERA:-0}"
CAMERA_DISTANCE="${SORIDORMI_MUJOCO_CAMERA_DISTANCE:-1.4}"
CAMERA_AZIMUTH="${SORIDORMI_MUJOCO_CAMERA_AZIMUTH:-135}"
CAMERA_ELEVATION="${SORIDORMI_MUJOCO_CAMERA_ELEVATION:--20}"
SIM_POLICY_PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend)
      SIM_BACKEND="${2:?--backend requires a value}"
      shift 2
      ;;
    --profile)
      SIM_POLICY_PROFILE="${2:?--profile requires a value}"
      shift 2
      ;;
    --viewer)
      VIEWER_ENABLED="1"
      shift
      ;;
    --no-viewer)
      VIEWER_ENABLED="0"
      shift
      ;;
    --follow-camera)
      FOLLOW_CAMERA="1"
      shift
      ;;
    --no-follow-camera)
      FOLLOW_CAMERA="0"
      shift
      ;;
    --camera-distance)
      CAMERA_DISTANCE="${2:?--camera-distance requires a value}"
      shift 2
      ;;
    --camera-azimuth)
      CAMERA_AZIMUTH="${2:?--camera-azimuth requires a value}"
      shift 2
      ;;
    --camera-elevation)
      CAMERA_ELEVATION="${2:?--camera-elevation requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f .env ]; then
  ./scripts/setup_env.sh
fi

AUTO_XHOST="${SORIDORMI_XHOST_AUTO:-1}"
XHOST_ADDED=0

is_true() {
  case "${1,,}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

enable_xhost_if_needed() {
  if ! is_true "$AUTO_XHOST"; then
    return 0
  fi

  if ! is_true "$VIEWER_ENABLED"; then
    return 0
  fi

  if [ -z "${DISPLAY:-}" ]; then
    echo "Warning: DISPLAY is not set. MuJoCo viewer may not open."
    return 0
  fi

  if ! command -v xhost >/dev/null 2>&1; then
    echo "Warning: xhost not found. Install it with: sudo apt install x11-xserver-utils"
    return 0
  fi

  echo "Allowing local Docker containers to access X11..."
  xhost +local:docker >/dev/null || true
  XHOST_ADDED=1
}

cleanup_xhost() {
  if [ "$XHOST_ADDED" = "1" ]; then
    echo "Removing local Docker X11 access..."
    xhost -local:docker >/dev/null 2>&1 || true
  fi
}

trap cleanup_xhost EXIT INT TERM

enable_xhost_if_needed

export SORIDORMI_SIM_BACKEND="${SIM_BACKEND}"
export SORIDORMI_MUJOCO_VIEWER="${VIEWER_ENABLED}"
export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${FOLLOW_CAMERA}"
export SORIDORMI_MUJOCO_CAMERA_DISTANCE="${CAMERA_DISTANCE}"
export SORIDORMI_MUJOCO_CAMERA_AZIMUTH="${CAMERA_AZIMUTH}"
export SORIDORMI_MUJOCO_CAMERA_ELEVATION="${CAMERA_ELEVATION}"
export SORIDORMI_SIM_POLICY_PROFILE="${SIM_POLICY_PROFILE}"

echo "Soridormi simulator server"
echo "=========================="
echo "Backend: ${SORIDORMI_SIM_BACKEND}"
echo "MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"
echo "MuJoCo follow camera: ${SORIDORMI_MUJOCO_FOLLOW_CAMERA}"
if [ "${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" = "1" ]; then
  echo "MuJoCo follow camera params: distance=${SORIDORMI_MUJOCO_CAMERA_DISTANCE} azimuth=${SORIDORMI_MUJOCO_CAMERA_AZIMUTH} elevation=${SORIDORMI_MUJOCO_CAMERA_ELEVATION}"
fi
if [ -n "${SORIDORMI_SIM_POLICY_PROFILE}" ]; then
  echo "Policy profile compatibility: ${SORIDORMI_SIM_POLICY_PROFILE}"
else
  echo "Policy profile compatibility: none"
fi

docker compose -f compose.sim.yaml run --rm \
  -e SORIDORMI_SIM_POLICY_PROFILE="${SORIDORMI_SIM_POLICY_PROFILE}" \
  -e SORIDORMI_SIM_BACKEND_OVERRIDE="${SORIDORMI_SIM_BACKEND}" \
  -e SORIDORMI_MUJOCO_VIEWER_OVERRIDE="${SORIDORMI_MUJOCO_VIEWER}" \
  -e SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE="${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" \
  -e SORIDORMI_MUJOCO_CAMERA_DISTANCE_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_DISTANCE}" \
  -e SORIDORMI_MUJOCO_CAMERA_AZIMUTH_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_AZIMUTH}" \
  -e SORIDORMI_MUJOCO_CAMERA_ELEVATION_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_ELEVATION}" \
  sim bash -lc '
    set -euo pipefail
    source /opt/venvs/sim/bin/activate

    if [ -n "${SORIDORMI_SIM_POLICY_PROFILE:-}" ]; then
      echo "Resolving simulator compatibility from policy profile: ${SORIDORMI_SIM_POLICY_PROFILE}"
      eval "$(python -m soridormi_runtime.policy_profiles "${SORIDORMI_SIM_POLICY_PROFILE}" --shell)"
    fi

    # CLI/env wrapper choices win over profile defaults for these two runtime
    # server controls. The profile still supplies MuJoCo official-compatibility
    # flags such as SORIDORMI_MUJOCO_USE_HOME_KEYFRAME and sensor/contact modes.
    export SORIDORMI_SIM_BACKEND="${SORIDORMI_SIM_BACKEND_OVERRIDE:-mujoco}"
    export SORIDORMI_MUJOCO_VIEWER="${SORIDORMI_MUJOCO_VIEWER_OVERRIDE:-0}"
    export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE:-0}"
    export SORIDORMI_MUJOCO_CAMERA_DISTANCE="${SORIDORMI_MUJOCO_CAMERA_DISTANCE_OVERRIDE:-1.4}"
    export SORIDORMI_MUJOCO_CAMERA_AZIMUTH="${SORIDORMI_MUJOCO_CAMERA_AZIMUTH_OVERRIDE:-135}"
    export SORIDORMI_MUJOCO_CAMERA_ELEVATION="${SORIDORMI_MUJOCO_CAMERA_ELEVATION_OVERRIDE:--20}"

    echo "Effective sim backend: ${SORIDORMI_SIM_BACKEND}"
    echo "Effective MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"
    echo "Effective MuJoCo follow camera: ${SORIDORMI_MUJOCO_FOLLOW_CAMERA}"
    if [ "${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" = "1" ]; then
      echo "Effective MuJoCo camera params: distance=${SORIDORMI_MUJOCO_CAMERA_DISTANCE} azimuth=${SORIDORMI_MUJOCO_CAMERA_AZIMUTH} elevation=${SORIDORMI_MUJOCO_CAMERA_ELEVATION}"
    fi
    echo "Effective MuJoCo profile flags: home_keyframe=${SORIDORMI_MUJOCO_USE_HOME_KEYFRAME:-0} official_reset=${SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE:-0} official_sensor=${SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE:-0} official_contact=${SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE:-0}"

    python -m soridormi_sim.mujoco_server
  '
