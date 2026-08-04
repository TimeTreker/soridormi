#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck source=scripts/x11_access.sh
source ./scripts/x11_access.sh

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
  --social-eyes         Generate visual-only social eyes attached to the face. default
  --no-social-eyes      Disable the generated visual eyes.
  --social-eye-frame    Show an RGB debug frame at the generated eye anchor.
  --no-social-eye-frame Hide the generated eye debug frame. default
  --rough-ground        Generate a temporary MuJoCo scene with small stone boxes.
  --no-rough-ground     Use the normal flat MuJoCo scene. default
  --rough-stone-height M
                        Approximate stone height in meters. default: 0.008
  --rough-stone-count N
                        Number of generated stones. default: 8
  --rough-stone-radius M
                        Approximate stone half-size/radius in meters. default: 0.018
  -h, --help            Show this help.

Examples:
  ./scripts/run_sim_server.sh --backend mujoco --no-viewer
  ./scripts/run_sim_server.sh --backend mujoco --viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --social-eyes
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --social-eye-frame
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --rough-ground

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
SOCIAL_EYES="${SORIDORMI_MUJOCO_SOCIAL_EYES:-1}"
SOCIAL_EYE_FRAME="${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME:-0}"
ROUGH_GROUND="${SORIDORMI_MUJOCO_ROUGH_GROUND:-0}"
ROUGH_STONE_HEIGHT="${SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT:-0.008}"
ROUGH_STONE_COUNT="${SORIDORMI_MUJOCO_ROUGH_STONE_COUNT:-8}"
ROUGH_STONE_RADIUS="${SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS:-0.018}"

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
    --social-eyes)
      SOCIAL_EYES="1"
      shift
      ;;
    --no-social-eyes)
      SOCIAL_EYES="0"
      shift
      ;;
    --social-eye-frame)
      SOCIAL_EYE_FRAME="1"
      shift
      ;;
    --no-social-eye-frame)
      SOCIAL_EYE_FRAME="0"
      shift
      ;;
    --rough-ground)
      ROUGH_GROUND="1"
      shift
      ;;
    --no-rough-ground)
      ROUGH_GROUND="0"
      shift
      ;;
    --rough-stone-height)
      ROUGH_STONE_HEIGHT="${2:?--rough-stone-height requires a value}"
      shift 2
      ;;
    --rough-stone-count)
      ROUGH_STONE_COUNT="${2:?--rough-stone-count requires a value}"
      shift 2
      ;;
    --rough-stone-radius)
      ROUGH_STONE_RADIUS="${2:?--rough-stone-radius requires a value}"
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

X11_CLEANED=0
cleanup_x11() {
  local rc=$?
  if [ "$X11_CLEANED" = "0" ]; then
    X11_CLEANED=1
    soridormi_x11_cleanup "$rc"
  fi
  return "$rc"
}
trap cleanup_x11 EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

soridormi_x11_acquire "$VIEWER_ENABLED"

export SORIDORMI_SIM_BACKEND="${SIM_BACKEND}"
export SORIDORMI_MUJOCO_VIEWER="${VIEWER_ENABLED}"
export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${FOLLOW_CAMERA}"
export SORIDORMI_MUJOCO_CAMERA_DISTANCE="${CAMERA_DISTANCE}"
export SORIDORMI_MUJOCO_CAMERA_AZIMUTH="${CAMERA_AZIMUTH}"
export SORIDORMI_MUJOCO_CAMERA_ELEVATION="${CAMERA_ELEVATION}"
export SORIDORMI_SIM_POLICY_PROFILE="${SIM_POLICY_PROFILE}"
export SORIDORMI_MUJOCO_SOCIAL_EYES="${SOCIAL_EYES}"
export SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME="${SOCIAL_EYE_FRAME}"
export SORIDORMI_MUJOCO_ROUGH_GROUND="${ROUGH_GROUND}"
export SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT="${ROUGH_STONE_HEIGHT}"
export SORIDORMI_MUJOCO_ROUGH_STONE_COUNT="${ROUGH_STONE_COUNT}"
export SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS="${ROUGH_STONE_RADIUS}"

echo "Soridormi simulator server"
echo "=========================="
echo "Backend: ${SORIDORMI_SIM_BACKEND}"
echo "MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"
echo "MuJoCo follow camera: ${SORIDORMI_MUJOCO_FOLLOW_CAMERA}"
echo "MuJoCo social eyes: ${SORIDORMI_MUJOCO_SOCIAL_EYES}"
echo "MuJoCo social eye frame: ${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME}"
echo "MuJoCo rough ground: ${SORIDORMI_MUJOCO_ROUGH_GROUND}"
if [ "${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" = "1" ]; then
  echo "MuJoCo follow camera params: distance=${SORIDORMI_MUJOCO_CAMERA_DISTANCE} azimuth=${SORIDORMI_MUJOCO_CAMERA_AZIMUTH} elevation=${SORIDORMI_MUJOCO_CAMERA_ELEVATION}"
fi
if [ -n "${SORIDORMI_SIM_POLICY_PROFILE}" ]; then
  echo "Policy profile compatibility: ${SORIDORMI_SIM_POLICY_PROFILE}"
else
  echo "Policy profile compatibility: none"
fi

docker compose -f compose.sim.yaml run --rm \
  "${SORIDORMI_X11_DOCKER_ARGS[@]}" \
  -e SORIDORMI_SIM_POLICY_PROFILE="${SORIDORMI_SIM_POLICY_PROFILE}" \
  -e SORIDORMI_SIM_BACKEND_OVERRIDE="${SORIDORMI_SIM_BACKEND}" \
  -e SORIDORMI_MUJOCO_VIEWER_OVERRIDE="${SORIDORMI_MUJOCO_VIEWER}" \
  -e SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE="${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" \
  -e SORIDORMI_MUJOCO_CAMERA_DISTANCE_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_DISTANCE}" \
  -e SORIDORMI_MUJOCO_CAMERA_AZIMUTH_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_AZIMUTH}" \
  -e SORIDORMI_MUJOCO_CAMERA_ELEVATION_OVERRIDE="${SORIDORMI_MUJOCO_CAMERA_ELEVATION}" \
  -e SORIDORMI_MUJOCO_SOCIAL_EYES_OVERRIDE="${SORIDORMI_MUJOCO_SOCIAL_EYES}" \
  -e SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME_OVERRIDE="${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME}" \
  -e SORIDORMI_MUJOCO_ROUGH_GROUND_OVERRIDE="${SORIDORMI_MUJOCO_ROUGH_GROUND}" \
  -e SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT_OVERRIDE="${SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT}" \
  -e SORIDORMI_MUJOCO_ROUGH_STONE_COUNT_OVERRIDE="${SORIDORMI_MUJOCO_ROUGH_STONE_COUNT}" \
  -e SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS_OVERRIDE="${SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS}" \
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
    export SORIDORMI_MUJOCO_SOCIAL_EYES="${SORIDORMI_MUJOCO_SOCIAL_EYES_OVERRIDE:-1}"
    export SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME="${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME_OVERRIDE:-0}"
    export SORIDORMI_MUJOCO_ROUGH_GROUND="${SORIDORMI_MUJOCO_ROUGH_GROUND_OVERRIDE:-0}"
    export SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT="${SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT_OVERRIDE:-0.008}"
    export SORIDORMI_MUJOCO_ROUGH_STONE_COUNT="${SORIDORMI_MUJOCO_ROUGH_STONE_COUNT_OVERRIDE:-8}"
    export SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS="${SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS_OVERRIDE:-0.018}"

    if [ "${SORIDORMI_SIM_BACKEND}" = "mujoco" ] && [ "${SORIDORMI_MUJOCO_SOCIAL_EYES}" = "1" ]; then
      BASE_MODEL="${MUJOCO_MODEL_PATH:-}"
      if [ -z "${BASE_MODEL}" ]; then
        BASE_MODEL="$(python - <<'PYMODEL'
from soridormi_sim.robot_config import load_robot_config
print(load_robot_config().model.path)
PYMODEL
)"
      fi
      # Keep generated scene/model overlays next to the Open Duck XMLs so MuJoCo
      # still resolves mesh assets relative to the original compiler context.
      SOCIAL_EYES_MODEL="$(dirname "${BASE_MODEL}")/soridormi_social_eyes_scene.xml"
      SOCIAL_EYE_FRAME_ARGS=()
      if [ "${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME}" = "1" ]; then
        SOCIAL_EYE_FRAME_ARGS+=(--debug-frame)
      fi
      python -m soridormi_sim.social_eye_scene         --base "${BASE_MODEL}"         --output "${SOCIAL_EYES_MODEL}"         "${SOCIAL_EYE_FRAME_ARGS[@]}"
      export MUJOCO_MODEL_PATH="${SOCIAL_EYES_MODEL}"
    fi

    if [ "${SORIDORMI_SIM_BACKEND}" = "mujoco" ] && [ "${SORIDORMI_MUJOCO_ROUGH_GROUND}" = "1" ]; then
      BASE_MODEL="${MUJOCO_MODEL_PATH:-}"
      if [ -z "${BASE_MODEL}" ]; then
        BASE_MODEL="$(python - <<'PYMODEL'
from soridormi_sim.robot_config import load_robot_config
print(load_robot_config().model.path)
PYMODEL
)"
      fi
      # Write the generated scene next to the original Open Duck XML. MuJoCo
      # resolves mesh and texture paths relative to the top-level XML/compiler
      # context, so writing the generated scene to /tmp can make included robot
      # XML files look for meshes in the wrong directory.
      ROUGH_MODEL="$(dirname "${BASE_MODEL}")/soridormi_rough_ground_scene.xml"
      python -m soridormi_sim.rough_ground_scene         --base "${BASE_MODEL}"         --output "${ROUGH_MODEL}"         --stone-count "${SORIDORMI_MUJOCO_ROUGH_STONE_COUNT}"         --stone-height "${SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT}"         --stone-radius "${SORIDORMI_MUJOCO_ROUGH_STONE_RADIUS}"
      export MUJOCO_MODEL_PATH="${ROUGH_MODEL}"
    fi

    echo "Effective sim backend: ${SORIDORMI_SIM_BACKEND}"
    echo "Effective MuJoCo viewer: ${SORIDORMI_MUJOCO_VIEWER}"
    echo "Effective MuJoCo follow camera: ${SORIDORMI_MUJOCO_FOLLOW_CAMERA}"
    echo "Effective MuJoCo social eyes: ${SORIDORMI_MUJOCO_SOCIAL_EYES}"
    echo "Effective MuJoCo social eye frame: ${SORIDORMI_MUJOCO_SOCIAL_EYE_FRAME}"
    echo "Effective MuJoCo rough ground: ${SORIDORMI_MUJOCO_ROUGH_GROUND}"
    if [ "${SORIDORMI_MUJOCO_FOLLOW_CAMERA}" = "1" ]; then
      echo "Effective MuJoCo camera params: distance=${SORIDORMI_MUJOCO_CAMERA_DISTANCE} azimuth=${SORIDORMI_MUJOCO_CAMERA_AZIMUTH} elevation=${SORIDORMI_MUJOCO_CAMERA_ELEVATION}"
    fi
    echo "Effective MuJoCo profile flags: home_keyframe=${SORIDORMI_MUJOCO_USE_HOME_KEYFRAME:-0} official_reset=${SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE:-0} official_sensor=${SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE:-0} official_contact=${SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE:-0}"

    python -m soridormi_sim.mujoco_server
  '
