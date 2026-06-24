#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SORIDORMI_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${CHROMIE_REPO:-}" ]; then
  DEFAULT_CHROMIE_REPO="$CHROMIE_REPO"
elif [ -x "$SORIDORMI_REPO/../chromie/scripts/start_voice_mujoco.sh" ]; then
  DEFAULT_CHROMIE_REPO="$SORIDORMI_REPO/../chromie"
else
  DEFAULT_CHROMIE_REPO="/home/chromie/github/chromie"
fi

CHROMIE_REPO="$DEFAULT_CHROMIE_REPO"
PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-open_duck_forward}"
MCP_PORT="${SORIDORMI_MCP_PORT:-8000}"
VIEWER=1
FOLLOW_CAMERA=1
BUILD_IMAGES=0
REBUILD_NO_CACHE=0
KEEP_RUNNING=0
AUTO_CONFIRM=1
PREFLIGHT_ONLY=0
AUDIO_SMOKE=0
AUDIO_SMOKE_SECONDS="${AUDIO_SMOKE_SECONDS:-3}"

usage() {
  cat <<'USAGE'
Usage: ./scripts/check_chromie_soridormi_effect.sh [options]

Start the paired operator check:
  microphone -> Chromie ASR/Router/Agent -> Soridormi MCP -> MuJoCo viewer
  speaker <- Chromie TTS

This is a MuJoCo/simulator check. It does not send hardware actuator commands.

Options:
  --chromie-repo DIR     Chromie checkout; default: ../chromie, then /home/chromie/github/chromie
  --profile NAME         Soridormi policy profile; default: open_duck_forward
  --mcp-port PORT        Host Soridormi MCP port; default: 8000
  --viewer               Open MuJoCo viewer; default
  --no-viewer            Run MuJoCo headless
  --follow-camera        Keep the viewer centered on the robot; default
  --no-follow-camera     Disable viewer follow camera
  --audio-smoke          Record and play back a short real mic/speaker sample before launch
  --audio-smoke-seconds N
                         Duration for --audio-smoke; default: 3
  --preflight-only       Check repositories, display, Docker, and audio devices, then exit
  --build                Build repository-owned images before startup
  --rebuild-no-cache     Rebuild Chromie images without cache; implies --build
  --require-confirmation Require spoken confirmation for simulator skills
  --auto-confirm         Use simulator confirmation exemptions; default
  --keep-running         Leave containers/simulator running after launcher exits
  -h, --help             Show this help
USAGE
}

log() {
  printf '[effect-check] %s\n' "$*"
}

warn() {
  printf '[effect-check][warn] %s\n' "$*" >&2
}

die() {
  printf '[effect-check][error] %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --chromie-repo) CHROMIE_REPO="${2:?--chromie-repo requires a directory}"; shift 2 ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --mcp-port) MCP_PORT="${2:?--mcp-port requires a value}"; shift 2 ;;
    --viewer) VIEWER=1; shift ;;
    --no-viewer) VIEWER=0; shift ;;
    --follow-camera) FOLLOW_CAMERA=1; shift ;;
    --no-follow-camera) FOLLOW_CAMERA=0; shift ;;
    --audio-smoke) AUDIO_SMOKE=1; shift ;;
    --audio-smoke-seconds) AUDIO_SMOKE_SECONDS="${2:?--audio-smoke-seconds requires a value}"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --build) BUILD_IMAGES=1; shift ;;
    --rebuild-no-cache) BUILD_IMAGES=1; REBUILD_NO_CACHE=1; shift ;;
    --require-confirmation) AUTO_CONFIRM=0; shift ;;
    --auto-confirm) AUTO_CONFIRM=1; shift ;;
    --keep-running) KEEP_RUNNING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

case "$AUDIO_SMOKE_SECONDS" in
  ''|*[!0-9]*) die "--audio-smoke-seconds must be a positive integer" ;;
esac
if [ "$AUDIO_SMOKE_SECONDS" -le 0 ]; then
  die "--audio-smoke-seconds must be greater than zero"
fi

if [ -d "$CHROMIE_REPO" ]; then
  CHROMIE_REPO="$(cd "$CHROMIE_REPO" && pwd)"
else
  die "Chromie repo not found: $CHROMIE_REPO"
fi

for cmd in docker python3; do
  command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
done

for path in \
  "$SORIDORMI_REPO/scripts/start_soridormi_mujoco.sh" \
  "$CHROMIE_REPO/scripts/start_voice_mujoco.sh" \
  "$CHROMIE_REPO/scripts/status_voice_mujoco.sh" \
  "$CHROMIE_REPO/scripts/check_voice_mujoco_logs.sh"; do
  [ -e "$path" ] || die "Missing required file: $path"
done

if ! docker info >/dev/null 2>&1; then
  die "Docker daemon is not reachable"
fi

if [ "$VIEWER" = "1" ] && [ -z "${DISPLAY:-}" ]; then
  die "DISPLAY is not set. Run from a graphical session or pass --no-viewer."
fi

print_audio_status() {
  log "Host audio device summary:"
  if command -v wpctl >/dev/null 2>&1; then
    wpctl status || warn "wpctl status failed"
  elif command -v pactl >/dev/null 2>&1; then
    pactl info || warn "pactl info failed"
    pactl get-default-source 2>/dev/null || true
    pactl get-default-sink 2>/dev/null || true
  else
    warn "Neither wpctl nor pactl is available; cannot list host microphone/speaker devices."
  fi

  local env_file="$CHROMIE_REPO/orchestrator/.env.local"
  if [ -f "$env_file" ]; then
    log "Chromie orchestrator audio device overrides from $env_file:"
    grep -E '^(ORCH_INPUT_DEVICE|ORCH_OUTPUT_DEVICE)=' "$env_file" || \
      warn "No ORCH_INPUT_DEVICE/ORCH_OUTPUT_DEVICE override found in $env_file"
  else
    warn "Chromie orchestrator/.env.local does not exist yet; start_chromie.sh will create it from the example."
  fi
}

run_audio_smoke() {
  local tmp="${TMPDIR:-/tmp}/chromie_soridormi_audio_smoke.wav"
  log "Recording ${AUDIO_SMOKE_SECONDS}s from the configured/default microphone."
  log "Speak now; the sample will be played back through the configured/default speaker."

  if command -v arecord >/dev/null 2>&1 && command -v aplay >/dev/null 2>&1; then
    arecord -q -d "$AUDIO_SMOKE_SECONDS" -f cd "$tmp"
    log "Playing microphone sample through speaker."
    aplay -q "$tmp"
    return 0
  fi

  if command -v parec >/dev/null 2>&1 && command -v paplay >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    timeout "$AUDIO_SMOKE_SECONDS" parec --file-format=wav "$tmp" >/dev/null 2>&1 || true
    [ -s "$tmp" ] || die "Microphone smoke recording produced no audio file: $tmp"
    log "Playing microphone sample through speaker."
    paplay "$tmp"
    return 0
  fi

  if command -v pw-record >/dev/null 2>&1 && command -v pw-play >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    timeout "$AUDIO_SMOKE_SECONDS" pw-record "$tmp" >/dev/null 2>&1 || true
    [ -s "$tmp" ] || die "Microphone smoke recording produced no audio file: $tmp"
    log "Playing microphone sample through speaker."
    pw-play "$tmp"
    return 0
  fi

  die "No supported audio smoke tools found. Install/use arecord+aplay, parec+paplay, or pw-record+pw-play."
}

print_audio_status
if [ "$AUDIO_SMOKE" = "1" ]; then
  run_audio_smoke
fi

cat <<EOF_READY

======================================================================
Chromie + Soridormi effect check
======================================================================
Soridormi repo: ${SORIDORMI_REPO}
Chromie repo:   ${CHROMIE_REPO}
Profile:        ${PROFILE}
MCP:            http://127.0.0.1:${MCP_PORT}/mcp
Viewer:         ${VIEWER}
Follow camera:  ${FOLLOW_CAMERA}
Real mic/speaker path: enabled by Chromie host Orchestrator
Hardware robot: not used

When ready, speak into the configured microphone, for example:
  Hello Chromie.
  What is the robot status?
  Please nod twice.
  Look at me for three seconds.
  Stop.

Watch the MuJoCo viewer for simulator-bounded body motion and listen for
Chromie's speaker response.
======================================================================
EOF_READY

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  log "Preflight complete."
  exit 0
fi

chromie_args=(--soridormi-repo "$SORIDORMI_REPO" --profile "$PROFILE" --mcp-port "$MCP_PORT")
if [ "$VIEWER" = "1" ]; then chromie_args+=(--viewer); else chromie_args+=(--no-viewer); fi
if [ "$FOLLOW_CAMERA" = "1" ]; then
  chromie_args+=(--follow-camera)
else
  chromie_args+=(--no-follow-camera)
fi
if [ "$BUILD_IMAGES" = "1" ]; then chromie_args+=(--build); fi
if [ "$REBUILD_NO_CACHE" = "1" ]; then chromie_args+=(--rebuild-no-cache); fi
if [ "$KEEP_RUNNING" = "1" ]; then chromie_args+=(--keep-running); fi
if [ "$AUTO_CONFIRM" = "1" ]; then
  chromie_args+=(--auto-confirm)
else
  chromie_args+=(--require-confirmation)
fi

log "Delegating to Chromie's maintained voice-to-MuJoCo launcher."
log "Status from another terminal: (cd '$CHROMIE_REPO' && ./scripts/status_voice_mujoco.sh)"
exec "$CHROMIE_REPO/scripts/start_voice_mujoco.sh" "${chromie_args[@]}"
