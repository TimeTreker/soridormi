#!/usr/bin/env bash
# Scoped X11 authorization helpers for Soridormi's Docker-hosted MuJoCo viewer.
#
# The caller owns the lifecycle: call soridormi_x11_acquire before starting the
# simulator and soridormi_x11_cleanup from an EXIT/INT/TERM trap. The helper
# prefers a temporary MIT-MAGIC-COOKIE file and falls back to one narrowly scoped
# local-user xhost rule. It never grants broad local/docker access.

SORIDORMI_X11_TEMP_DIR=""
SORIDORMI_X11_AUTH_FILE=""
SORIDORMI_X11_XHOST_USER=""
SORIDORMI_X11_XHOST_RULE_ADDED=0
SORIDORMI_X11_ACQUIRED=0
SORIDORMI_X11_DOCKER_ARGS=()

_soridormi_x11_true() {
  case "${1,,}" in
    1|true|yes|on|y) return 0 ;;
    *) return 1 ;;
  esac
}

_soridormi_x11_authority_candidates() {
  local candidate
  for candidate in \
    "${XAUTHORITY:-}" \
    "${XDG_RUNTIME_DIR:-}/gdm/Xauthority" \
    "/run/user/${UID:-$(id -u)}/gdm/Xauthority" \
    "${HOME:-}/.Xauthority"; do
    [ -n "$candidate" ] || continue
    [ -r "$candidate" ] || continue
    printf '%s\n' "$candidate"
  done | awk '!seen[$0]++'
}

_soridormi_x11_cookie_entries() {
  command -v xauth >/dev/null 2>&1 || return 1

  local entries=""
  # First respect the caller's normal xauth resolution. This also covers
  # sessions where XAUTHORITY is intentionally unset and xauth uses ~/.Xauthority.
  entries="$(xauth nlist "${DISPLAY}" 2>/dev/null || true)"
  if [ -n "$entries" ]; then
    printf '%s\n' "$entries"
    return 0
  fi

  local authority
  while IFS= read -r authority; do
    entries="$(XAUTHORITY="$authority" xauth nlist "${DISPLAY}" 2>/dev/null || true)"
    if [ -n "$entries" ]; then
      printf '%s\n' "$entries"
      return 0
    fi
  done < <(_soridormi_x11_authority_candidates)

  return 1
}

_soridormi_x11_xhost_usable() {
  command -v xhost >/dev/null 2>&1 || return 1
  xhost >/dev/null 2>&1
}

soridormi_x11_preflight() {
  local viewer_enabled="${1:-1}"
  _soridormi_x11_true "$viewer_enabled" || return 0

  if [ -z "${DISPLAY:-}" ]; then
    echo "[soridormi][error] DISPLAY is not set. Use --no-viewer." >&2
    return 1
  fi

  if _soridormi_x11_cookie_entries >/dev/null 2>&1; then
    return 0
  fi

  if _soridormi_x11_xhost_usable; then
    return 0
  fi

  echo "[soridormi][error] MuJoCo viewer requires usable X11 authorization." >&2
  echo "[soridormi][hint] Install xauth (preferred) or x11-xserver-utils, or use --no-viewer." >&2
  return 1
}

_soridormi_x11_prepare_cookie() {
  local entries
  entries="$(_soridormi_x11_cookie_entries)" || return 1
  [ -n "$entries" ] || return 1

  local runtime_parent="${XDG_RUNTIME_DIR:-/tmp}"
  SORIDORMI_X11_TEMP_DIR="$(mktemp -d "${runtime_parent%/}/soridormi-x11-${UID:-$(id -u)}-XXXXXX")"
  chmod 700 "$SORIDORMI_X11_TEMP_DIR"
  SORIDORMI_X11_AUTH_FILE="$SORIDORMI_X11_TEMP_DIR/Xauthority"
  : > "$SORIDORMI_X11_AUTH_FILE"
  chmod 600 "$SORIDORMI_X11_AUTH_FILE"

  # FamilyWild makes the copied cookie valid inside a container whose hostname
  # differs from the desktop host. This is the standard xauth container bridge.
  if ! printf '%s\n' "$entries" \
      | sed -e 's/^..../ffff/' \
      | xauth -f "$SORIDORMI_X11_AUTH_FILE" nmerge - >/dev/null 2>&1; then
    rm -rf "$SORIDORMI_X11_TEMP_DIR"
    SORIDORMI_X11_TEMP_DIR=""
    SORIDORMI_X11_AUTH_FILE=""
    return 1
  fi

  if ! xauth -f "$SORIDORMI_X11_AUTH_FILE" nlist 2>/dev/null | grep -q .; then
    rm -rf "$SORIDORMI_X11_TEMP_DIR"
    SORIDORMI_X11_TEMP_DIR=""
    SORIDORMI_X11_AUTH_FILE=""
    return 1
  fi

  SORIDORMI_X11_DOCKER_ARGS=(
    -e "DISPLAY=${DISPLAY}"
    -e "XAUTHORITY=/tmp/soridormi.Xauthority"
    -v "${SORIDORMI_X11_AUTH_FILE}:/tmp/soridormi.Xauthority:ro"
  )
  echo "[soridormi] Prepared scoped Xauthority for the MuJoCo viewer."
  return 0
}

_soridormi_x11_prepare_xhost_fallback() {
  _soridormi_x11_xhost_usable || return 1

  SORIDORMI_X11_XHOST_USER="${SORIDORMI_X11_LOCAL_USER:-$(id -un)}"
  local rule="SI:localuser:${SORIDORMI_X11_XHOST_USER}"
  local listing
  listing="$(xhost 2>/dev/null)" || return 1

  if ! sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$listing" \
      | grep -Fqx "$rule"; then
    echo "[soridormi] Granting scoped X11 access to local user ${SORIDORMI_X11_XHOST_USER}."
    if ! xhost "+${rule}" >/dev/null 2>&1; then
      echo "[soridormi][error] Failed to grant scoped X11 access for ${rule}." >&2
      return 1
    fi
    SORIDORMI_X11_XHOST_RULE_ADDED=1
  else
    echo "[soridormi] Reusing existing scoped X11 access for ${rule}."
  fi

  SORIDORMI_X11_DOCKER_ARGS=(-e "DISPLAY=${DISPLAY}")
  return 0
}

soridormi_x11_acquire() {
  local viewer_enabled="${1:-1}"
  SORIDORMI_X11_DOCKER_ARGS=()
  _soridormi_x11_true "$viewer_enabled" || return 0

  soridormi_x11_preflight "$viewer_enabled" || return 1

  if _soridormi_x11_prepare_cookie; then
    SORIDORMI_X11_ACQUIRED=1
    return 0
  fi

  if _soridormi_x11_prepare_xhost_fallback; then
    SORIDORMI_X11_ACQUIRED=1
    return 0
  fi

  echo "[soridormi][error] Could not establish scoped X11 access for DISPLAY=${DISPLAY}." >&2
  return 1
}

soridormi_x11_cleanup() {
  local rc="${1:-0}"

  if [ "$SORIDORMI_X11_XHOST_RULE_ADDED" = "1" ] \
      && [ -n "$SORIDORMI_X11_XHOST_USER" ] \
      && command -v xhost >/dev/null 2>&1; then
    local rule="SI:localuser:${SORIDORMI_X11_XHOST_USER}"
    echo "[soridormi] Removing scoped X11 access added by this launcher."
    xhost "-${rule}" >/dev/null 2>&1 || \
      echo "[soridormi][warn] Could not remove scoped X11 rule ${rule}." >&2
  fi

  if [ -n "$SORIDORMI_X11_TEMP_DIR" ] && [ -d "$SORIDORMI_X11_TEMP_DIR" ]; then
    rm -rf "$SORIDORMI_X11_TEMP_DIR"
  fi

  SORIDORMI_X11_TEMP_DIR=""
  SORIDORMI_X11_AUTH_FILE=""
  SORIDORMI_X11_XHOST_USER=""
  SORIDORMI_X11_XHOST_RULE_ADDED=0
  SORIDORMI_X11_ACQUIRED=0
  SORIDORMI_X11_DOCKER_ARGS=()
  return "$rc"
}
