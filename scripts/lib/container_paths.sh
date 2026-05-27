#!/usr/bin/env bash
# Helpers for wrapper scripts that run commands inside compose.sim.yaml.
# Host ./data is mounted in the runtime container as /data, so paths selected
# by the host shell must be rewritten before they are passed into the container.

soridormi_to_container_data_path() {
  local path="$1"
  local repo_root="${SORIDORMI_REPO_ROOT:-$(pwd)}"

  case "${path}" in
    /data|/data/*)
      printf '%s\n' "${path}"
      ;;
    data)
      printf '/data\n'
      ;;
    data/*)
      printf '/data/%s\n' "${path#data/}"
      ;;
    "${repo_root}/data")
      printf '/data\n'
      ;;
    "${repo_root}/data"/*)
      printf '/data/%s\n' "${path#"${repo_root}/data/"}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

soridormi_translate_container_data_args() {
  local -n _translated_args_ref="$1"
  shift
  local arg
  _translated_args_ref=()
  for arg in "$@"; do
    _translated_args_ref+=("$(soridormi_to_container_data_path "${arg}")")
  done
}
