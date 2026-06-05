#!/usr/bin/env bash

find_latest_policy_log() {
  local log_dir="${1:-data/logs}"
  local latest=""

  if [[ ! -d "${log_dir}" ]]; then
    return 0
  fi

  latest="$(
    ls -1t \
      "${log_dir}"/parity_*.mcap \
      "${log_dir}"/parity_*.jsonl \
      "${log_dir}"/policy_*.mcap \
      "${log_dir}"/policy_*.jsonl \
      "${log_dir}"/runtime_*.mcap \
      "${log_dir}"/runtime_*.jsonl \
      2>/dev/null | head -n 1 || true
  )"

  printf '%s\n' "${latest}"
}
