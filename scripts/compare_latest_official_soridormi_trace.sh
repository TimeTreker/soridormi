#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/compare_official_soridormi_trace.sh \
  "${SORIDORMI_OFFICIAL_TRACE:-/data/official_baseline/latest_official_baseline.trace.jsonl}" \
  "${SORIDORMI_TRACE_LOG:-}"
