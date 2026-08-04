#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python -m soridormi_runtime.clearance_teacher_comparison "$@"
