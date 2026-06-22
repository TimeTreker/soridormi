#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python -m soridormi_runtime.m10_teacher_comparison "$@"
