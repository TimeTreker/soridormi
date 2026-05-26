#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

summary="${1:-data/official_baseline/latest_official_baseline.json}"
if [ ! -f "$summary" ]; then
  echo "No official baseline summary found: $summary"
  echo "Run: ./scripts/run_official_forward_baseline.sh"
  exit 1
fi

python3 - "$summary" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(f"Summary: {path}")
print(f"kind: {payload.get('kind')}")
print(f"command: {payload.get('command')}")
print(f"policy_steps: {payload.get('policy_steps')}")
print(f"sim_time_seconds: {payload.get('sim_time_seconds')}")
print(f"base_displacement_xyz: {payload.get('base_displacement_xyz')}")
print(f"action_stats: {payload.get('action_stats')}")
print(f"motor_target_stats: {payload.get('motor_target_stats')}")
print(f"contact_stats: {payload.get('contact_stats')}")
PY
