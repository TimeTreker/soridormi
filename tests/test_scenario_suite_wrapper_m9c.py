from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

READY_LOCOMOTION_SCENARIOS = [
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
    "startup_tail_clearance_v1",
    "s_turn_reversal_v1",
    "turn_stop_settle_v1",
]


def test_evaluate_scenario_suite_dry_run_json_stdout(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    result = subprocess.run(
        [
            "bash",
            "scripts/evaluate_scenario_suite.sh",
            "--dry-run-only",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    payload = json.loads(result.stdout)
    assert payload["scenario_ids"] == READY_LOCOMOTION_SCENARIOS
    assert (output_dir / "suite_plan.json").exists()
    assert "Scenario suite dry-run" not in result.stdout
