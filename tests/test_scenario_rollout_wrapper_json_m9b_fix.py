from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_scenario_rollout_shell_json_dry_run_is_machine_readable(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/evaluate_scenario_rollout.sh",
            "--scenario",
            "flat_walk_varied_speed_v1",
            "--dry-run-only",
            "--json",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    payload = json.loads(result.stdout)
    assert payload["scenario_id"] == "flat_walk_varied_speed_v1"
    assert payload["skill_id"] == "walk_velocity"
    assert "Soridormi scenario rollout evaluation" in result.stderr
    assert "Soridormi scenario rollout evaluation" not in result.stdout


def test_scenario_rollout_shell_json_keeps_runtime_noise_off_stdout(tmp_path: Path) -> None:
    log_prefix = "m9b_json_stdout_test"
    fake_script_path = tmp_path / "fake_run_skill_in_sim.sh"
    fake_script = f"""#!/usr/bin/env bash
set -euo pipefail
printf 'runtime stdout noise should not reach JSON stdout\\n'
mkdir -p data/logs
cat > data/logs/{log_prefix}_fake.jsonl <<'JSONL'
{{"type":"runtime_step","step_index":0,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.0,"base_position_xyz":[0.0,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.00,0.04,0.0],[0.05,-0.04,0.04]],"feet_contacts":[1.0,0.0]}}}}
{{"type":"runtime_step","step_index":1,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.1,"base_position_xyz":[0.04,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.04,0.04,0.04],[0.10,-0.04,0.0]],"feet_contacts":[0.0,1.0]}}}}
{{"type":"runtime_step","step_index":2,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.2,"base_position_xyz":[0.08,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.15,0.04,0.0],[0.12,-0.04,0.04]],"feet_contacts":[1.0,0.0]}}}}
{{"type":"runtime_step","step_index":3,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.3,"base_position_xyz":[0.12,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.16,0.04,0.04],[0.22,-0.04,0.0]],"feet_contacts":[0.0,1.0]}}}}
{{"type":"runtime_step","step_index":4,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.4,"base_position_xyz":[0.16,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.28,0.04,0.0],[0.24,-0.04,0.04]],"feet_contacts":[1.0,0.0]}}}}
{{"type":"runtime_step","step_index":5,"scenario_id":"flat_walk_varied_speed_v1","skill_id":"walk_velocity","state":{{"time":0.5,"base_position_xyz":[0.20,0.0,0.30],"base_quat_wxyz":[1.0,0.0,0.0,0.0],"feet_position_xyz":[[0.30,0.04,0.04],[0.36,-0.04,0.0]],"feet_contacts":[0.0,1.0]}}}}
JSONL
"""

    try:
        fake_script_path.write_text(fake_script, encoding="utf-8")
        fake_script_path.chmod(0o755)
        env = dict(os.environ)
        env["SORIDORMI_RUN_SKILL_IN_SIM_SH"] = str(fake_script_path)
        result = subprocess.run(
            [
                "bash",
                "scripts/evaluate_scenario_rollout.sh",
                "--scenario",
                "flat_walk_varied_speed_v1",
                "--log-prefix",
                log_prefix,
                "--output-dir",
                str(tmp_path / "out"),
                "--json",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    finally:
        for path in Path("data/logs").glob(f"{log_prefix}*.jsonl"):
            path.unlink(missing_ok=True)

    payload = json.loads(result.stdout)
    assert payload["scenario_id"] == "flat_walk_varied_speed_v1"
    assert payload["ok"] is True
    assert "runtime stdout noise" not in result.stdout
    assert "runtime stdout noise" in result.stderr
