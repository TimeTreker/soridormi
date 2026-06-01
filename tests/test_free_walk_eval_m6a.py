from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from soridormi_runtime.free_walk_eval import DEFAULT_FREE_WALK_SUITE, render_free_walk_suite_check, validate_free_walk_suite


def test_default_free_walk_suite_is_bounded_and_covers_required_tags() -> None:
    result = validate_free_walk_suite(DEFAULT_FREE_WALK_SUITE)

    assert result.ok, result.errors
    assert result.scenario_count >= 8
    for tag in result.required_tags:
        assert tag in result.present_tags
    assert any(item.command["x"] < 0 for item in result.scenarios)
    assert any(item.command["yaw"] > 0 for item in result.scenarios)
    assert any(item.command["yaw"] < 0 for item in result.scenarios)


def test_free_walk_suite_rejects_out_of_envelope_command(tmp_path: Path) -> None:
    suite = {
        "name": "bad_free_walk_suite",
        "default_steps": 100,
        "scenarios": [
            {
                "name": "too_fast",
                "tags": ["free_walk", "stand", "forward", "backward", "yaw", "curve", "lateral"],
                "steps": 100,
                "command": {"x": 0.5, "y": 0.0, "yaw": 0.0},
            }
        ],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")

    result = validate_free_walk_suite(path)

    assert not result.ok
    assert any("max_abs_x" in error for error in result.errors)


def test_free_walk_suite_report_mentions_each_scenario() -> None:
    result = validate_free_walk_suite(DEFAULT_FREE_WALK_SUITE)
    report = render_free_walk_suite_check(result)

    assert "Soridormi free-walk suite check" in report
    assert "zero_stand" in report
    assert "forward_curve_left" in report


def test_free_walk_eval_json_cli() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.free_walk_eval",
            "--suite",
            str(DEFAULT_FREE_WALK_SUITE),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["suite_name"] == "open_duck_free_walk_eval_v1"
    assert payload["scenario_count"] >= 8
