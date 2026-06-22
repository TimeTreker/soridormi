from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from soridormi_runtime.m10_clearance_readiness import DEFAULT_REQUIRED_SCENARIOS
from soridormi_runtime.m10_teacher_comparison import compare_m10_teacher_suites


def _suite(*, distance_scale: float = 1.0, stuck_add: float = 0.0) -> dict:
    scenarios = []
    for index, scenario_id in enumerate(DEFAULT_REQUIRED_SCENARIOS, start=1):
        distance = 0.1 * index
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "ok": True,
                "fallen": False,
                "forward_distance_m": distance * distance_scale,
                "mean_forward_speed_mps": distance * distance_scale / 5.0,
                "stuck_ratio": 0.02 + stuck_add,
                "swing_clearance_p50_m": 0.01,
            }
        )
    return {
        "ok": True,
        "scenario_count": len(scenarios),
        "passed_count": len(scenarios),
        "failed_count": 0,
        "missing_count": 0,
        "scenario_results": scenarios,
        "summary_metrics": {
            "fallen_count": 0,
            "total_forward_distance_m": sum(item["forward_distance_m"] for item in scenarios),
            "mean_forward_speed_mps": sum(
                item["mean_forward_speed_mps"] for item in scenarios
            )
            / len(scenarios),
            "max_stuck_ratio": max(item["stuck_ratio"] for item in scenarios),
        },
    }


def test_teacher_comparison_passes_matching_candidate_without_claiming_clearance() -> None:
    result = compare_m10_teacher_suites(_suite(), _suite(distance_scale=0.9))

    assert result.ok
    assert result.status == "TEACHER_COMPARISON_PASS"
    assert len(result.scenarios) == len(DEFAULT_REQUIRED_SCENARIOS)
    assert "does not replace" in result.warnings[0]


def test_teacher_comparison_fails_slow_candidate() -> None:
    result = compare_m10_teacher_suites(_suite(), _suite(distance_scale=0.5))

    assert not result.ok
    assert result.status == "TEACHER_COMPARISON_FAIL"
    assert any("forward distance ratio" in error for error in result.errors)


def test_teacher_comparison_fails_missing_required_scenario() -> None:
    candidate = _suite()
    candidate["scenario_results"].pop()

    result = compare_m10_teacher_suites(_suite(), candidate)

    assert not result.ok
    assert any("missing from candidate suite" in error for error in result.errors)


def test_teacher_comparison_script_writes_reports(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    output_dir = tmp_path / "comparison"
    reference.write_text(json.dumps(_suite()), encoding="utf-8")
    candidate.write_text(json.dumps(_suite(distance_scale=0.9)), encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "scripts/compare_policy_teacher_suite.sh",
            str(reference),
            str(candidate),
            "--output-dir",
            str(output_dir),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "TEACHER_COMPARISON_PASS"
    assert (output_dir / "policy_teacher_comparison.json").exists()
    assert "does not replace" in (
        output_dir / "policy_teacher_comparison.md"
    ).read_text(encoding="utf-8")
