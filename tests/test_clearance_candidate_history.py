from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.clearance_candidate_history import (
    DEFAULT_REFERENCE_PROFILE,
    build_clearance_candidate_history,
    render_markdown,
)


def _report_payload(
    *,
    scenario_id: str,
    ok: bool,
    swing_p50: float = 0.018,
    swing_p05: float = 0.012,
    low_ratio: float = 0.30,
    forward_distance_m: float = 0.70,
    fallen: bool = False,
) -> dict:
    return {
        "ok": ok,
        "scenario_id": scenario_id,
        "sample_count": 20,
        "duration_s": 2.0,
        "acceptance_thresholds": {
            "require_foot_metrics": True,
            "min_swing_clearance_m": 0.015,
            "max_low_clearance_ratio": 0.25,
        },
        "metrics": {
            "samples_with_feet": 20,
            "swing_clearance_p05_m": swing_p05,
            "swing_clearance_p50_m": swing_p50,
            "low_clearance_swing_ratio": low_ratio,
            "forward_distance_m": forward_distance_m,
            "stuck_ratio": 0.0,
            "fallen": fallen,
        },
        "checks": [
            {
                "name": "foot_metrics_present",
                "ok": True,
                "value": 20,
                "threshold": "present",
                "severity": "error",
            },
            {
                "name": "low_clearance_swing_ratio",
                "ok": low_ratio <= 0.25,
                "value": low_ratio,
                "threshold": 0.25,
                "severity": "error",
            },
            {
                "name": "swing_clearance_p50_m",
                "ok": swing_p50 >= 0.015,
                "value": swing_p50,
                "threshold": 0.015,
                "severity": "error",
            },
        ],
        "stride_step_report": {"samples_with_feet": 20},
        "errors": [] if ok else ["clearance failed"],
        "warnings": [],
    }


def _write_suite_report(root: Path, profile: str, scenario_id: str, payload: dict) -> None:
    report_path = root / profile / scenario_id / "scenario_rollout_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_three_scenario_suite(
    root: Path,
    profile: str,
    *,
    flat_low_ratio: float,
    start_low_ratio: float,
    curve_low_ratio: float,
    distance: float = 0.70,
) -> None:
    values = {
        "flat_walk_varied_speed_v1": flat_low_ratio,
        "start_stop_velocity_ramp_v1": start_low_ratio,
        "curve_turn_walk_v1": curve_low_ratio,
    }
    for scenario_id, low_ratio in values.items():
        _write_suite_report(
            root,
            profile,
            scenario_id,
            _report_payload(
                scenario_id=scenario_id,
                ok=low_ratio <= 0.25,
                low_ratio=low_ratio,
                forward_distance_m=distance,
            ),
        )


def test_clearance_history_keeps_reference_when_candidates_regress(tmp_path: Path) -> None:
    root = tmp_path / "scenario_eval"
    _write_three_scenario_suite(
        root,
        DEFAULT_REFERENCE_PROFILE,
        flat_low_ratio=0.268,
        start_low_ratio=0.257,
        curve_low_ratio=0.308,
    )
    _write_three_scenario_suite(
        root,
        "candidate_regresses_curve",
        flat_low_ratio=0.260,
        start_low_ratio=0.249,
        curve_low_ratio=0.340,
    )

    report = build_clearance_candidate_history(scenario_eval_root=root)
    rendered = render_markdown(report)

    assert not report.ok
    assert report.retained_profile == DEFAULT_REFERENCE_PROFILE
    candidates = {item["profile"]: item for item in report.candidates}
    assert candidates[DEFAULT_REFERENCE_PROFILE]["retention_status"] == "RETAINED_REFERENCE"
    assert (
        candidates["candidate_regresses_curve"]["retention_status"]
        == "REJECT_REFERENCE_REGRESSION"
    )
    assert any("broader clearance redesign" in item for item in report.recommendations)
    assert "candidate_regresses_curve" in rendered


def test_clearance_history_marks_reference_beating_but_blocked_candidate(tmp_path: Path) -> None:
    root = tmp_path / "scenario_eval"
    _write_three_scenario_suite(
        root,
        DEFAULT_REFERENCE_PROFILE,
        flat_low_ratio=0.268,
        start_low_ratio=0.257,
        curve_low_ratio=0.308,
    )
    _write_three_scenario_suite(
        root,
        "candidate_improves_all_but_still_blocked",
        flat_low_ratio=0.260,
        start_low_ratio=0.252,
        curve_low_ratio=0.290,
    )

    report = build_clearance_candidate_history(scenario_eval_root=root)

    assert not report.ok
    assert report.reference_beating_blocked_count == 1
    assert report.best_candidate_profile == "candidate_improves_all_but_still_blocked"
    best = report.candidates[0]
    assert best["retention_status"] == "BEATS_REFERENCE_BUT_CLEARANCE_BLOCKED"
    assert best["candidate_beats_reference"] is True


def test_clearance_history_cli_and_wrapper_write_reports(tmp_path: Path) -> None:
    root = tmp_path / "scenario_eval"
    output_dir = tmp_path / "history"
    _write_three_scenario_suite(
        root,
        DEFAULT_REFERENCE_PROFILE,
        flat_low_ratio=0.268,
        start_low_ratio=0.257,
        curve_low_ratio=0.308,
    )
    _write_three_scenario_suite(
        root,
        "candidate_ready",
        flat_low_ratio=0.200,
        start_low_ratio=0.210,
        curve_low_ratio=0.220,
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/report_clearance_candidate_history.sh",
            "--scenario-eval-root",
            str(root),
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
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["ready_count"] == 1
    assert payload["best_candidate_profile"] == "candidate_ready"
    assert (output_dir / "clearance_candidate_history.json").exists()
    assert "candidate_ready" in (output_dir / "clearance_candidate_history.md").read_text()


def test_clearance_history_module_cli_strict_fails_without_ready_candidate(tmp_path: Path) -> None:
    root = tmp_path / "scenario_eval"
    output_dir = tmp_path / "history"
    _write_three_scenario_suite(
        root,
        DEFAULT_REFERENCE_PROFILE,
        flat_low_ratio=0.268,
        start_low_ratio=0.257,
        curve_low_ratio=0.308,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.clearance_candidate_history",
            "--scenario-eval-root",
            str(root),
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

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["retained_profile"] == DEFAULT_REFERENCE_PROFILE
