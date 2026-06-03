from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.scenario_suite_eval import (
    build_scenario_suite_plan,
    build_scenario_suite_report,
    render_suite_markdown,
)


def _write_report(path: Path, *, scenario_id: str, ok: bool, distance: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": ok,
                "scenario_id": scenario_id,
                "scenario_title": scenario_id.replace("_", " "),
                "scenario_status": "mujoco_registry_ready",
                "scenario_family": "locomotion_flat",
                "expected_skill_id": "walk_velocity",
                "sample_count": 20,
                "duration_s": 2.0,
                "metrics": {
                    "forward_distance_m": distance,
                    "mean_forward_speed_mps": distance / 2.0,
                    "stuck_ratio": 0.0,
                    "fallen": False,
                    "touchdown_count": 6,
                    "swing_clearance_p50_m": 0.03,
                },
                "errors": [] if ok else ["forward_distance_m: 0 < 0.1"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_suite_plan_defaults_to_ready_locomotion_scenarios_only() -> None:
    plan = build_scenario_suite_plan()

    assert plan.scenario_ids == [
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    ]
    assert all(item["run_plan"] for item in plan.selected)
    assert any(item["scenario_id"] == "look_direction_stationary_v1" for item in plan.skipped)


def test_suite_plan_can_include_planned_terrain_when_requested() -> None:
    plan = build_scenario_suite_plan(include_planned=True, families=["locomotion_terrain"])

    assert "rough_ground_walk_v1" in plan.scenario_ids
    assert "small_stones_walk_v1" in plan.scenario_ids


def test_suite_report_aggregates_pass_fail_and_missing(tmp_path: Path) -> None:
    passing = tmp_path / "flat" / "scenario_rollout_report.json"
    failing = tmp_path / "curve" / "scenario_rollout_report.json"
    _write_report(passing, scenario_id="flat_walk_varied_speed_v1", ok=True, distance=0.3)
    _write_report(failing, scenario_id="curve_turn_walk_v1", ok=False, distance=0.0)

    report = build_scenario_suite_report(
        [passing, failing],
        expected_scenarios=["flat_walk_varied_speed_v1", "curve_turn_walk_v1", "missing_scenario"],
    )

    assert not report.ok
    assert report.scenario_count == 3
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.missing_count == 1
    assert report.summary_metrics["total_forward_distance_m"] == 0.3
    assert any("missing_scenario" in error for error in report.errors)


def test_render_suite_markdown_contains_summary_table(tmp_path: Path) -> None:
    path = tmp_path / "flat" / "scenario_rollout_report.json"
    _write_report(path, scenario_id="flat_walk_varied_speed_v1", ok=True)

    rendered = render_suite_markdown(build_scenario_suite_report([path]))

    assert "Soridormi scenario suite report" in rendered
    assert "Scenario results" in rendered
    assert "flat_walk_varied_speed_v1" in rendered
