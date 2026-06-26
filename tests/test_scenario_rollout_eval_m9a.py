from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.scenario_rollout_eval import (
    ScenarioRolloutThresholds,
    build_scenario_run_plan,
    evaluate_scenario_rollout,
    render_markdown,
)


def _row(
    step: int,
    *,
    base_x: float,
    contacts: list[float],
    left_x: float,
    right_x: float,
    base_z: float = 0.30,
    scenario_id: str = "flat_walk_varied_speed_v1",
    skill_id: str = "walk_velocity",
) -> dict:
    left_contact = contacts[0] >= 0.5
    right_contact = contacts[1] >= 0.5
    return {
        "type": "runtime_step",
        "step_index": step,
        "scenario_id": scenario_id,
        "skill_id": skill_id,
        "state": {
            "time": step * 0.1,
            "base_position_xyz": [base_x, 0.0, base_z],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "feet_position_xyz": [
                [left_x, 0.04, 0.0 if left_contact else 0.04],
                [right_x, -0.04, 0.0 if right_contact else 0.04],
            ],
            "feet_contacts": contacts,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _passing_rows() -> list[dict]:
    return [
        _row(0, base_x=0.00, contacts=[1.0, 0.0], left_x=0.00, right_x=0.05),
        _row(1, base_x=0.04, contacts=[0.0, 1.0], left_x=0.04, right_x=0.10),
        _row(2, base_x=0.08, contacts=[1.0, 0.0], left_x=0.15, right_x=0.12),
        _row(3, base_x=0.12, contacts=[0.0, 1.0], left_x=0.16, right_x=0.22),
        _row(4, base_x=0.16, contacts=[1.0, 0.0], left_x=0.28, right_x=0.24),
        _row(5, base_x=0.20, contacts=[0.0, 1.0], left_x=0.30, right_x=0.36),
    ]


def test_build_scenario_run_plan_uses_manifest_command_space() -> None:
    plan = build_scenario_run_plan("flat_walk_varied_speed_v1", duration_s=4.0, control_hz=50.0)

    assert plan.scenario_id == "flat_walk_varied_speed_v1"
    assert plan.skill_id == "walk_velocity"
    assert plan.profile == "open_duck_forward"
    assert plan.steps == 200
    assert plan.args["duration_s"] == 4.0
    assert 0.0 < plan.args["vx_mps"] <= 0.25
    assert plan.args["yaw_radps"] == 0.0
    assert plan.environment_context["terrain_type"] == "flat"


def test_build_curve_scenario_run_plan_picks_visible_turn() -> None:
    plan = build_scenario_run_plan("curve_turn_walk_v1", duration_s=4.0, control_hz=20.0)

    assert plan.skill_id == "curve_walk"
    assert plan.steps == 80
    assert plan.args["vx_mps"] > 0.0
    assert plan.args["yaw_radps"] > 0.0


def test_build_wbc_clearance_enrichment_run_plans_are_bounded_skills() -> None:
    startup = build_scenario_run_plan("startup_tail_clearance_v1", control_hz=20.0)
    reversal = build_scenario_run_plan("s_turn_reversal_v1", control_hz=20.0)
    settle = build_scenario_run_plan("turn_stop_settle_v1", control_hz=20.0)

    assert startup.skill_id == "walk_velocity"
    assert startup.args["duration_s"] == 7.0
    assert 0.0 < startup.args["vx_mps"] <= 0.16
    assert startup.task_context["clearance_focus"] == "startup_tail"

    assert reversal.skill_id == "curve_walk"
    assert reversal.args["vx_mps"] > 0.0
    assert reversal.args["yaw_radps"] > 0.0
    assert reversal.task_context["clearance_focus"] == "turn_reversal"

    assert settle.skill_id == "curve_walk"
    assert settle.args["duration_s"] == 6.25
    assert settle.args["vx_mps"] > 0.0
    assert settle.args["yaw_radps"] > 0.0
    assert settle.task_context["clearance_focus"] == "turn_stop_settle"


def test_evaluate_scenario_rollout_passes_progressing_log(tmp_path: Path) -> None:
    path = tmp_path / "walk.jsonl"
    _write_jsonl(path, _passing_rows())

    report = evaluate_scenario_rollout(
        path,
        scenario_id="flat_walk_varied_speed_v1",
        thresholds=ScenarioRolloutThresholds(min_distance_m=0.15, min_mean_forward_speed_mps=0.05),
    )

    assert report.ok
    assert report.scenario_id == "flat_walk_varied_speed_v1"
    assert report.expected_skill_id == "walk_velocity"
    assert report.metrics["forward_distance_m"] == 0.20
    assert report.metrics["mean_forward_speed_mps"] == 0.4
    assert not report.errors
    assert any(check["name"] == "forward_distance_m" and check["ok"] for check in report.checks)


def test_evaluate_scenario_rollout_fails_no_progress(tmp_path: Path) -> None:
    path = tmp_path / "stuck.jsonl"
    rows = [
        _row(0, base_x=0.0, contacts=[1.0, 0.0], left_x=0.0, right_x=0.0),
        _row(1, base_x=0.0, contacts=[0.0, 1.0], left_x=0.0, right_x=0.0),
        _row(2, base_x=0.0, contacts=[1.0, 0.0], left_x=0.0, right_x=0.0),
    ]
    _write_jsonl(path, rows)

    report = evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1")

    assert not report.ok
    assert any("forward_distance_m" in error for error in report.errors)
    assert any("mean_forward_speed_mps" in error for error in report.errors)


def test_evaluate_scenario_rollout_fails_fall(tmp_path: Path) -> None:
    path = tmp_path / "fall.jsonl"
    _write_jsonl(
        path,
        [
            _row(0, base_x=0.0, contacts=[1.0, 0.0], left_x=0.0, right_x=0.0, base_z=0.30),
            _row(1, base_x=0.1, contacts=[0.0, 1.0], left_x=0.1, right_x=0.1, base_z=0.08),
        ],
    )

    report = evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1")

    assert not report.ok
    assert report.metrics["fallen"] is True
    assert any("not_fallen" in error for error in report.errors)


def test_render_markdown_contains_acceptance_table(tmp_path: Path) -> None:
    path = tmp_path / "walk.jsonl"
    _write_jsonl(path, _passing_rows())

    rendered = render_markdown(evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1"))

    assert "Soridormi scenario rollout report" in rendered
    assert "flat_walk_varied_speed_v1" in rendered
    assert "Acceptance checks" in rendered
    assert "forward_distance_m" in rendered
