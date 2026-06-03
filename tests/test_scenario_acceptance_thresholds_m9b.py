from __future__ import annotations

import json
import subprocess
from pathlib import Path

from soridormi_runtime.scenario_curriculum import get_scenario_definition, list_scenarios
from soridormi_runtime.scenario_rollout_eval import (
    ScenarioRolloutThresholds,
    evaluate_scenario_rollout,
    overlay_threshold_overrides,
    thresholds_from_scenario_manifest,
)


def _row(step: int, base_x: float, *, scenario_id: str = "flat_walk_varied_speed_v1") -> dict:
    contact_left = step % 2 == 0
    return {
        "type": "runtime_step",
        "step_index": step,
        "scenario_id": scenario_id,
        "skill_id": "walk_velocity",
        "state": {
            "time": step * 0.1,
            "base_position_xyz": [base_x, 0.0, 0.30],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "feet_position_xyz": [
                [base_x, 0.04, 0.0 if contact_left else 0.04],
                [base_x + 0.04, -0.04, 0.04 if contact_left else 0.0],
            ],
            "feet_contacts": [1.0 if contact_left else 0.0, 0.0 if contact_left else 1.0],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_mujoco_locomotion_scenarios_define_rollout_acceptance_thresholds() -> None:
    required_keys = {
        "schema_version",
        "min_distance_m",
        "min_mean_forward_speed_mps",
        "max_stuck_sample_ratio",
        "require_not_fallen",
        "min_touchdown_count",
        "min_swing_clearance_m",
        "max_low_clearance_ratio",
        "require_foot_metrics",
        "min_base_z_m",
        "max_abs_roll_pitch_rad",
    }

    for scenario in list_scenarios():
        thresholds = scenario.acceptance_thresholds
        assert required_keys.issubset(thresholds), scenario.id
        assert thresholds["schema_version"] == "m9.scenario_rollout_acceptance.v1"
        assert thresholds["min_distance_m"] >= 0.0, scenario.id
        assert thresholds["max_stuck_sample_ratio"] >= 0.0, scenario.id
        assert isinstance(thresholds["require_not_fallen"], bool), scenario.id


def test_thresholds_resolve_from_scenario_manifest() -> None:
    scenario = get_scenario_definition("flat_walk_varied_speed_v1")

    thresholds = thresholds_from_scenario_manifest(scenario)

    assert thresholds.min_distance_m == 0.15
    assert thresholds.min_mean_forward_speed_mps == 0.03
    assert thresholds.max_stuck_sample_ratio == 0.20
    assert thresholds.require_not_fallen is True


def test_manifest_thresholds_are_used_by_default(tmp_path: Path) -> None:
    path = tmp_path / "short_walk.jsonl"
    _write_jsonl(
        path,
        [
            _row(0, 0.00),
            _row(1, 0.04),
            _row(2, 0.08),
            _row(3, 0.12),
        ],
    )

    report = evaluate_scenario_rollout(path, scenario_id="flat_walk_varied_speed_v1")

    assert not report.ok
    assert report.threshold_source == "scenario_manifest"
    assert report.acceptance_thresholds["min_distance_m"] == 0.15
    assert report.metrics["forward_distance_m"] == 0.12
    assert any(check["name"] == "forward_distance_m" and not check["ok"] for check in report.checks)


def test_explicit_threshold_override_can_relax_manifest_requirement(tmp_path: Path) -> None:
    path = tmp_path / "short_walk_relaxed.jsonl"
    _write_jsonl(
        path,
        [
            _row(0, 0.00),
            _row(1, 0.04),
            _row(2, 0.08),
            _row(3, 0.12),
        ],
    )

    report = evaluate_scenario_rollout(
        path,
        scenario_id="flat_walk_varied_speed_v1",
        thresholds=ScenarioRolloutThresholds(min_distance_m=0.10, min_mean_forward_speed_mps=0.02),
    )

    assert report.ok
    assert report.threshold_source == "explicit"
    assert report.acceptance_thresholds["min_distance_m"] == 0.10


def test_overlay_threshold_overrides_only_changes_explicit_fields() -> None:
    base = ScenarioRolloutThresholds(min_distance_m=0.15, max_stuck_sample_ratio=0.20, require_foot_metrics=False)

    overlaid = overlay_threshold_overrides(base, min_distance_m=0.05, require_foot_metrics=True)

    assert overlaid.min_distance_m == 0.05
    assert overlaid.max_stuck_sample_ratio == 0.20
    assert overlaid.require_foot_metrics is True


def test_cli_uses_manifest_thresholds_without_override(tmp_path: Path) -> None:
    path = tmp_path / "short_walk_cli.jsonl"
    _write_jsonl(path, [_row(0, 0.00), _row(1, 0.04), _row(2, 0.08), _row(3, 0.12)])

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.scenario_rollout_eval",
            "--scenario",
            "flat_walk_varied_speed_v1",
            "--log",
            str(path),
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["threshold_source"] == "scenario_manifest"
    assert payload["acceptance_thresholds"]["min_distance_m"] == 0.15


def test_cli_threshold_override_marks_source_explicit(tmp_path: Path) -> None:
    path = tmp_path / "short_walk_cli_override.jsonl"
    _write_jsonl(path, [_row(0, 0.00), _row(1, 0.04), _row(2, 0.08), _row(3, 0.12)])

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.scenario_rollout_eval",
            "--scenario",
            "flat_walk_varied_speed_v1",
            "--log",
            str(path),
            "--min-distance-m",
            "0.10",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    payload = json.loads(result.stdout)
    assert payload["threshold_source"] == "explicit"
    assert payload["acceptance_thresholds"]["min_distance_m"] == 0.10
