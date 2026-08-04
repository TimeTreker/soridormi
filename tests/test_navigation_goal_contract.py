from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = Path("configs/navigation/open_duck_mini_v2_navigation_contract.json")


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_navigation_contract_rejects_raw_language_boundary() -> None:
    contract = _load_contract()

    assert contract["schema_version"] == "soridormi.navigation_contract.v1"
    assert contract["natural_language_allowed"] is False
    assert contract["example_rejected_request"] == "walk forward to the house"
    assert "raw_natural_language_goal" in contract["refusal_conditions"]
    assert contract["low_level_policy_boundary"]["natural_language_to_policy"] is False
    assert contract["low_level_policy_boundary"]["raw_perception_to_policy"] is False


def test_navigation_contract_defines_full_goal_pipeline() -> None:
    contract = _load_contract()
    stages = [stage["stage"] for stage in contract["pipeline"]]

    assert stages == [
        "target_resolution",
        "localization",
        "route_planning",
        "local_motion_planning",
        "body_execution",
    ]
    assert all(stage["required"] is True for stage in contract["pipeline"])


def test_navigation_goal_schema_requires_bounded_structured_goal() -> None:
    contract = _load_contract()
    schema = contract["structured_goal_schema"]
    required = set(schema["required"])

    assert "target_ref" in required
    assert "target_confidence" in required
    assert "max_distance_m" in required
    assert schema["properties"]["max_distance_m"]["maximum"] <= 3.0
    assert "goal_tolerance" in schema["properties"]["stop_condition"]["enum"]
    assert "obstacle_detected" in schema["properties"]["stop_condition"]["enum"]


def test_navigation_contract_marks_navigation_not_executable_yet() -> None:
    contract = _load_contract()
    status = contract["current_status"]

    assert status["navigate_to_target"] == "future_perception"
    assert status["trajectory_follow"] == "future_evaluation"
    assert status["hardware_enabled"] is False
    assert status["mujoco_first"] is True
