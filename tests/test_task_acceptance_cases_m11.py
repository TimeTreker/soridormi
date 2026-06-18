from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import pytest

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService


TASK_ACCEPTANCE_PATH = Path("task_acceptance_cases/mcp_task_acceptance.yaml")
FORBIDDEN_KEY_PARTS = {
    "action_14d",
    "joint_targets",
    "motor_commands",
    "torque_commands",
    "actuator_ctrl",
}


def _load_suite() -> dict[str, Any]:
    payload = yaml.safe_load(TASK_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            assert not any(part in lowered for part in FORBIDDEN_KEY_PARTS), key
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def _recommended_action_names(payload: dict[str, Any]) -> list[str]:
    return [str(action["action"]) for action in payload["recommended_next_actions"]]


def _expected_preview_actions(expected: dict[str, Any], preview: dict[str, Any]) -> list[str] | None:
    expected_actions = expected.get("recommended_next_actions")
    if not isinstance(expected_actions, list):
        return None
    if preview["accepted"] is True and preview["phase"] == "planning":
        return ["submit_task_when_confirmed"]
    if preview["execution_mode"] in {"skill_dry_run", "skill_sequence_dry_run"}:
        return ["submit_task_when_confirmed", *expected_actions[1:]]
    return [str(action) for action in expected_actions]


def test_task_acceptance_case_file_is_structured() -> None:
    suite = _load_suite()

    assert suite["schema_version"] == "soridormi.task_acceptance.v1"
    assert suite["suite_id"] == "mcp_task_acceptance"
    assert suite["owner"] == "soridormi"
    assert suite["policy_boundary"]["natural_language_to_low_level_policy"] is False
    assert suite["policy_boundary"]["low_level_output"] == "action_14d"
    assert len(suite["cases"]) >= 8
    for case in suite["cases"]:
        assert case["id"].startswith("task_acceptance.")
        assert case["natural_language_command"]
        assert "task_type" in case["task_submit"]
        assert case["expected"]["no_motion"] is True
        assert "no_hardware_actuation" in case["safety_checks"]
        _assert_no_forbidden_keys(case["task_submit"])
        _assert_no_forbidden_keys(case["expected"].get("recommended_next_actions", []))


def test_task_acceptance_cases_replay_against_local_mcp_task_service() -> None:
    suite = _load_suite()
    service = SoridormiLocalToolService()

    for case in suite["cases"]:
        result = service.call_tool("soridormi.task.submit", case["task_submit"])
        expected = case["expected"]

        assert result["accepted"] is expected["accepted"], case["id"]
        assert result["status"] == expected["status"], case["id"]
        assert result["phase"] == expected["phase"], case["id"]
        assert result["execution_mode"] == expected["execution_mode"], case["id"]
        assert result["no_motion"] is True, case["id"]
        assert result["reason_code"] == expected.get("reason_code"), case["id"]
        assert isinstance(result["plan_steps"], list), case["id"]
        assert result["task_graph"]["schema_version"] == "soridormi.task_graph.v1", case["id"]
        assert result["task_graph"]["task_ref"] == result["task_id"], case["id"]
        assert result["task_graph"]["raw_control_allowed"] is False, case["id"]
        assert result["task_graph"]["nodes"], case["id"]
        _assert_no_forbidden_keys(result["plan_steps"])
        _assert_no_forbidden_keys(result["task_graph"])
        _assert_no_forbidden_keys(result["blocked_subsystems"])

        if "skill_id" in expected:
            assert result["skill_id"] == expected["skill_id"], case["id"]
        if "skill_sequence" in expected:
            assert [step["skill_id"] for step in result["skill_sequence"]] == expected["skill_sequence"], case["id"]
            assert [step["skill_id"] for step in result["plan_steps"]] == expected["skill_sequence"], case["id"]
        if "blocked_subsystems" in expected:
            assert result["blocked_subsystems"] == expected["blocked_subsystems"], case["id"]
        if "recommended_next_actions" in expected:
            assert _recommended_action_names(result) == expected["recommended_next_actions"], case["id"]

        events = service.call_tool("soridormi.task.events", {"task_id": result["task_id"]})
        assert events["events"], case["id"]
        if result["terminal"]:
            assert events["events"][-1]["phase"] == result["phase"], case["id"]


def test_task_acceptance_cases_preview_without_persisting_task_records() -> None:
    suite = _load_suite()
    service = SoridormiLocalToolService()

    for case in suite["cases"]:
        preview = service.call_tool("soridormi.task.preview", case["task_submit"])
        expected = case["expected"]

        assert preview["preview_id"].startswith("soridormi-preview-"), case["id"]
        assert "task_id" not in preview, case["id"]
        assert preview["persistent"] is False, case["id"]
        assert preview["would_record_task_on_submit"] is True, case["id"]
        assert preview["accepted"] is expected["accepted"], case["id"]
        assert preview["status"] == expected["status"], case["id"]
        assert preview["phase"] == expected["phase"], case["id"]
        assert preview["execution_mode"] == expected["execution_mode"], case["id"]
        assert preview["no_motion"] is True, case["id"]
        assert preview["reason_code"] == expected.get("reason_code"), case["id"]
        assert preview["task_graph"]["schema_version"] == "soridormi.task_graph.v1", case["id"]
        assert preview["task_graph"]["task_ref"] == preview["preview_id"], case["id"]
        assert preview["task_graph"]["raw_control_allowed"] is False, case["id"]
        assert preview["task_graph"]["nodes"], case["id"]
        _assert_no_forbidden_keys(preview["plan_steps"])
        _assert_no_forbidden_keys(preview["task_graph"])
        _assert_no_forbidden_keys(preview["blocked_subsystems"])
        _assert_no_forbidden_keys(preview["recommended_next_actions"])
        expected_preview_actions = _expected_preview_actions(expected, preview)
        if expected_preview_actions is not None:
            assert _recommended_action_names(preview) == expected_preview_actions, case["id"]

        with pytest.raises(KeyError, match="task not found"):
            service.call_tool(
                "soridormi.task.status",
                {"task_id": preview["preview_id"]},
            )
