from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService


TEXT_INPUT_ACCEPTANCE_PATH = Path("task_acceptance_cases/chromie_text_input_acceptance.yaml")

FORBIDDEN_TEST_ONLY_KEYS = {
    "cli_args",
    "expect_args",
    "expect_skill",
    "expect-skill",
    "script_args",
}
FORBIDDEN_LOW_LEVEL_KEY_PARTS = {
    "action_14d",
    "joint_targets",
    "motor_commands",
    "torque_commands",
    "actuator_ctrl",
}
# Outcomes Chromie's architecture produces. There is no pre-semantic route label:
# Chromie understands every request, and its Value Judgment refuses harmful ones.
CHROMIE_OUTCOMES = {
    "body_capability",
    "conversation_only",
    "social_response",
    "value_judgment_refusal",
}
BODY_TASK_FIELDS = (
    "soridormi_body_task",
    "optional_soridormi_body_task",
    "interim_soridormi_body_task",
)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _load_suite() -> dict[str, Any]:
    payload = yaml.safe_load(TEXT_INPUT_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            assert lowered not in FORBIDDEN_TEST_ONLY_KEYS, key
            assert not any(part in lowered for part in FORBIDDEN_LOW_LEVEL_KEY_PARTS), key
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def _body_task_entries(case: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for field_name in BODY_TASK_FIELDS:
        value = case.get(field_name)
        if value is None:
            continue
        assert isinstance(value, dict), case["id"]
        entries.append(value)
    return entries


def _assert_expected_task_result(case_id: str, result: dict[str, Any], expected: dict[str, Any]) -> None:
    assert result["accepted"] is expected["accepted"], case_id
    assert result["status"] == expected["status"], case_id
    assert result["phase"] == expected["phase"], case_id
    assert result["execution_mode"] == expected["execution_mode"], case_id
    assert result["no_motion"] is expected["no_motion"], case_id
    assert result["reason_code"] == expected.get("reason_code"), case_id
    assert result["task_graph"]["raw_control_allowed"] is False, case_id

    if "skill_id" in expected:
        assert result["skill_id"] == expected["skill_id"], case_id
    if "skill_sequence" in expected:
        assert [step["skill_id"] for step in result["skill_sequence"]] == expected["skill_sequence"], case_id
        assert [step["skill_id"] for step in result["plan_steps"]] == expected["skill_sequence"], case_id
    if "blocked_subsystems" in expected:
        assert result["blocked_subsystems"] == expected["blocked_subsystems"], case_id


def test_text_input_acceptance_file_is_structured() -> None:
    suite = _load_suite()

    assert suite["schema_version"] == "soridormi.chromie_text_input_acceptance.v2"
    assert suite["suite_id"] == "chromie_text_input_acceptance"
    assert suite["owner"] == "soridormi"
    boundary = suite["policy_boundary"]
    assert boundary["raw_user_text_owned_by"] == "chromie"
    assert boundary["soridormi_receives_only_structured_body_tasks"] is True
    assert boundary["natural_language_to_low_level_policy"] is False
    assert boundary["no_cli_expectation_arguments"] is True

    cases = suite["cases"]
    assert len(cases) >= 8
    assert any(_contains_cjk(case["raw_user_text"]) for case in cases)
    assert any(case["expected_chromie_outcome"]["outcome"] == "conversation_only" for case in cases)
    assert any(
        case["expected_chromie_outcome"]["outcome"] == "value_judgment_refusal" for case in cases
    )

    seen_ids: set[str] = set()
    for case in cases:
        assert case["id"].startswith("text_input.")
        assert case["id"] not in seen_ids
        seen_ids.add(case["id"])
        assert case["raw_user_text"]
        assert case["language"]
        assert "expected_chromie_route" not in case, case["id"]
        assert case["expected_chromie_outcome"]["outcome"] in CHROMIE_OUTCOMES, case["id"]
        assert "interim_soridormi_body_task" not in case, case["id"]
        assert case["expected_user_response"]["response_kind"], case["id"]
        _assert_no_forbidden_keys(case)


def test_conversation_only_text_inputs_do_not_submit_body_tasks() -> None:
    suite = _load_suite()

    for case in suite["cases"]:
        outcome = case["expected_chromie_outcome"]["outcome"]
        if outcome in {"conversation_only", "value_judgment_refusal"}:
            assert not _body_task_entries(case), case["id"]
            assert case.get("soridormi_body_task") is None, case["id"]
            assert case.get("soridormi_skill_plan") is None, case["id"]


def test_text_input_body_tasks_replay_against_soridormi_task_service() -> None:
    suite = _load_suite()
    service = SoridormiLocalToolService()

    for case in suite["cases"]:
        for task_entry in _body_task_entries(case):
            result = service.call_tool("soridormi.task.submit", task_entry["submit"])
            _assert_expected_task_result(case["id"], result, task_entry["expected"])


def test_text_input_compound_walk_and_blink_preserves_both_actions() -> None:
    suite = _load_suite()
    by_id = {case["id"]: case for case in suite["cases"]}
    case = by_id["text_input.en_walk_and_blink_compound"]
    task_entry = case["soridormi_body_task"]

    result = SoridormiLocalToolService().call_tool(
        "soridormi.task.submit",
        task_entry["submit"],
    )

    assert [step["skill_id"] for step in result["skill_sequence"]] == [
        "walk_velocity",
        "blink_eyes",
    ]
    submitted_sequence = task_entry["submit"]["parameters"]["sequence"]
    assert submitted_sequence[0]["parameters"]["vx_mps"] == 0.20
    assert submitted_sequence[1]["parameters"]["count"] == 1
    assert result["estimated_duration_s"] >= 10.0
    assert result["no_motion"] is True


def test_object_delivery_is_a_live_simulation_skill_not_a_blocked_task() -> None:
    suite = _load_suite()
    by_id = {case["id"]: case for case in suite["cases"]}
    case = by_id["text_input.bring_water_delivery"]
    forbidden = case["expected_user_response"]["forbidden_claims"]
    assert "object delivery is not available yet" in forbidden
    assert case.get("soridormi_body_task") is None

    entry = case["soridormi_skill_plan"]
    plan = SoridormiLocalToolService().call_tool(
        "soridormi.skill.create_plan",
        {"skill_id": entry["skill_id"], "parameters": entry["parameters"]},
    )
    assert plan["skill_id"] == entry["expected"]["skill_id"]
    assert plan["requires_confirmation"] is entry["expected"]["requires_confirmation"]
    assert entry["parameters"]["recipient"]["referent_id"] == "requester"
