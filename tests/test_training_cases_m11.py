from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.scenario_curriculum import load_scenario_manifest
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest, skills_by_id


TRAINING_CASE_DIR = Path("training_cases")
EXPECTED_SUITES = {
    "locomotion_basic",
    "head_gestures",
    "compound_skills",
    "safety_recovery",
    "chromie_interaction_commands",
    "navigation_goals",
}
CASE_STATUSES = {"planned", "mujoco_eval_ready", "training_ready", "unsupported_current_robot"}
REQUIRED_CASE_KEYS = {
    "id",
    "status",
    "priority",
    "natural_language_command",
    "target_skill",
    "parameters",
    "task_context",
    "environment_context",
    "expected_outcome",
    "safety_checks",
    "timeout_s",
    "pass_fail_metrics",
    "validation",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    return payload


def _case_files() -> list[Path]:
    return sorted(TRAINING_CASE_DIR.glob("*.yaml"))


def _all_suites() -> list[dict[str, Any]]:
    return [_load_yaml(path) for path in _case_files()]


def test_training_case_library_has_expected_suites() -> None:
    suites = _all_suites()

    assert {suite["suite_id"] for suite in suites} == EXPECTED_SUITES
    assert all(suite["schema_version"] == "soridormi.training_cases.v1" for suite in suites)
    assert all(suite["robot_profile"] == "open_duck_mini_v2" for suite in suites)
    assert all(suite["owner"] == "soridormi" for suite in suites)


def test_training_cases_preserve_policy_boundary() -> None:
    for suite in _all_suites():
        boundary = suite["policy_boundary"]
        assert boundary["natural_language_to_low_level_policy"] is False, suite["suite_id"]
        assert boundary["low_level_output"] == "action_14d", suite["suite_id"]
        assert "planner_role" in boundary, suite["suite_id"]

        for case in suite["cases"]:
            assert case["natural_language_command"]
            assert "natural_language_not_sent_to_policy" in case["safety_checks"] or any(
                key in str(case.get("task_context", {}))
                for key in ("speech_text_to_policy", "raw_perception_to_policy")
            ) or suite["suite_id"] != "chromie_interaction_commands", case["id"]


def test_training_cases_are_structured_and_unique() -> None:
    seen_ids: set[str] = set()

    for suite in _all_suites():
        cases = suite["cases"]
        assert cases, suite["suite_id"]
        assert [case["priority"] for case in cases] == sorted(case["priority"] for case in cases)
        for case in cases:
            assert REQUIRED_CASE_KEYS.issubset(case), case.get("id")
            assert case["id"] not in seen_ids
            seen_ids.add(case["id"])
            assert case["status"] in CASE_STATUSES, case["id"]
            assert isinstance(case["timeout_s"], (int, float)) and case["timeout_s"] > 0, case["id"]
            assert case["safety_checks"], case["id"]
            assert "no_hardware_actuation" in case["safety_checks"], case["id"]
            assert case["pass_fail_metrics"], case["id"]

            for metric in case["pass_fail_metrics"]:
                assert {"name", "operator", "value"}.issubset(metric), case["id"]


def test_training_cases_reference_known_skills_and_scenarios() -> None:
    skill_ids = set(skills_by_id(load_skill_manifest(DEFAULT_SKILL_MANIFEST)))
    scenario_ids = {scenario["id"] for scenario in load_scenario_manifest()["scenarios"]}

    for suite in _all_suites():
        for case in suite["cases"]:
            assert case["target_skill"] in skill_ids, case["id"]
            if "scenario_id" in case:
                assert case["scenario_id"] in scenario_ids, case["id"]
            for step in case.get("sequence", []):
                assert step["skill"] in skill_ids, case["id"]


def test_training_case_library_covers_requested_curriculum() -> None:
    cases = {case["id"]: case for suite in _all_suites() for case in suite["cases"]}

    expected_case_ids = {
        "locomotion_basic.stand_still_safe",
        "locomotion_basic.walk_forward_slow_medium_fast",
        "locomotion_basic.stop_cleanly",
        "locomotion_basic.turn_left_right_in_place",
        "locomotion_basic.curve_walk",
        "locomotion_basic.backward_tiny_step",
        "head_gestures.nod_yes_natural",
        "head_gestures.shake_no_natural",
        "head_gestures.look_cardinal_directions",
        "head_gestures.look_at_person",
        "head_gestures.expressive_idle_while_speaking",
        "compound_skills.walk_while_speaking",
        "compound_skills.walk_then_nod",
        "compound_skills.turn_then_look_at_person",
        "compound_skills.stop_during_walking",
        "compound_skills.cancel_during_multi_step_action",
        "safety_recovery.stumble_or_fall_detection",
        "safety_recovery.safe_idle_after_task",
        "safety_recovery.emergency_stop",
        "safety_recovery.recovery_from_bad_command",
        "safety_recovery.timeout_partial_execution",
        "chromie_interaction.walk_forward_10_seconds",
        "chromie_interaction.turn_left_then_nod_twice",
        "chromie_interaction.sing_while_walking",
        "chromie_interaction.come_closer_slowly",
        "chromie_interaction.stop_now",
        "chromie_interaction.look_at_me_and_say_hello",
        "navigation_goals.reject_unresolved_house_goal",
        "navigation_goals.relative_short_goal_then_stop",
        "navigation_goals.stop_before_unknown_obstacle",
    }

    assert set(cases) == expected_case_ids
    assert cases["chromie_interaction.sing_while_walking"]["external_capabilities"] == [
        "chromie.speech",
        "chromie.audio",
    ]
    assert cases["safety_recovery.recovery_from_bad_command"]["parameters"]["rejected_command"]["vx_mps"] > 1.0
