from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "skills" / "open_duck_mini_v2_skills.json"


REQUIRED_SKILLS = {
    "stand_idle",
    "stop",
    "walk_velocity",
    "walk_forward",
    "walk_backward",
    "turn_in_place",
    "turn_left",
    "turn_right",
    "curve_walk",
    "curve_left",
    "curve_right",
    "sidestep",
    "sidestep_left",
    "sidestep_right",
    "trajectory_follow",
    "step_over_obstacle",
    "rough_ground_walk",
    "run",
    "sit_down",
    "stand_up",
    "crouch",
    "recover_stand",
    "balance_recover",
    "look_direction",
    "look_at_person",
    "track_person",
    "nod_yes",
    "shake_no",
    "bow",
    "greeting",
    "express_attention",
    "wave_hand",
    "point_direction",
    "high_five",
}


SUPPORTED_ACTUATOR_GROUPS = {"legs", "head_neck"}
UNSUPPORTED_ACTUATOR_GROUPS = {"arms_hands"}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _skills_by_id() -> dict[str, dict]:
    manifest = _load_manifest()
    return {skill["id"]: skill for skill in manifest["skills"]}


def test_skill_manifest_is_valid_json_and_declares_full_universe() -> None:
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["robot"] == "open_duck_mini_v2"
    assert manifest["milestone"] == "M7_skill_platform"
    assert manifest["strategy"]["define_all_skills_first"] is True
    assert manifest["strategy"]["land_implementation_incrementally"] is True
    assert isinstance(manifest["skills"], list)
    assert len(manifest["skills"]) >= 30


def test_skill_manifest_contains_declared_skill_universe() -> None:
    skill_ids = set(_skills_by_id())
    assert REQUIRED_SKILLS <= skill_ids


def test_skill_ids_are_unique_and_categorized() -> None:
    manifest = _load_manifest()
    skill_ids = [skill["id"] for skill in manifest["skills"]]
    assert len(skill_ids) == len(set(skill_ids))

    categories = {skill["category"] for skill in manifest["skills"]}
    assert {"locomotion", "posture", "social", "hardware_extension"} <= categories


def test_status_and_execution_values_are_declared() -> None:
    manifest = _load_manifest()
    statuses = set(manifest["status_vocab"])
    executions = set(manifest["execution_vocab"])

    for skill in manifest["skills"]:
        assert skill["status"] in statuses
        assert skill["execution"] in executions
        assert skill["implementation_phase"]


def test_no_skill_enables_hardware_by_default() -> None:
    manifest = _load_manifest()
    defaults = manifest["defaults"]
    assert defaults["sim_first"] is True
    assert defaults["hardware_enabled"] is False

    for skill in manifest["skills"]:
        safety = skill["safety"]
        assert "interruptible" in safety
        assert "fallback" in safety
        assert safety["hardware_enabled"] is False
        assert skill["status"] != "available_hardware"


def test_current_robot_supports_legs_and_head_neck_but_not_arms() -> None:
    manifest = _load_manifest()
    actuator_groups = manifest["actuator_groups"]
    assert actuator_groups["legs"]["supported"] is True
    assert actuator_groups["head_neck"]["supported"] is True
    assert actuator_groups["arms_hands"]["supported"] is False

    assert set(actuator_groups["legs"]["actuators"]) == {
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    }
    assert set(actuator_groups["head_neck"]["actuators"]) == {
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    }


def test_first_available_subset_is_small_and_supported() -> None:
    skills = _skills_by_id()
    available = [
        skill
        for skill in skills.values()
        if skill["status"] in {"available_sim", "available_sim_experimental"}
    ]
    assert 6 <= len(available) <= 8

    for skill in available:
        required = set(skill["required_actuator_groups"])
        assert required <= SUPPORTED_ACTUATOR_GROUPS


def test_arm_and_hand_social_skills_are_declared_but_unsupported() -> None:
    skills = _skills_by_id()
    for skill_id in ["wave_hand", "point_direction", "high_five"]:
        skill = skills[skill_id]
        assert skill["category"] == "hardware_extension"
        assert skill["status"] == "unsupported_current_robot"
        assert set(skill["required_actuator_groups"]) == UNSUPPORTED_ACTUATOR_GROUPS
        assert skill["execution"] == "future_hardware_extension"


def test_head_social_skills_are_planned_without_arm_requirement() -> None:
    skills = _skills_by_id()
    for skill_id in ["look_direction", "look_at_person", "nod_yes", "shake_no", "bow", "express_attention"]:
        skill = skills[skill_id]
        assert skill["category"] == "social"
        assert set(skill["required_actuator_groups"]) <= SUPPORTED_ACTUATOR_GROUPS
        assert "arms_hands" not in skill["required_actuator_groups"]


def test_obstacle_run_and_posture_remain_future_not_executable() -> None:
    skills = _skills_by_id()
    assert skills["step_over_obstacle"]["status"] == "future_residual_rl"
    assert skills["rough_ground_walk"]["status"] == "future_residual_rl"
    assert skills["run"]["status"] == "future"
    assert skills["sit_down"]["status"] == "future_pose_teacher"
    assert skills["stand_up"]["status"] == "future_pose_teacher"
