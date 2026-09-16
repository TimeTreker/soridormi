from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from soridormi_runtime.skill_manifest import (
    parameters_schema_for_skill,
    validate_skill_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "skills" / "open_duck_mini_v2_skills.json"


REQUIRED_SKILLS = {
    "acquire_and_deliver_resource",
    "acquire_resource",
    "deliver_resource",
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
    "navigate_to_target",
    "step_over_obstacle",
    "rough_ground_walk",
    "run",
    "sit_down",
    "stand_up",
    "crouch",
    "recover_stand",
    "balance_recover",
    "neutral_head",
    "look_direction",
    "look_at_person",
    "blink_eyes",
    "track_person",
    "nod_yes",
    "shake_no",
    "bow",
    "greeting",
    "express_attention",
    "wave_hand",
    "celebrate",
    "hug_gesture",
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
    assert manifest["capability_profile"] == "open_duck_mini_v2_skill_platform"
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
    assert {"locomotion", "navigation", "posture", "social", "resource", "hardware_extension"} <= categories


def test_status_and_execution_values_are_declared() -> None:
    manifest = _load_manifest()
    statuses = set(manifest["status_vocab"])
    executions = set(manifest["execution_vocab"])

    for skill in manifest["skills"]:
        assert skill["status"] in statuses
        assert skill["execution"] in executions
        assert skill["capability_group"]
        assert not str(skill["capability_group"]).startswith("M")


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
    assert 6 <= len(available) <= 24

    for skill in available:
        required = set(skill["required_actuator_groups"])
        assert required <= SUPPORTED_ACTUATOR_GROUPS




def test_resource_capabilities_publish_dynamic_plan_coverage() -> None:
    skills = _skills_by_id()
    expected = {
        "acquire_resource": {
            "requires": [],
            "provides": ["resource_acquired"],
            "completion": ["resource_acquired"],
        },
        "deliver_resource": {
            "requires": ["resource_acquired"],
            "provides": ["resource_delivered"],
            "completion": ["resource_delivered"],
        },
        "acquire_and_deliver_resource": {
            "requires": [],
            "provides": ["resource_acquired", "resource_delivered"],
            "completion": ["resource_acquired", "resource_delivered"],
        },
    }
    for skill_id, contract_expected in expected.items():
        skill = skills[skill_id]
        assert skill["status"] == "available_sim_experimental"
        assert skill["execution"] == "composite"
        assert skill["sim_mock_only"] is True
        assert skill["safety"]["hardware_enabled"] is False
        scope = skill["metadata"]["semantic_scope"]
        assert scope["responsibility_type"] == "acquire_and_deliver_resource"
        assert scope["resource_kinds"] == ["physical_object"]
        contract = skill["metadata"]["resource_contract"]
        assert contract["result_field"] == "resource_outcome"
        assert contract["plan_requires"] == contract_expected["requires"]
        assert contract["plan_provides"] == contract_expected["provides"]
        assert contract["completion_requires"] == contract_expected["completion"]
    assert skills["deliver_resource"]["metadata"]["semantic_scope"]["delivery_modes"] == [
        "physical_handover"
    ]


def test_compact_manifest_parameters_without_defaults_are_required() -> None:
    skills = _skills_by_id()

    look_schema = parameters_schema_for_skill(skills["look_at_person"])
    assert look_schema["required"] == ["target_ref"]

    resource_schema = parameters_schema_for_skill(
        skills["acquire_and_deliver_resource"]
    )
    assert resource_schema["required"] == ["resource", "source", "recipient"]

    blink_schema = parameters_schema_for_skill(skills["blink_eyes"])
    assert "count" not in blink_schema.get("required", [])
    assert skills["acquire_and_deliver_resource"]["metadata"]["semantic_scope"][
        "delivery_modes"
    ] == ["physical_handover"]


def test_walk_velocity_declares_human_semantic_argument_realization() -> None:
    walk = _skills_by_id()["walk_velocity"]
    realization = walk["metadata"]["argument_realization"]

    assert realization["forward_speed"]["source_entity_type"] == "speed"
    assert realization["forward_speed"]["arguments"] == ["vx_mps"]
    assert realization["duration"]["source_entity_type"] == "duration"
    assert realization["duration"]["arguments"] == ["duration_s"]


def test_turn_declares_semantic_orientation_facade() -> None:
    turn = _skills_by_id()["turn_in_place"]
    metadata = turn["metadata"]
    facade = metadata["semantic_facade"]
    semantic_schema = facade["input_schema"]
    provider_schema = parameters_schema_for_skill(turn)

    assert semantic_schema["required"] == ["direction"]
    assert semantic_schema["properties"]["direction"]["enum"] == ["left", "right"]
    assert "yaw_radps" not in semantic_schema["properties"]
    assert "yaw_radps" in provider_schema["properties"]
    yaw = facade["provider_realizations"]["yaw_radps"]
    assert yaw == {
        "kind": "signed_magnitude",
        "direction_argument": "direction",
        "magnitude_argument": "turn_rate_radps",
        "positive_direction": "left",
        "negative_direction": "right",
        "default_magnitude": 0.12,
    }

    contracts = metadata["argument_realization"]
    assert contracts["turn_direction"]["arguments"] == ["direction"]
    assert contracts["turn_duration"]["arguments"] == ["duration_s"]
    assert contracts["turn_count"]["arguments"] == ["count"]
    assert all(
        "yaw_radps" not in contract["arguments"] for contract in contracts.values()
    )


def test_other_signed_provider_axes_publish_semantic_facades() -> None:
    skills = _skills_by_id()
    for skill_id, provider_arg, magnitude_arg, default_magnitude in (
        ("curve_walk", "yaw_radps", "turn_rate_radps", 0.1),
        ("sidestep", "vy_mps", "lateral_speed_mps", 0.02),
    ):
        skill = skills[skill_id]
        facade = skill["metadata"]["semantic_facade"]
        semantic_schema = facade["input_schema"]
        provider_schema = parameters_schema_for_skill(skill)
        assert semantic_schema["properties"]["direction"]["enum"] == ["left", "right"]
        assert provider_arg not in semantic_schema["properties"]
        assert provider_arg in provider_schema["properties"]
        realization = facade["provider_realizations"][provider_arg]
        assert realization["kind"] == "signed_magnitude"
        assert realization["direction_argument"] == "direction"
        assert realization["magnitude_argument"] == magnitude_arg
        assert realization["positive_direction"] == "left"
        assert realization["negative_direction"] == "right"
        assert realization["default_magnitude"] == default_magnitude


def test_manifest_rejects_semantic_facade_with_unknown_provider_target() -> None:
    manifest = deepcopy(_load_manifest())
    turn = next(skill for skill in manifest["skills"] if skill["id"] == "turn_in_place")
    facade = turn["metadata"]["semantic_facade"]
    facade["provider_realizations"]["unknown_axis"] = facade["provider_realizations"].pop(
        "yaw_radps"
    )

    result = validate_skill_manifest(manifest)

    assert not result.ok
    assert any("unknown provider argument 'unknown_axis'" in error for error in result.errors)


def test_argument_realization_is_validated_against_semantic_facade() -> None:
    manifest = deepcopy(_load_manifest())
    turn = next(skill for skill in manifest["skills"] if skill["id"] == "turn_in_place")
    turn["metadata"]["argument_realization"]["turn_direction"]["arguments"] = [
        "yaw_radps"
    ]

    result = validate_skill_manifest(manifest)

    assert not result.ok
    assert any("unknown parameters ['yaw_radps']" in error for error in result.errors)


def test_look_at_person_declares_trusted_target_argument_realization() -> None:
    look = _skills_by_id()["look_at_person"]
    realizations = look["metadata"]["argument_realization"]
    entity = realizations["person_entity_target"]
    addressee = realizations["person_addressee_target"]

    assert entity["source_entity_type"] == "entity"
    assert addressee["source_entity_type"] == "addressee"
    assert entity["planner_owned"] is True
    assert addressee["arguments"] == ["target_ref"]
    assert "trusted target evidence" in entity["contract"]


def test_acquire_and_deliver_declares_structured_binding_realization() -> None:
    resource = _skills_by_id()["acquire_and_deliver_resource"]
    realizations = resource["metadata"]["argument_realization"]

    assert realizations["physical_resource_entity"]["arguments"] == ["resource"]
    assert realizations["physical_resource_location"]["arguments"] == ["source"]
    assert realizations["physical_resource_distance"]["arguments"] == ["source"]
    assert realizations["physical_resource_recipient"]["arguments"] == ["recipient"]
    assert all(item["planner_owned"] is True for item in realizations.values())


def test_skill_manifest_rejects_argument_realization_for_unknown_parameter() -> None:
    manifest = deepcopy(_load_manifest())
    walk = next(skill for skill in manifest["skills"] if skill["id"] == "walk_velocity")
    walk["metadata"]["argument_realization"]["forward_speed"]["arguments"] = [
        "undeclared_speed"
    ]

    result = validate_skill_manifest(manifest)

    assert result.ok is False
    assert any(
        "argument_realization.forward_speed.arguments names unknown parameters"
        in error
        for error in result.errors
    )

def test_visual_arm_social_skills_are_sim_only_while_contact_skills_stay_unsupported() -> None:
    skills = _skills_by_id()
    for skill_id in ["wave_hand", "celebrate", "hug_gesture"]:
        skill = skills[skill_id]
        assert skill["category"] == "social"
        assert skill["status"] == "available_sim_experimental"
        assert skill["execution"] == "visual_arm_gesture"
        assert skill["required_actuator_groups"] == []
        assert skill["safety"]["hardware_enabled"] is False
        assert skill["concurrency"]["write_resources"] == ["visual.arms"]
        assert skill["concurrency"]["control_coupling"] == "independent_output"

    for skill_id in ["point_direction", "high_five"]:
        skill = skills[skill_id]
        assert skill["category"] == "hardware_extension"
        assert skill["status"] == "unsupported_current_robot"
        assert set(skill["required_actuator_groups"]) == UNSUPPORTED_ACTUATOR_GROUPS
        assert skill["execution"] == "future_hardware_extension"


def test_head_social_skills_are_planned_without_arm_requirement() -> None:
    skills = _skills_by_id()
    behavior_domains = {
        "neutral_head": ["social_attention", "posture_expression"],
        "look_direction": ["social_attention", "orientation"],
        "look_at_person": ["social_attention", "orientation"],
        "nod_yes": ["social_attention", "acknowledgement"],
        "shake_no": ["social_attention", "acknowledgement"],
        "bow": ["social_attention", "deference"],
        "express_attention": ["social_attention", "posture_expression"],
        "blink_eyes": ["social_attention", "facial_expression"],
    }
    for skill_id, expected_domains in behavior_domains.items():
        skill = skills[skill_id]
        assert skill["category"] == "social"
        assert set(skill["required_actuator_groups"]) <= SUPPORTED_ACTUATOR_GROUPS
        assert "arms_hands" not in skill["required_actuator_groups"]
        assert skill["metadata"]["behavior_domains"] == expected_domains


def test_blink_eyes_is_visual_expression_not_motor_control() -> None:
    skill = _skills_by_id()["blink_eyes"]

    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "visual_expression"
    assert skill["required_actuator_groups"] == []
    assert "visual-only" in skill["notes"]


def test_obstacle_run_and_posture_remain_future_not_executable() -> None:
    skills = _skills_by_id()
    assert skills["walk_forward"]["status"] == "available_sim"
    assert skills["walk_forward"]["execution"] == "skill_wrapper"
    assert skills["navigate_to_target"]["status"] == "future_perception"
    assert skills["navigate_to_target"]["execution"] == "navigation_pipeline"
    assert skills["step_over_obstacle"]["status"] == "future_residual_rl"
    assert skills["rough_ground_walk"]["status"] == "future_residual_rl"
    assert skills["run"]["status"] == "future"
    assert skills["sit_down"]["status"] == "future_pose_teacher"
    assert skills["stand_up"]["status"] == "future_pose_teacher"


def test_available_skills_declare_physical_concurrency_contracts() -> None:
    manifest = _load_manifest()
    ability_classes = set(manifest["ability_class_vocab"])
    control_couplings = set(manifest["control_coupling_vocab"])
    resources = set(manifest["physical_resources"])

    assert ability_classes == {"subtle_expression", "locomotion_whole_body"}
    assert control_couplings == {
        "independent_output",
        "body_command_overlay",
        "primary_body_controller",
        "standalone_body_motion",
    }

    for skill in manifest["skills"]:
        if skill["status"] not in {"available_sim", "available_sim_experimental"}:
            continue
        concurrency = skill["concurrency"]
        assert concurrency["ability_class"] in ability_classes
        assert concurrency["control_coupling"] in control_couplings
        assert concurrency["write_resources"]
        assert set(concurrency["write_resources"]) <= resources


def test_concurrency_contract_distinguishes_locomotion_overlay_and_visual_output() -> None:
    skills = _skills_by_id()

    walk = skills["walk_velocity"]["concurrency"]
    assert walk["ability_class"] == "locomotion_whole_body"
    assert walk["control_coupling"] == "primary_body_controller"
    assert walk["write_resources"] == ["body.primary_motion"]

    gaze = skills["look_at_person"]["concurrency"]
    assert gaze["ability_class"] == "subtle_expression"
    assert gaze["control_coupling"] == "body_command_overlay"
    assert gaze["write_resources"] == ["body.head_pose"]
    assert gaze["locomotion_envelope"]["max_abs_head_yaw_rad_during_locomotion"] <= 0.18

    blink = skills["blink_eyes"]["concurrency"]
    assert blink["control_coupling"] == "independent_output"
    assert blink["write_resources"] == ["visual.eyes"]

    nod = skills["nod_yes"]["concurrency"]
    assert nod["control_coupling"] == "standalone_body_motion"
    assert "body.primary_motion" in nod["write_resources"]
