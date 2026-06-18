from __future__ import annotations

from soridormi_runtime.mcp.task_tools import KNOWN_TASK_TYPES, task_capabilities_payload
from soridormi_runtime.task_capabilities import (
    DEFAULT_TASK_CAPABILITY_MANIFEST,
    load_task_capability_manifest,
    task_capabilities_by_type,
    validate_task_capability_manifest,
)


def test_default_task_capability_manifest_is_valid() -> None:
    manifest = load_task_capability_manifest(DEFAULT_TASK_CAPABILITY_MANIFEST)
    validation = validate_task_capability_manifest(manifest)

    assert validation.ok is True, validation.errors
    assert manifest["robot"] == "open_duck_mini_v2"
    assert manifest["task_api_no_motion"] is True
    assert "attack" in manifest["unsafe_task_types"]
    assert "skill_registry" in manifest["ready_subsystems"]


def test_task_tools_known_task_types_follow_manifest_order() -> None:
    manifest = load_task_capability_manifest()
    expected = tuple(
        task["task_type"]
        for task in manifest["task_types"]
    )

    assert KNOWN_TASK_TYPES == expected


def test_task_capabilities_payload_projects_manifest_readiness() -> None:
    manifest = load_task_capability_manifest()
    manifest_by_type = task_capabilities_by_type(manifest)
    payload = task_capabilities_payload(
        mode="sim",
        backend="mujoco",
        emergency_stop=False,
    )
    payload_by_type = {
        task["task_type"]: task
        for task in payload["task_types"]
    }

    assert payload["task_api_no_motion"] is manifest["task_api_no_motion"]
    assert payload["physical_execution_note"] == manifest["physical_execution_note"]
    assert payload["ready_subsystems"] == manifest["ready_subsystems"]
    assert payload["unsafe_task_types"] == sorted(manifest["unsafe_task_types"])

    for task_type, declared in manifest_by_type.items():
        projected = payload_by_type[task_type]
        assert projected["description"] == declared["description"]
        assert projected["readiness"] == declared["readiness"]
        assert projected["execution_modes"] == declared["execution_modes"]
        assert projected["required_subsystems"] == declared["required_subsystems"]
        assert projected["missing_subsystems"] == declared["missing_subsystems"]
        assert projected["external_dependencies"] == declared["external_dependencies"]
        assert projected["reason_code"] == declared["reason_code"]
        assert projected["recommended_actions"] == declared["recommended_actions"]
        assert projected["physical_execution_ready"] is False


def test_emergency_stop_overlays_submit_availability_only() -> None:
    nominal = task_capabilities_payload(
        mode="sim",
        backend="mujoco",
        emergency_stop=False,
    )
    stopped = task_capabilities_payload(
        mode="sim",
        backend="mujoco",
        emergency_stop=True,
    )

    nominal_by_type = {
        task["task_type"]: task
        for task in nominal["task_types"]
    }
    stopped_by_type = {
        task["task_type"]: task
        for task in stopped["task_types"]
    }

    assert stopped["safe_idle"] is False
    assert nominal_by_type["move_velocity"]["persistent_submit_allowed"] is True
    assert stopped_by_type["move_velocity"]["persistent_submit_allowed"] is False
    assert stopped_by_type["move_velocity"]["readiness"] == "skill_dry_run_ready"
    assert stopped_by_type["navigate_to_location"]["reason_code"] == (
        nominal_by_type["navigate_to_location"]["reason_code"]
    )


def test_future_tasks_keep_blocking_reason_codes() -> None:
    manifest = load_task_capability_manifest()
    by_type = task_capabilities_by_type(manifest)

    assert by_type["approach_target"]["reason_code"] == "missing_perception_pipeline"
    assert by_type["navigate_to_location"]["reason_code"] == "missing_navigation_pipeline"
    assert by_type["deliver_object"]["reason_code"] == "missing_manipulation_capability"
    assert by_type["stop_now"]["reason_code"] == "use_safety_tool_for_immediate_stop"
