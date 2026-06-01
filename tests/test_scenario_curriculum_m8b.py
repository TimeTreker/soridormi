from __future__ import annotations

import pytest

from soridormi_runtime.scenario_curriculum import (
    COLLECTOR_READY_STATUSES,
    ScenarioCurriculumError,
    get_scenario_definition,
    list_scenarios,
    validate_scenario_for_teacher_collection,
)


def test_list_scenarios_is_priority_ordered() -> None:
    scenarios = list_scenarios()

    assert scenarios
    assert [item.priority for item in scenarios] == sorted(item.priority for item in scenarios)
    assert scenarios[0].id == "flat_walk_varied_speed_v1"


def test_scenario_definition_exposes_velocity_ranges_and_context() -> None:
    scenario = get_scenario_definition("flat_walk_varied_speed_v1")

    assert scenario.status in COLLECTOR_READY_STATUSES
    assert scenario.primary_skill == "walk_velocity"
    assert scenario.command_range("vx_mps") == (-0.03, 0.25)
    assert scenario.command_range_text("yaw_radps") == "-0.08,0.08"
    assert scenario.task_context["skill_family"] == "locomotion"
    assert scenario.environment_context["terrain_type"] == "flat"
    assert "velocity_conditioned" in scenario.dataset_tags


def test_validate_scenario_for_teacher_collection_rejects_planned_by_default() -> None:
    scenario = get_scenario_definition("rough_ground_walk_v1")

    with pytest.raises(ScenarioCurriculumError, match="--allow-planned-scenario"):
        validate_scenario_for_teacher_collection(scenario)

    warnings = validate_scenario_for_teacher_collection(scenario, allow_planned=True)
    assert warnings
    assert "metadata-ready" in warnings[0]


def test_velocity_collector_rejects_non_velocity_social_scenario() -> None:
    scenario = get_scenario_definition("look_direction_stationary_v1")

    with pytest.raises(ScenarioCurriculumError, match="--allow-planned-scenario"):
        validate_scenario_for_teacher_collection(scenario)

    with pytest.raises(ScenarioCurriculumError, match="vx_mps"):
        validate_scenario_for_teacher_collection(scenario, allow_planned=True)
