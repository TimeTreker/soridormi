import json
from pathlib import Path


MANIFEST_PATH = Path("configs/scenarios/open_duck_mini_v2_scenarios.json")
REQUIRED_SCENARIO_KEYS = {
    "id",
    "title",
    "status",
    "priority",
    "family",
    "skills",
    "description",
    "task_context",
    "environment_context",
    "command_space",
    "success_metrics",
    "dataset_tags",
}
REQUIRED_CONTEXT_INPUTS = {
    "robot_state",
    "desired_command_or_desired_state",
    "task_context",
    "environment_context",
    "short_history",
}


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_scenario_manifest_exists_and_loads():
    manifest = load_manifest()

    assert manifest["schema_version"] == "soridormi.scenario_curriculum.v1"
    assert manifest["robot_profile"] == "open_duck_mini_v2"
    assert manifest["defaults"]["backend"] == "mujoco"
    assert manifest["defaults"]["hardware_enabled"] is False
    assert manifest["defaults"]["duration_limit_mode"] == "rollout_steps"
    assert len(manifest["scenarios"]) >= 8


def test_policy_context_contract_is_structured_not_language():
    manifest = load_manifest()
    contract = manifest["policy_context_contract"]

    assert set(contract["low_level_input"]) == REQUIRED_CONTEXT_INPUTS
    assert contract["low_level_output"] == "action_14d"
    assert contract["natural_language_allowed"] is False


def test_scenarios_have_unique_ids_and_valid_statuses():
    manifest = load_manifest()
    statuses = set(manifest["status_values"])
    scenarios = manifest["scenarios"]
    ids = [scenario["id"] for scenario in scenarios]

    assert len(ids) == len(set(ids))
    assert all(scenario["status"] in statuses for scenario in scenarios)
    assert all(scenario["priority"] > 0 for scenario in scenarios)
    assert ids == sorted(ids, key=lambda scenario_id: next(s["priority"] for s in scenarios if s["id"] == scenario_id))


def test_each_scenario_has_required_context_metrics_and_tags():
    manifest = load_manifest()
    families = set(manifest["scenario_families"])

    for scenario in manifest["scenarios"]:
        assert REQUIRED_SCENARIO_KEYS.issubset(scenario), scenario["id"]
        assert scenario["family"] in families, scenario["id"]
        assert scenario["skills"], scenario["id"]
        assert scenario["dataset_tags"], scenario["id"]
        assert "skill_family" in scenario["task_context"], scenario["id"]
        assert "terrain_type" in scenario["environment_context"], scenario["id"]
        assert "required" in scenario["success_metrics"], scenario["id"]
        assert "fall" in scenario["success_metrics"]["required"], scenario["id"]


def test_stage_one_scenarios_cover_velocity_yaw_and_ramps():
    manifest = load_manifest()
    scenario_by_id = {scenario["id"]: scenario for scenario in manifest["scenarios"]}

    flat = scenario_by_id["flat_walk_varied_speed_v1"]["command_space"]
    transitions = scenario_by_id["start_stop_velocity_ramp_v1"]["command_space"]
    turning = scenario_by_id["curve_turn_walk_v1"]["command_space"]

    assert flat["vx_mps"][0] <= -0.03
    assert flat["vx_mps"][1] >= 0.25
    assert transitions["vx_mps"][0] == 0.0
    assert any("stop" in ramp for ramp in transitions["ramps"])
    assert turning["yaw_radps"][0] <= -0.20
    assert turning["yaw_radps"][1] >= 0.20


def test_wbc_clearance_enrichment_scenarios_are_ready_and_tagged():
    manifest = load_manifest()
    scenario_by_id = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    expected = {
        "startup_tail_clearance_v1": "startup_tail",
        "s_turn_reversal_v1": "turn_reversal",
        "turn_stop_settle_v1": "turn_stop_settle",
    }

    for scenario_id, clearance_focus in expected.items():
        scenario = scenario_by_id[scenario_id]
        tags = set(scenario["dataset_tags"])

        assert scenario["status"] == "mujoco_registry_ready"
        assert scenario["task_context"]["clearance_focus"] == clearance_focus
        assert scenario["task_context"]["requires_progress"] is True
        assert "wbc_clearance_v0" in tags
        assert "clearance" in tags
        assert "min_foot_clearance_m" in scenario["success_metrics"]["required"]
        assert scenario["acceptance_thresholds"]["require_foot_metrics"] is True


def test_obstacle_and_terrain_scenarios_are_explicitly_contextualized():
    manifest = load_manifest()
    scenario_by_id = {scenario["id"]: scenario for scenario in manifest["scenarios"]}

    rough = scenario_by_id["rough_ground_walk_v1"]
    stop_before = scenario_by_id["stop_before_obstacle_v1"]
    step_over = scenario_by_id["step_over_low_obstacle_v1"]

    assert rough["environment_context"]["terrain_type"] == "rough_ground"
    assert "min_foot_clearance_m" in rough["success_metrics"]["required"]
    assert stop_before["environment_context"]["obstacles"]
    assert stop_before["success_metrics"].get("maximum_obstacle_contact_count") == 0
    assert step_over["task_context"]["requires_obstacle_crossing"] is True
    assert step_over["success_metrics"].get("obstacle_crossed_required") is True
