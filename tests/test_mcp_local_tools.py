from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle


def _recommended_action_names(payload: dict[str, object]) -> list[str]:
    return [
        str(action["action"])
        for action in payload["recommended_next_actions"]  # type: ignore[index]
    ]


def test_local_tool_service_creates_and_executes_dry_run_motion_plan() -> None:
    service = SoridormiLocalToolService(mode="sim")
    plan = service.call_tool(
        "soridormi.motion.create_plan",
        {"commands": [{"vx": 0.08, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}]},
    )
    assert plan["requires_confirmation"] is True
    result = service.call_tool("soridormi.motion.execute_plan", {"plan_id": plan["plan_id"]})
    assert result["completed"] is True
    assert result["dry_run_only"] is True


def test_local_tool_service_rejects_out_of_bounds_motion_command() -> None:
    service = SoridormiLocalToolService()
    with pytest.raises(ValueError, match="outside"):
        service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 1.0, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}]},
        )


def test_emergency_stop_blocks_motion_execution() -> None:
    service = SoridormiLocalToolService()
    plan = service.call_tool(
        "soridormi.motion.create_plan",
        {"commands": [{"vx": 0.0, "vy": 0.0, "yaw": 0.1, "duration_s": 0.5}]},
    )
    service.call_tool("soridormi.safety.emergency_stop", {"reason": "test"})
    with pytest.raises(RuntimeError, match="emergency_stop"):
        service.call_tool("soridormi.motion.execute_plan", {"plan_id": plan["plan_id"]})


def test_manifest_points_to_local_tool_cli_transport() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    assert {agent.transport.kind for agent in bundle.agents} == {"local_cli"}
    assert all("soridormi_runtime.mcp.call_tool" in agent.transport.args for agent in bundle.agents)


def test_call_tool_cli_returns_status_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "soridormi_runtime.mcp.call_tool", "soridormi.robot.get_status", "--compact"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["output"]["mode"] == "sim"
    assert payload["output"]["safe_idle"] is True


@pytest.mark.parametrize(
    ("mode", "recommendation_only"),
    [
        ("sim", False),
        ("hardware_shadow", True),
        ("hardware_dry_run", False),
    ],
)
def test_named_skill_provider_is_no_motion_in_all_safe_modes(
    mode: str,
    recommendation_only: bool,
) -> None:
    service = SoridormiLocalToolService(mode=mode)
    catalog = service.call_tool("soridormi.skill.list", {})
    skills = {skill["skill_id"]: skill for skill in catalog["skills"]}
    assert catalog["mode"] == mode
    assert "nod_yes" in skills
    assert skills["walk_forward"]["execution"] == "skill_wrapper"
    assert skills["walk_forward"]["semantic_speed_presets_mps"]["quick"] == pytest.approx(0.16)

    planned = service.call_tool(
        "soridormi.skill.create_plan",
        {"skill_id": "nod_yes", "parameters": {"count": 2}},
    )
    executed = service.call_tool(
        "soridormi.skill.execute_plan",
        {"plan_id": planned["plan_id"]},
    )

    assert executed["completed"] is True
    assert executed["skill_id"] == "nod_yes"
    assert executed["no_motion"] is True
    assert executed["recommendation_only"] is recommendation_only
    assert service.get_status()["active_task"] is None


def test_resource_acquisition_mock_is_sim_only_and_returns_completion_evidence() -> None:
    sim = SoridormiLocalToolService(mode="sim")
    sim_skills = {
        skill["skill_id"]: skill
        for skill in sim.call_tool("soridormi.skill.list", {})["skills"]
    }
    assert {"acquire_resource", "deliver_resource", "acquire_and_deliver_resource"} <= set(
        sim_skills
    )
    resource_skill = sim_skills["acquire_and_deliver_resource"]
    assert resource_skill["metadata"]["semantic_scope"] == {
        "responsibility_type": "acquire_and_deliver_resource",
        "resource_kinds": ["physical_object"],
        "delivery_modes": ["physical_handover"],
        "acquisition": "provider_owned",
        "source_resolution": "provider_owned",
    }
    assert resource_skill["metadata"]["resource_contract"]["result_field"] == (
        "resource_outcome"
    )

    planned = sim.call_tool(
        "soridormi.skill.create_plan",
        {
            "skill_id": "acquire_and_deliver_resource",
            "parameters": {
                "resource": {
                    "kind": "physical_object",
                    "description": "a cup of water",
                    "quantity": "one",
                    "attributes": {},
                },
                "source": {
                    "status": "unknown",
                    "description": "",
                    "bindings": {},
                },
                "recipient": {
                    "description": "requester",
                    "referent_id": None,
                },
            },
        },
    )
    executed = sim.call_tool(
        "soridormi.skill.execute_plan",
        {"plan_id": planned["plan_id"]},
    )
    assert executed["completed"] is True
    assert executed["no_motion"] is True
    assert executed["resource_outcome"]["resource_acquired"] is True
    assert executed["resource_outcome"]["resource_delivered"] is True
    assert executed["resource_outcome"]["mocked_simulation"] is True

    for mode in ("hardware_shadow", "hardware_dry_run"):
        service = SoridormiLocalToolService(mode=mode)
        skill_ids = {
            skill["skill_id"]
            for skill in service.call_tool("soridormi.skill.list", {})["skills"]
        }
        assert {
            "acquire_resource",
            "deliver_resource",
            "acquire_and_deliver_resource",
        }.isdisjoint(skill_ids)
        with pytest.raises(ValueError, match="unavailable outside sim mode"):
            service.call_tool(
                "soridormi.skill.create_plan",
                {
                    "skill_id": "acquire_and_deliver_resource",
                    "parameters": {
                        "resource": {
                            "kind": "physical_object",
                            "description": "a cup of water",
                        },
                        "source": {"status": "unknown"},
                        "recipient": {"description": "requester"},
                    },
                },
            )


def test_granular_resource_mocks_require_acquire_before_deliver() -> None:
    service = SoridormiLocalToolService(mode="sim")
    resource = {"kind": "physical_object", "description": "a cup of water"}
    deliver_plan = service.call_tool(
        "soridormi.skill.create_plan",
        {
            "skill_id": "deliver_resource",
            "parameters": {"resource": resource, "recipient": {"description": "requester"}},
        },
    )
    with pytest.raises(RuntimeError, match="requires the matching simulated acquired resource"):
        service.call_tool("soridormi.skill.execute_plan", {"plan_id": deliver_plan["plan_id"]})

    acquire_plan = service.call_tool(
        "soridormi.skill.create_plan",
        {
            "skill_id": "acquire_resource",
            "parameters": {"resource": resource, "source": {"status": "unknown"}},
        },
    )
    acquired = service.call_tool(
        "soridormi.skill.execute_plan", {"plan_id": acquire_plan["plan_id"]}
    )
    assert acquired["resource_outcome"]["resource_acquired"] is True
    assert acquired["resource_outcome"]["resource_delivered"] is False

    delivered = service.call_tool(
        "soridormi.skill.execute_plan", {"plan_id": deliver_plan["plan_id"]}
    )
    assert delivered["resource_outcome"]["resource_delivered"] is True


def test_named_skill_plan_accepts_valid_chromie_proposal_metadata() -> None:
    service = SoridormiLocalToolService()

    result = service.call_tool(
        "soridormi.skill.create_plan",
        {
            "skill_id": "nod_yes",
            "chromie_intent": {
                "execution_mode": "proposed",
                "execution_semantics": "proposal_from_chromie",
                "requires_runtime_validation": True,
                "physical_state_source": "soridormi_runtime",
                "chromie_must_not_provide_physical_coordinates": True,
                "soridormi_owns_pose_estimation": True,
                "route_stage": "goal_interpreter",
            },
        },
    )

    assert result["skill_id"] == "nod_yes"


@pytest.mark.parametrize(
    "chromie_intent",
    [
        {"execution_mode": "execute"},
        {
            "execution_mode": "proposed",
            "execution_semantics": "proposal_from_chromie",
            "requires_runtime_validation": False,
        },
        {
            "execution_mode": "proposed",
            "execution_semantics": "proposal_from_chromie",
            "requires_runtime_validation": True,
            "route_context": {"target_coordinates": [1.0, 2.0, 3.0]},
        },
    ],
)
def test_named_skill_plan_rejects_invalid_chromie_proposal_metadata(
    chromie_intent: object,
) -> None:
    service = SoridormiLocalToolService()

    with pytest.raises(ValueError, match="chromie_intent"):
        service.call_tool(
            "soridormi.skill.create_plan",
            {"skill_id": "nod_yes", "chromie_intent": chromie_intent},
        )


def test_test_only_fault_injection_is_one_shot_and_clearable() -> None:
    service = SoridormiLocalToolService()
    configured = service.call_tool(
        "soridormi.testing.configure_fault",
        {"scenario_id": "skill_unavailable"},
    )
    first = service.call_tool("soridormi.skill.list", {})
    second = service.call_tool("soridormi.skill.list", {})
    cleared = service.call_tool("soridormi.testing.clear_faults", {})

    assert configured["configured"] is True
    assert next(
        skill for skill in first["skills"] if skill["skill_id"] == "nod_yes"
    )["available"] is False
    assert next(
        skill for skill in second["skills"] if skill["skill_id"] == "nod_yes"
    )["available"] is True
    assert cleared["cleared"] is True


def test_delayed_fault_does_not_block_cancellation() -> None:
    service = SoridormiLocalToolService()
    plan = service.call_tool(
        "soridormi.skill.create_plan",
        {"skill_id": "nod_yes", "parameters": {"count": 2}},
    )
    service.call_tool(
        "soridormi.testing.configure_fault",
        {"scenario_id": "operator_cancel"},
    )
    worker = threading.Thread(
        target=service.call_tool,
        args=("soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]}),
        daemon=True,
    )
    worker.start()
    time.sleep(0.05)

    started = time.perf_counter()
    cancelled = service.call_tool("soridormi.motion.cancel", {})
    elapsed = time.perf_counter() - started

    assert cancelled["cancelled"] is True
    assert elapsed < 0.25


def test_manifest_declares_provider_readiness_contract() -> None:
    bundle = build_soridormi_capability_bundle(mode="hardware_shadow")
    by_name = {
        tool.name: tool
        for agent in bundle.agents
        for tool in agent.tools
    }
    readiness = bundle.metadata["provider_readiness"]

    assert readiness["safe_modes"] == [
        "sim",
        "hardware_shadow",
        "hardware_dry_run",
    ]
    assert by_name["soridormi.testing.configure_fault"].llm_visible is False
    assert by_name["soridormi.testing.clear_faults"].llm_visible is False
    for name in (
        "soridormi.robot.get_status",
        "soridormi.motion.cancel",
        "soridormi.safety.monitor_motion",
        "soridormi.skill.list",
        "soridormi.skill.create_plan",
        "soridormi.skill.execute_plan",
    ):
        assert set(readiness["safe_modes"]).issubset(
            by_name[name].availability.modes
        )


def test_local_stop_cancel_and_status_report_safe_idle() -> None:
    service = SoridormiLocalToolService()

    assert service.call_tool("soridormi.robot.get_status", {})["safe_idle"] is True
    assert service.call_tool("soridormi.motion.stop", {})["safe_idle"] is True
    assert service.call_tool("soridormi.motion.cancel", {})["safe_idle"] is True
    emergency = service.call_tool(
        "soridormi.safety.emergency_stop",
        {"reason": "test"},
    )

    assert emergency["safe_idle"] is False
    assert service.call_tool("soridormi.robot.get_status", {})["safe_idle"] is False


def test_task_submit_records_contract_only_task_and_can_cancel() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "speak_while_moving",
            "summary": "speak while walking remains a cross-agent planning hold",
            "parameters": {"speech_ref": "hello", "motion_ref": "slow_walk"},
            "task_context": {"source": "unit_test"},
            "timeout_s": 10.0,
        },
    )
    status = service.call_tool(
        "soridormi.task.status",
        {"task_id": submitted["task_id"]},
    )
    events = service.call_tool(
        "soridormi.task.events",
        {"task_id": submitted["task_id"]},
    )
    cancelled = service.call_tool(
        "soridormi.task.cancel",
        {"task_id": submitted["task_id"], "reason": "unit test complete"},
    )
    after_cancel_events = service.call_tool(
        "soridormi.task.events",
        {
            "task_id": submitted["task_id"],
            "after_sequence": events["next_after_sequence"],
        },
    )

    assert submitted["accepted"] is True
    assert submitted["status"] == "accepted"
    assert submitted["phase"] == "planning"
    assert submitted["terminal"] is False
    assert "cancelled" in submitted["allowed_next_phases"]
    assert submitted["execution_mode"] == "contract_only"
    assert submitted["no_motion"] is True
    assert [step["kind"] for step in submitted["plan_steps"]] == [
        "speech_coordination",
        "exact_capability_selection",
        "resource_validated_body_activity",
    ]
    assert submitted["plan_steps"][-1]["recommended_tools"] == [
        "soridormi.activity.get_capabilities",
        "soridormi.activity.compile",
        "soridormi.activity.execute",
    ]
    assert status["task_type"] == "speak_while_moving"
    assert status["phase"] == "planning"
    assert events["events"][0]["type"] == "task_accepted"
    assert [event["type"] for event in events["events"]] == [
        "task_accepted",
        "task_resolving",
        "task_planning",
        "task_execution_held",
    ]
    assert events["events"][-1]["phase"] == "planning"
    assert events["schema_version"] == "soridormi.task_events.v1"
    assert events["status"] == "accepted"
    assert events["phase"] == "planning"
    assert events["terminal"] is False
    assert events["safe_idle"] is True
    assert events["returned_count"] == 4
    assert events["latest_sequence"] == 4
    assert events["next_after_sequence"] == 4
    assert events["has_more"] is False
    assert events["poll_recommendation"] == {
        "action": "continue_polling_or_cancel",
        "owner": "chromie",
        "priority": "normal",
        "recommended_after_sequence": 4,
        "recommended_poll_interval_s": 0.5,
        "recommended_tools": [
            "soridormi.task.events",
            "soridormi.task.status",
            "soridormi.task.cancel",
        ],
        "reason_code": "task_active",
    }
    assert cancelled["cancelled"] is True
    assert cancelled["status"] == "cancelled"
    assert cancelled["phase"] == "cancelled"
    assert cancelled["terminal"] is True
    assert [event["type"] for event in after_cancel_events["events"]] == [
        "task_cancelled",
    ]
    assert after_cancel_events["terminal"] is True
    assert after_cancel_events["poll_recommendation"] == {
        "action": "stop_polling",
        "owner": "chromie",
        "priority": "normal",
        "recommended_after_sequence": 5,
        "recommended_tools": [],
        "reason_code": "operator_cancelled",
    }


def test_task_submit_with_client_ref_is_retry_safe() -> None:
    service = SoridormiLocalToolService()
    payload = {
        "client_task_ref": "chromie-task-123",
        "task_type": "speak_while_moving",
        "summary": "sing while walking",
        "parameters": {"speech_ref": "song", "motion_ref": "slow_walk"},
        "timeout_s": 10.0,
    }

    first = service.call_tool("soridormi.task.submit", payload)
    replay = service.call_tool("soridormi.task.submit", payload)
    status = service.call_tool(
        "soridormi.task.status",
        {"client_task_ref": "chromie-task-123"},
    )
    cancelled = service.call_tool(
        "soridormi.task.cancel",
        {"client_task_ref": "chromie-task-123", "reason": "retry-safe cancel"},
    )

    assert first["task_id"] == replay["task_id"]
    assert first["client_task_ref"] == "chromie-task-123"
    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["events_count"] == first["events_count"]
    assert status["task_id"] == first["task_id"]
    assert status["client_task_ref"] == "chromie-task-123"
    assert cancelled["cancelled"] is True
    assert cancelled["task_id"] == first["task_id"]
    assert cancelled["client_task_ref"] == "chromie-task-123"


def test_task_submit_rejects_client_ref_reuse_with_different_payload() -> None:
    service = SoridormiLocalToolService()

    service.call_tool(
        "soridormi.task.submit",
        {
            "client_task_ref": "chromie-task-conflict",
            "task_type": "speak_while_moving",
            "summary": "sing while walking",
            "parameters": {"speech_ref": "song", "motion_ref": "slow_walk"},
        },
    )

    with pytest.raises(ValueError, match="client_task_ref"):
        service.call_tool(
            "soridormi.task.submit",
            {
                "client_task_ref": "chromie-task-conflict",
                "task_type": "speak_while_moving",
                "summary": "different payload",
                "parameters": {"speech_ref": "different", "motion_ref": "slow_walk"},
            },
        )


def test_task_status_expires_planning_hold_after_timeout() -> None:
    service = SoridormiLocalToolService()
    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "client_task_ref": "chromie-timeout-task",
            "task_type": "speak_while_moving",
            "summary": "sing while walking",
            "parameters": {"speech_ref": "song", "motion_ref": "slow_walk"},
            "timeout_s": 1.0,
        },
    )
    record = service.task_store.tasks[submitted["task_id"]]
    record.created_at = record.created_at - 2.0

    status = service.call_tool(
        "soridormi.task.status",
        {"client_task_ref": "chromie-timeout-task"},
    )
    events = service.call_tool(
        "soridormi.task.events",
        {"task_id": submitted["task_id"], "after_sequence": submitted["events_count"]},
    )
    repeated_status = service.call_tool(
        "soridormi.task.status",
        {"task_id": submitted["task_id"]},
    )

    assert status["status"] == "failed"
    assert status["phase"] == "failed"
    assert status["terminal"] is True
    assert status["accepted"] is False
    assert status["reason_code"] == "task_timeout"
    assert status["expired"] is True
    assert status["deadline_at"] == record.deadline_at
    assert status["timeout_elapsed_s"] is not None
    assert [event["type"] for event in events["events"]] == ["task_timed_out"]
    assert events["terminal"] is True
    assert events["expired"] is True
    assert events["poll_recommendation"]["action"] == "stop_polling"
    assert events["poll_recommendation"]["reason_code"] == "task_timeout"
    assert repeated_status["events_count"] == status["events_count"]


def test_task_timeout_with_emergency_policy_reports_dedicated_stop_hint() -> None:
    service = SoridormiLocalToolService()
    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "speak_while_moving",
            "summary": "sing while walking",
            "parameters": {"speech_ref": "song", "motion_ref": "slow_walk"},
            "timeout_s": 1.0,
            "cancellation_policy": "emergency_stop_on_timeout",
        },
    )
    record = service.task_store.tasks[submitted["task_id"]]
    record.created_at = record.created_at - 2.0

    status = service.call_tool(
        "soridormi.task.status",
        {"task_id": submitted["task_id"]},
    )

    assert status["reason_code"] == "task_timeout_emergency_stop_required"
    assert _recommended_action_names(status) == [
        "report_task_timeout",
        "call_emergency_stop_if_motion_active",
    ]


def test_task_preview_is_non_persistent_and_reports_plan_steps() -> None:
    service = SoridormiLocalToolService()

    preview = service.call_tool(
        "soridormi.task.preview",
        {
            "task_type": "navigate_to_location",
            "summary": "walk forward to the house",
            "parameters": {"target_label": "house"},
        },
    )

    assert preview["preview_id"].startswith("soridormi-preview-")
    assert "task_id" not in preview
    assert preview["persistent"] is False
    assert preview["would_record_task_on_submit"] is True
    assert preview["accepted"] is False
    assert preview["reason_code"] == "missing_navigation_pipeline"
    assert preview["blocked_subsystems"] == [
        "target_resolution",
        "localization",
        "route_planner",
        "obstacle_avoidance",
    ]
    assert [step["layer"] for step in preview["plan_steps"]] == [
        "sensing",
        "localization",
        "routing",
        "planning",
        "control",
    ]
    assert preview["task_graph"]["schema_version"] == "soridormi.task_graph.v1"
    assert preview["task_graph"]["task_ref"] == preview["preview_id"]
    assert preview["task_graph"]["current_phase"] == "refused"
    assert preview["task_graph"]["raw_control_allowed"] is False
    assert [node["layer"] for node in preview["task_graph"]["nodes"]] == [
        "sensing",
        "localization",
        "routing",
        "planning",
        "control",
    ]
    assert preview["task_graph"]["nodes"][0]["node_id"] == "step-1"
    assert preview["task_graph"]["nodes"][0]["blocked"] is True
    assert preview["task_graph"]["edges"][0] == {
        "from": "step-1",
        "to": "step-2",
        "kind": "sequence",
    }
    assert _recommended_action_names(preview) == [
        "report_blocked_or_clarify",
        "do_not_lower_to_velocity_recipe",
    ]
    with pytest.raises(KeyError, match="task not found"):
        service.call_tool("soridormi.task.status", {"task_id": preview["preview_id"]})


def test_task_get_capabilities_reports_soridormi_readiness() -> None:
    service = SoridormiLocalToolService()

    payload = service.call_tool("soridormi.task.get_capabilities", {})
    by_type = {
        task["task_type"]: task
        for task in payload["task_types"]
    }

    assert payload["schema_version"] == "soridormi.task_capabilities.v1"
    assert payload["task_api_no_motion"] is True
    assert "skill_registry" in payload["ready_subsystems"]
    assert "walk_forward" in payload["executable_skill_ids"]
    assert "walk_velocity" in payload["executable_skill_ids"]
    assert by_type["move_forward"]["readiness"] == "skill_dry_run_ready"
    assert by_type["move_velocity"]["readiness"] == "skill_dry_run_ready"
    assert by_type["move_velocity"]["physical_execution_ready"] is False
    assert by_type["navigate_to_location"]["readiness"] == "future_blocked"
    assert by_type["navigate_to_location"]["missing_subsystems"] == [
        "target_resolution",
        "localization",
        "route_planner",
        "obstacle_avoidance",
    ]
    assert by_type["speak_while_moving"]["external_dependencies"] == [
        "chromie_speech_coordination",
    ]
    assert by_type["speak_while_moving"]["missing_subsystems"] == []
    assert "body_activity_scheduler" in payload["ready_subsystems"]
    assert "body_command_composer" in payload["ready_subsystems"]
    assert "physical_resource_arbiter" in payload["ready_subsystems"]
    assert by_type["speak_while_moving"]["recommended_actions"] == [
        "preview_task",
        "select_exact_speech_and_body_capabilities",
        "compile_body_activity",
        "coordinate_peer_lanes",
        "monitor_or_cancel_coordinated_group",
    ]
    assert by_type["stop_now"]["readiness"] == "safety_redirect"
    assert by_type["stop_now"]["recommended_actions"] == [
        "call_dedicated_stop_tool",
    ]


def test_task_get_capabilities_reflects_emergency_stop_submit_block() -> None:
    service = SoridormiLocalToolService()
    service.call_tool("soridormi.safety.emergency_stop", {"reason": "test"})

    payload = service.call_tool("soridormi.task.get_capabilities", {})
    by_type = {
        task["task_type"]: task
        for task in payload["task_types"]
    }

    assert payload["emergency_stop"] is True
    assert payload["safe_idle"] is False
    assert by_type["move_velocity"]["persistent_submit_allowed"] is False
    assert by_type["stop_now"]["persistent_submit_allowed"] is False


def test_task_submit_completes_skill_dry_run_for_gesture() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "perform_gesture",
            "summary": "nod twice",
            "parameters": {"gesture": "nod_yes", "count": 2, "duration_s": 2.0},
        },
    )
    events = service.call_tool(
        "soridormi.task.events",
        {"task_id": submitted["task_id"]},
    )

    assert submitted["accepted"] is True
    assert submitted["status"] == "completed"
    assert submitted["phase"] == "completed"
    assert submitted["terminal"] is True
    assert submitted["execution_mode"] == "skill_dry_run"
    assert submitted["no_motion"] is True
    assert submitted["skill_id"] == "nod_yes"
    assert submitted["estimated_duration_s"] == 2.0
    assert [event["type"] for event in events["events"]] == [
        "task_accepted",
        "task_resolving",
        "task_planning",
        "task_executing",
        "task_monitoring",
        "task_completed",
    ]
    assert events["events"][-1]["skill_id"] == "nod_yes"
    assert events["terminal"] is True
    assert events["returned_count"] == len(events["events"])
    assert events["poll_recommendation"]["action"] == "stop_polling"
    assert events["poll_recommendation"]["recommended_after_sequence"] == (
        events["next_after_sequence"]
    )


def test_task_submit_completes_skill_dry_run_for_velocity_task() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "move_velocity",
            "summary": "walk forward slowly",
            "parameters": {"vx_mps": 0.1, "duration_s": 2.0},
        },
    )

    assert submitted["status"] == "completed"
    assert submitted["phase"] == "completed"
    assert submitted["execution_mode"] == "skill_dry_run"
    assert submitted["skill_id"] == "walk_velocity"
    assert submitted["no_motion"] is True


def test_task_submit_completes_skill_dry_run_for_semantic_forward_walk() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "move_forward",
            "summary": "walk forward slowly",
            "parameters": {"speed": "slow", "duration_s": 2.0},
        },
    )

    assert submitted["status"] == "completed"
    assert submitted["phase"] == "completed"
    assert submitted["execution_mode"] == "skill_dry_run"
    assert submitted["skill_id"] == "walk_forward"
    assert submitted["no_motion"] is True
    assert submitted["plan_steps"][0]["summary"].startswith("Dry-run walk_forward")


def test_task_submit_completes_skill_sequence_dry_run() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "skill_sequence",
            "summary": "turn left then nod twice",
            "parameters": {
                "sequence": [
                    {
                        "skill_id": "turn_in_place",
                        "parameters": {"yaw_radps": 0.12, "duration_s": 2.0},
                    },
                    {
                        "skill_id": "nod_yes",
                        "parameters": {"count": 2, "duration_s": 2.0},
                    },
                ]
            },
        },
    )
    events = service.call_tool(
        "soridormi.task.events",
        {"task_id": submitted["task_id"]},
    )

    assert submitted["status"] == "completed"
    assert submitted["phase"] == "completed"
    assert submitted["execution_mode"] == "skill_sequence_dry_run"
    assert submitted["no_motion"] is True
    assert [step["skill_id"] for step in submitted["skill_sequence"]] == [
        "turn_in_place",
        "nod_yes",
    ]
    assert [step["skill_id"] for step in submitted["plan_steps"]] == [
        "turn_in_place",
        "nod_yes",
    ]
    assert submitted["task_graph"]["schema_version"] == "soridormi.task_graph.v1"
    assert submitted["task_graph"]["task_ref"] == submitted["task_id"]
    assert submitted["task_graph"]["current_phase"] == "completed"
    assert submitted["task_graph"]["terminal"] is True
    assert submitted["task_graph"]["raw_control_allowed"] is False
    assert [node["skill_id"] for node in submitted["task_graph"]["nodes"]] == [
        "turn_in_place",
        "nod_yes",
    ]
    assert submitted["task_graph"]["edges"] == [
        {"from": "step-1", "to": "step-2", "kind": "sequence"},
    ]
    assert {step["layer"] for step in submitted["plan_steps"]} == {"skill"}
    assert all(step["status"] == "dry_run_completed" for step in submitted["plan_steps"])
    assert _recommended_action_names(submitted) == [
        "report_contract_dry_run_complete",
        "use_skill_execution_path_for_physical_motion",
        "do_not_report_physical_completion",
    ]
    assert submitted["estimated_duration_s"] == 4.0
    assert events["events"][-1]["details"]["skill_ids"] == [
        "turn_in_place",
        "nod_yes",
    ]


def test_task_preview_can_compile_skill_sequence_without_persisting() -> None:
    service = SoridormiLocalToolService()

    preview = service.call_tool(
        "soridormi.task.preview",
        {
            "task_type": "skill_sequence",
            "summary": "turn left then nod twice",
            "parameters": {
                "sequence": [
                    {
                        "skill_id": "turn_in_place",
                        "parameters": {"yaw_radps": 0.12, "duration_s": 2.0},
                    },
                    {
                        "skill_id": "nod_yes",
                        "parameters": {"count": 2, "duration_s": 2.0},
                    },
                ]
            },
        },
    )

    assert preview["status"] == "completed"
    assert preview["phase"] == "completed"
    assert preview["execution_mode"] == "skill_sequence_dry_run"
    assert preview["persistent"] is False
    assert [step["skill_id"] for step in preview["skill_sequence"]] == [
        "turn_in_place",
        "nod_yes",
    ]
    assert [step["skill_id"] for step in preview["plan_steps"]] == [
        "turn_in_place",
        "nod_yes",
    ]
    assert _recommended_action_names(preview) == [
        "submit_task_when_confirmed",
        "use_skill_execution_path_for_physical_motion",
        "do_not_report_physical_completion",
    ]


def test_task_submit_fails_closed_for_future_navigation_and_manipulation() -> None:
    service = SoridormiLocalToolService()

    navigation = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "navigate_to_location",
            "summary": "walk forward to the house",
            "parameters": {"target_label": "house"},
        },
    )
    delivery = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "deliver_object",
            "summary": "bring me water",
            "parameters": {"object": "water"},
        },
    )

    assert navigation["accepted"] is False
    assert navigation["status"] == "refused"
    assert navigation["phase"] == "refused"
    assert navigation["terminal"] is True
    assert navigation["reason_code"] == "missing_navigation_pipeline"
    assert navigation["task_graph"]["terminal"] is True
    assert navigation["task_graph"]["nodes"][0]["kind"] == "target_resolution"
    assert navigation["blocked_subsystems"] == [
        "target_resolution",
        "localization",
        "route_planner",
        "obstacle_avoidance",
    ]
    assert [step["layer"] for step in navigation["plan_steps"]] == [
        "sensing",
        "localization",
        "routing",
        "planning",
        "control",
    ]
    assert all(step["no_motion"] is True for step in navigation["plan_steps"])
    assert _recommended_action_names(navigation) == [
        "report_blocked_or_clarify",
        "do_not_lower_to_velocity_recipe",
    ]
    assert delivery["accepted"] is False
    assert delivery["reason_code"] == "missing_manipulation_capability"
    assert "manipulation_capability" in delivery["blocked_subsystems"]


def test_task_submit_rejects_unsafe_and_low_level_control_payloads() -> None:
    service = SoridormiLocalToolService()

    unsafe = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "fight",
            "summary": "fight that person",
            "parameters": {},
        },
    )

    assert unsafe["accepted"] is False
    assert unsafe["terminal"] is True
    assert unsafe["reason_code"] == "unsafe_task"
    assert unsafe["blocked_subsystems"] == ["human_safety_policy"]
    assert unsafe["plan_steps"][0]["layer"] == "safety"
    assert _recommended_action_names(unsafe) == [
        "report_refusal",
        "do_not_lower_to_motion",
    ]
    with pytest.raises(ValueError, match="low-level robot control field"):
        service.call_tool(
            "soridormi.task.submit",
            {
                "task_type": "move_velocity",
                "parameters": {"action_14d": [0.0] * 14},
            },
        )


def test_task_submit_refuses_when_emergency_stop_is_active() -> None:
    service = SoridormiLocalToolService()
    service.call_tool("soridormi.safety.emergency_stop", {"reason": "test"})

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "move_velocity",
            "parameters": {"vx_mps": 0.1, "duration_s": 1.0},
        },
    )

    assert submitted["accepted"] is False
    assert submitted["reason_code"] == "emergency_stop_active"
    assert submitted["safe_idle"] is False
    assert _recommended_action_names(submitted) == ["maintain_stop_and_report"]


def test_task_recover_safe_idle_reaches_terminal_completed_phase() -> None:
    service = SoridormiLocalToolService()

    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "recover_safe_idle",
            "summary": "return to safe idle",
        },
    )
    cancelled = service.call_tool(
        "soridormi.task.cancel",
        {"task_id": submitted["task_id"]},
    )
    events = service.call_tool(
        "soridormi.task.events",
        {"task_id": submitted["task_id"], "after_sequence": 2},
    )

    assert submitted["accepted"] is True
    assert submitted["status"] == "completed"
    assert submitted["phase"] == "completed"
    assert submitted["terminal"] is True
    assert cancelled["cancelled"] is False
    assert cancelled["phase"] == "completed"
    assert [event["type"] for event in events["events"]] == [
        "task_planning",
        "task_completed",
    ]
    assert events["schema_version"] == "soridormi.task_events.v1"
    assert events["latest_sequence"] == 4
    assert events["next_after_sequence"] == 4
    assert events["returned_count"] == 2
    assert events["has_more"] is False
    assert events["terminal"] is True
    assert events["poll_recommendation"]["action"] == "stop_polling"


def test_task_events_rejects_negative_cursor() -> None:
    service = SoridormiLocalToolService()
    submitted = service.call_tool(
        "soridormi.task.submit",
        {
            "task_type": "recover_safe_idle",
            "summary": "return to safe idle",
        },
    )

    with pytest.raises(ValueError, match="after_sequence"):
        service.call_tool(
            "soridormi.task.events",
            {"task_id": submitted["task_id"], "after_sequence": -1},
        )
