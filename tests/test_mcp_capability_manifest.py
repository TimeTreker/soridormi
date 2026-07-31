from __future__ import annotations

from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle


def test_soridormi_manifest_exports_robot_motion_and_safety_tools() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    tool_names = {tool.name for agent in bundle.agents for tool in agent.tools}
    assert "soridormi.robot.get_status" in tool_names
    assert "soridormi.motion.create_plan" in tool_names
    assert "soridormi.motion.execute_plan" in tool_names
    assert "soridormi.task.get_capabilities" in tool_names
    assert "soridormi.task.preview" in tool_names
    assert "soridormi.task.submit" in tool_names
    assert "soridormi.task.status" in tool_names
    assert "soridormi.task.events" in tool_names
    assert "soridormi.task.cancel" in tool_names
    assert "soridormi.safety.emergency_stop" in tool_names


def test_soridormi_manifest_has_no_chromie_user_interaction_tools() -> None:
    bundle = build_soridormi_capability_bundle()
    tool_names = {tool.name for agent in bundle.agents for tool in agent.tools}
    assert not any(name.startswith("chromie.") for name in tool_names)
    assert not any("ask_confirmation" in name for name in tool_names)
    assert not any("speak" in name for name in tool_names)


def test_motion_execute_requires_confirmation_and_monitoring() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    execute = tools["soridormi.motion.execute_plan"]
    assert execute.safety_class == "physical_motion"
    assert execute.confirmation.required is True
    assert execute.monitoring.requires_safety_monitor is True
    assert "soridormi.safety.monitor_motion" in execute.monitoring.recommended_monitor_tools


def test_named_skill_execute_is_declared_as_physical_motion_boundary() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    execute = tools["soridormi.skill.execute_plan"]

    assert execute.safety_class == "physical_motion"
    assert "physical_motion" in execute.effects
    assert execute.confirmation.required is True
    assert execute.monitoring.requires_safety_monitor is True
    assert "no_motion" in execute.output_schema["properties"]
    assert "recommendation_only" in execute.output_schema["properties"]


def test_named_skill_plan_declares_chromie_proposal_boundary() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    create_plan = tools["soridormi.skill.create_plan"]
    proposal = create_plan.input_schema["properties"]["chromie_intent"]

    assert proposal["properties"]["execution_mode"]["const"] == "proposed"
    assert (
        proposal["properties"]["execution_semantics"]["const"]
        == "proposal_from_chromie"
    )
    assert proposal["properties"]["requires_runtime_validation"]["const"] is True
    assert create_plan.llm_hints["chromie_intent_contract"].startswith(
        "Chromie sends proposal metadata only"
    )


def test_status_schema_exposes_safe_idle_for_chromie() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    status = tools["soridormi.robot.get_status"]

    assert "active_task" in status.output_schema["properties"]
    assert "safe_idle" in status.output_schema["properties"]


def test_task_submit_is_contract_only_no_motion_surface() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    submit = tools["soridormi.task.submit"]

    assert submit.safety_class == "planning_only"
    assert "no_motion_contract" in submit.effects
    assert submit.confirmation.required is False
    assert submit.input_schema["additionalProperties"] is False
    assert "navigate_to_location" in submit.input_schema["properties"]["task_type"]["enum"]
    assert "skill_sequence" in submit.input_schema["properties"]["task_type"]["enum"]
    assert "action_14d" in submit.llm_hints["forbidden_fields"]
    assert submit.output_schema["properties"]["no_motion"]["type"] == "boolean"
    assert submit.output_schema["properties"]["phase"]["enum"] == [
        "accepted",
        "resolving",
        "planning",
        "executing",
        "monitoring",
        "recovering",
        "completed",
        "failed",
        "cancelled",
        "refused",
    ]
    assert submit.output_schema["properties"]["terminal"]["type"] == "boolean"
    assert "skill_id" in submit.output_schema["properties"]
    assert "skill_summary" in submit.output_schema["properties"]
    assert "skill_sequence" in submit.output_schema["properties"]
    assert "plan_steps" in submit.output_schema["properties"]
    assert "task_graph" in submit.output_schema["properties"]
    assert "blocked_subsystems" in submit.output_schema["properties"]
    assert "recommended_next_actions" in submit.output_schema["properties"]
    assert "estimated_duration_s" in submit.output_schema["properties"]
    assert "plan_step_boundary" in submit.llm_hints
    assert "next_action_boundary" in submit.llm_hints


def test_task_preview_is_non_persistent_planning_surface() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    preview = tools["soridormi.task.preview"]

    assert preview.safety_class == "planning_only"
    assert preview.execution.side_effect_free is True
    assert "embodied_task_preview" in preview.effects
    assert "preview_id" in preview.output_schema["properties"]
    assert "task_id" not in preview.output_schema["properties"]
    assert "persistent" in preview.output_schema["properties"]
    assert "plan_steps" in preview.output_schema["properties"]
    assert "task_graph" in preview.output_schema["properties"]
    assert "blocked_subsystems" in preview.output_schema["properties"]
    assert "recommended_next_actions" in preview.output_schema["properties"]
    assert "plan_step_boundary" in preview.llm_hints
    assert "next_action_boundary" in preview.llm_hints


def test_task_get_capabilities_is_read_only_soridormi_readiness_surface() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    capabilities = tools["soridormi.task.get_capabilities"]

    assert capabilities.safety_class == "safe_read"
    assert capabilities.execution.side_effect_free is True
    assert "task_capability_readiness" in capabilities.effects
    assert "task_types" in capabilities.output_schema["properties"]
    assert "ready_subsystems" in capabilities.output_schema["properties"]
    assert "readiness_profile" in capabilities.output_schema["properties"]
    assert "physical_execution_boundary" in capabilities.llm_hints


def test_task_cancel_is_safety_control_but_not_physical_motion() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    cancel = tools["soridormi.task.cancel"]

    assert cancel.safety_class == "safety_critical"
    assert "task_lifecycle" in cancel.effects
    assert "physical_motion" not in cancel.effects
    assert "phase" in cancel.output_schema["properties"]
    assert "terminal" in cancel.output_schema["properties"]


def test_task_events_schema_exposes_monitoring_cursor_contract() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    events = tools["soridormi.task.events"]
    properties = events.output_schema["properties"]
    input_properties = events.input_schema["properties"]

    assert input_properties["client_task_ref"]["maxLength"] == 128
    assert "task_id" not in events.input_schema.get("required", [])
    assert properties["schema_version"]["type"] == "string"
    assert properties["client_task_ref"]["type"] == ["string", "null"]
    assert properties["terminal"]["type"] == "boolean"
    assert properties["deadline_at"]["type"] == "number"
    assert properties["expired"]["type"] == "boolean"
    assert properties["latest_sequence"]["type"] == "integer"
    assert properties["next_after_sequence"]["type"] == "integer"
    assert properties["has_more"]["type"] == "boolean"
    assert properties["poll_recommendation"]["type"] == "object"
    for required in (
        "schema_version",
        "client_task_ref",
        "status",
        "phase",
        "terminal",
        "safe_idle",
        "deadline_at",
        "expired",
        "returned_count",
        "latest_sequence",
        "next_after_sequence",
        "has_more",
        "poll_recommendation",
    ):
        assert required in events.output_schema["required"]


def test_task_submit_status_and_cancel_schema_expose_client_ref_and_timeout_contract() -> None:
    bundle = build_soridormi_capability_bundle()
    tools = {tool.name: tool for agent in bundle.agents for tool in agent.tools}
    submit = tools["soridormi.task.submit"]
    status = tools["soridormi.task.status"]
    cancel = tools["soridormi.task.cancel"]

    assert submit.input_schema["properties"]["client_task_ref"]["maxLength"] == 128
    assert submit.output_schema["properties"]["client_task_ref"]["type"] == [
        "string",
        "null",
    ]
    assert submit.output_schema["properties"]["idempotent_replay"]["type"] == "boolean"
    assert submit.output_schema["properties"]["deadline_at"]["type"] == "number"
    assert submit.output_schema["properties"]["expired"]["type"] == "boolean"
    assert submit.output_schema["properties"]["timeout_elapsed_s"]["type"] == [
        "number",
        "null",
    ]
    assert "client_task_ref" in status.input_schema["properties"]
    assert "client_task_ref" in cancel.input_schema["properties"]
    assert cancel.output_schema["properties"]["client_task_ref"]["type"] == [
        "string",
        "null",
    ]


def test_restricted_tools_are_hidden_if_added() -> None:
    bundle = build_soridormi_capability_bundle()
    assert all(tool.safety_class != "restricted" or not tool.llm_visible for agent in bundle.agents for tool in agent.tools)


def test_soridormi_bundle_includes_dag_contract_without_chromie_tools() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    contract = bundle.dag_contract
    assert contract["source"] == "soridormi"
    assert "soridormi.motion.execute_plan" in contract["physical_motion_tools"]
    assert "soridormi.task.get_capabilities" in contract["embodied_task_tools"]
    assert "soridormi.task.preview" in contract["embodied_task_tools"]
    assert "soridormi.task.submit" in contract["embodied_task_tools"]
    assert "chromie.ask_confirmation" in contract["host_required_tools"]
    tool_names = {tool.name for agent in bundle.agents for tool in agent.tools}
    assert "chromie.ask_confirmation" not in tool_names


def test_dag_contract_recommends_monitoring_motion_execution() -> None:
    bundle = build_soridormi_capability_bundle()
    sequence = "\n".join(bundle.dag_contract["default_short_motion_sequence"])
    assert "soridormi.safety.monitor_motion" in sequence
    assert "soridormi.motion.execute_plan" in sequence
