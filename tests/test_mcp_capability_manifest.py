from __future__ import annotations

from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle


def test_soridormi_manifest_exports_robot_motion_and_safety_tools() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    tool_names = {tool.name for agent in bundle.agents for tool in agent.tools}
    assert "soridormi.robot.get_status" in tool_names
    assert "soridormi.motion.create_plan" in tool_names
    assert "soridormi.motion.execute_plan" in tool_names
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


def test_restricted_tools_are_hidden_if_added() -> None:
    bundle = build_soridormi_capability_bundle()
    assert all(tool.safety_class != "restricted" or not tool.llm_visible for agent in bundle.agents for tool in agent.tools)
