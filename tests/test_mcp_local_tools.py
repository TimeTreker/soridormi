from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle


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
    assert catalog["mode"] == mode
    assert any(skill["skill_id"] == "nod_yes" for skill in catalog["skills"])

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
