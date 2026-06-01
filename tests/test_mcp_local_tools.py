from __future__ import annotations

import json
import subprocess
import sys

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
