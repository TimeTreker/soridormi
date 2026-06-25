from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_api import VisualExpressionCommand
from soridormi_runtime.skill_execution import SkillExecutionError, SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest
from soridormi_runtime.visual_expression_skill import (
    execute_visual_expression_plan,
    validate_visual_expression_plan,
)


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def test_blink_eyes_manifest_is_visual_only_social_skill() -> None:
    skill = _registry().skills["blink_eyes"]

    assert skill["category"] == "social"
    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "visual_expression"
    assert skill["required_actuator_groups"] == []
    assert skill["safety"]["hardware_enabled"] is False
    assert "visual-only" in skill["notes"]


def test_blink_eyes_plan_rejects_non_integer_count() -> None:
    with pytest.raises(SkillExecutionError, match="integer"):
        _registry().create_plan("blink_eyes", {"count": 1.5})


def test_visual_expression_command_validates_known_eye_states() -> None:
    command = VisualExpressionCommand(expression="eyes_closed", intensity=0.5)

    assert command.expression == "eyes_closed"
    assert command.intensity == pytest.approx(0.5)


def test_sim_robot_forwards_visual_expression_to_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soridormi_runtime.backends import sim as sim_backend

    class FakeRobotApiClient:
        def __init__(self, *, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.visual_expressions: list[VisualExpressionCommand] = []

        def set_visual_expression(self, command: VisualExpressionCommand) -> str:
            self.visual_expressions.append(command)
            return f"visual expression applied: {command.expression}"

    monkeypatch.setattr(sim_backend, "RobotApiClient", FakeRobotApiClient)
    robot = sim_backend.SimRobot(host="127.0.0.1", port=5555)

    message = robot.set_visual_expression(
        VisualExpressionCommand(expression="eyes_closed", intensity=0.5)
    )

    assert message == "visual expression applied: eyes_closed"
    assert robot.client.visual_expressions[0].expression == "eyes_closed"


def test_validate_visual_expression_plan_rejects_motor_skill() -> None:
    with pytest.raises(SkillExecutionError, match="unsupported visual expression skill"):
        validate_visual_expression_plan("nod_yes", "scripted_keyframe")


def test_execute_blink_eyes_dry_run_reports_visual_segments() -> None:
    plan, result = execute_visual_expression_plan(
        "blink_eyes",
        {"count": 1, "closed_duration_s": 0.08, "open_duration_s": 0.12},
        dry_run=True,
    )

    assert plan["skill_id"] == "blink_eyes"
    assert result.executed is False
    assert result.steps == 3
    assert [segment["expression"] for segment in result.visual_expressions] == [
        "eyes_open",
        "eyes_closed",
        "eyes_open",
    ]


def test_visual_expression_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.visual_expression_skill",
            "blink_eyes",
            "--args",
            json.dumps({"count": 1, "closed_duration_s": 0.08, "open_duration_s": 0.12}),
            "--backend",
            "mujoco",
            "--dry-run",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["plan"]["skill_id"] == "blink_eyes"
    assert payload["result"]["steps"] == 3
    assert payload["result"]["visual_expressions"][1]["expression"] == "eyes_closed"


def test_visual_expression_dry_run_imports_without_pyzmq() -> None:
    script = r'''
import builtins
import json

real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "zmq" or name.startswith("zmq."):
        raise ModuleNotFoundError("No module named 'zmq'", name="zmq")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import

from soridormi_runtime.visual_expression_skill import execute_visual_expression_plan

_, result = execute_visual_expression_plan("blink_eyes", {"count": 1}, dry_run=True)
print(json.dumps({"ok": True, "steps": result.steps}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["steps"] == 3


def test_visual_expression_module_help_documents_blink_eyes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.visual_expression_skill",
            "--help",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert "blink_eyes" in proc.stdout
    assert "--dry-run" in proc.stdout
