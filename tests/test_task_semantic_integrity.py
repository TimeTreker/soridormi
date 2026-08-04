from __future__ import annotations

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.task_tools import EmbodiedTaskStore, task_capabilities_payload


def test_task_capabilities_project_live_safe_idle() -> None:
    moving = task_capabilities_payload(
        mode="sim",
        backend="runtime",
        emergency_stop=False,
        safe_idle=False,
    )
    idle = task_capabilities_payload(
        mode="sim",
        backend="runtime",
        emergency_stop=False,
        safe_idle=True,
    )
    emergency = task_capabilities_payload(
        mode="sim",
        backend="runtime",
        emergency_stop=True,
        safe_idle=True,
    )

    assert moving["safe_idle"] is False
    assert idle["safe_idle"] is True
    assert emergency["safe_idle"] is False


def test_recover_safe_idle_fails_when_live_body_is_not_idle() -> None:
    result = EmbodiedTaskStore().preview_task(
        {"task_type": "recover_safe_idle", "summary": "confirm safe idle"},
        mode="sim",
        backend="runtime",
        emergency_stop=False,
        safe_idle=False,
    )

    assert result["accepted"] is False
    assert result["status"] == "failed"
    assert result["phase"] == "failed"
    assert result["safe_idle"] is False
    assert result["reason_code"] == "robot_not_safe_idle"
    assert "does not confirm safe idle" in result["reason"]


def test_recover_safe_idle_completes_only_when_live_body_is_idle() -> None:
    result = EmbodiedTaskStore().preview_task(
        {"task_type": "recover_safe_idle", "summary": "confirm safe idle"},
        mode="sim",
        backend="runtime",
        emergency_stop=False,
        safe_idle=True,
    )

    assert result["accepted"] is True
    assert result["status"] == "completed"
    assert result["phase"] == "completed"
    assert result["safe_idle"] is True
    assert result["reason_code"] is None


def test_look_at_target_without_target_fails_closed() -> None:
    result = SoridormiLocalToolService().call_tool(
        "soridormi.task.preview",
        {
            "task_type": "look_at_target",
            "summary": "look toward an unspecified target",
            "parameters": {"target_yaw_rad": 0.1},
        },
    )

    assert result["accepted"] is False
    assert result["status"] == "failed"
    assert result["reason_code"] == "skill_planning_failed"
    assert "must not invent a target" in result["reason"]


def test_look_at_target_without_direction_fails_closed() -> None:
    result = SoridormiLocalToolService().call_tool(
        "soridormi.task.preview",
        {
            "task_type": "look_at_target",
            "summary": "look toward the doorway",
            "parameters": {"target_label": "doorway"},
        },
    )

    assert result["accepted"] is False
    assert result["status"] == "failed"
    assert result["reason_code"] == "skill_planning_failed"
    assert "must not invent a body direction" in result["reason"]


def test_look_at_target_preserves_explicit_target_and_direction() -> None:
    result = SoridormiLocalToolService().call_tool(
        "soridormi.task.preview",
        {
            "task_type": "look_at_target",
            "summary": "look toward the doorway",
            "parameters": {
                "target_label": "doorway",
                "target_yaw_rad": 0.1,
            },
        },
    )

    assert result["accepted"] is True
    assert result["status"] == "completed"
    assert result["skill_id"] == "look_at_person"
    assert result["no_motion"] is True
