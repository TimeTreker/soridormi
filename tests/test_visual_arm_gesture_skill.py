from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_api import VisualArmPoseCommand
from soridormi_runtime.skill_execution import SkillExecutionError
from soridormi_runtime.visual_arm_gesture_skill import (
    execute_visual_arm_gesture_plan,
    validate_visual_arm_gesture_plan,
)


def test_visual_arm_pose_command_supports_one_sided_display() -> None:
    command = VisualArmPoseCommand(pose="wave_up", side="left")

    assert command.pose == "wave_up"
    assert command.side == "left"
    assert VisualArmPoseCommand(pose="rest").side == "both"


def test_validate_visual_arm_gesture_plan_rejects_contact_skill() -> None:
    with pytest.raises(SkillExecutionError, match="unsupported visual arm gesture skill"):
        validate_visual_arm_gesture_plan("high_five", "future_hardware_extension")


@pytest.mark.parametrize(
    ("skill_id", "parameters", "expected_poses"),
    [
        (
            "wave_hand",
            {"side": "right", "count": 1, "duration_s": 1.2},
            ["rest", "wave_up", "wave_out", "rest"],
        ),
        ("celebrate", {"duration_s": 2.0}, ["rest", "celebrate", "rest"]),
        (
            "hug_gesture",
            {"duration_s": 2.4},
            ["rest", "welcome_open", "welcome_close", "rest"],
        ),
    ],
)
def test_execute_visual_arm_gesture_dry_run_reports_pose_trace(
    skill_id: str,
    parameters: dict[str, object],
    expected_poses: list[str],
) -> None:
    plan, result = execute_visual_arm_gesture_plan(
        skill_id,
        parameters,
        dry_run=True,
    )

    assert plan["skill_id"] == skill_id
    assert result.executed is False
    assert result.steps == len(expected_poses)
    assert [segment["pose"] for segment in result.visual_arm_poses] == expected_poses
    assert result.visual_arm_poses[-1]["pose"] == "rest"


def test_visual_arm_gesture_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.visual_arm_gesture_skill",
            "wave_hand",
            "--args",
            json.dumps({"side": "left", "count": 1, "duration_s": 1.2}),
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
    assert payload["plan"]["execution"] == "visual_arm_gesture"
    assert payload["result"]["visual_arm_poses"][1]["side"] == "left"


def test_visual_arm_gesture_dry_run_imports_without_pyzmq() -> None:
    script = r'''
import builtins
import json

real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "zmq" or name.startswith("zmq."):
        raise ModuleNotFoundError("No module named 'zmq'", name="zmq")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import

from soridormi_runtime.visual_arm_gesture_skill import execute_visual_arm_gesture_plan

_, result = execute_visual_arm_gesture_plan("celebrate", {}, dry_run=True)
print(json.dumps({"ok": True, "steps": result.steps}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert json.loads(proc.stdout) == {"ok": True, "steps": 3}
