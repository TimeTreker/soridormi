from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.scripted_head_skill import execute_scripted_head_plan
import soridormi_runtime.scripted_head_skill as scripted_head_skill
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def _state(*, head_pitch: float = 0.2, head_yaw: float = -0.3) -> RobotState:
    positions = [0.0] * len(JOINT_NAMES)
    positions[JOINT_NAMES.index("head_pitch")] = head_pitch
    positions[JOINT_NAMES.index("head_yaw")] = head_yaw
    controls = list(positions)
    return RobotState(
        time=0.0,
        joints=JointState(
            names=list(JOINT_NAMES),
            positions=positions,
            velocities=[0.0] * len(JOINT_NAMES),
            torques=[0.0] * len(JOINT_NAMES),
        ),
        imu=IMUState(),
        base_position_xyz=[0.0, 0.0, 0.30],
        actuator_ctrl=controls,
    )


class _FakeRobotApiClient:
    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.closed = False
        self.last_state = _state()

    def read_state(self) -> RobotState:
        return self.last_state

    def step_motor_command(self, command):  # type: ignore[no-untyped-def]
        by_name = dict(zip(command.names, command.positions))
        self.last_state = _state(
            head_pitch=float(by_name.get("head_pitch", 0.0)),
            head_yaw=float(by_name.get("head_yaw", 0.0)),
        )
        return self.last_state

    def close(self) -> None:
        self.closed = True


def test_neutral_head_manifest_declares_sim_only_fallback_skill() -> None:
    registry = _registry()
    skill = registry.skills["neutral_head"]

    assert skill["category"] == "social"
    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "scripted_keyframe"
    assert skill["required_actuator_groups"] == ["head_neck"]
    assert skill["safety"]["hardware_enabled"] is False


def test_neutral_head_plan_targets_straight_ahead_pose() -> None:
    plan = _registry().create_plan("neutral_head", {"duration_s": 3.0})

    assert plan.commands == ()
    assert len(plan.keyframes) == 1
    keyframe = plan.keyframes[0]
    assert keyframe.duration_s == pytest.approx(3.0)
    assert keyframe.positions_by_name == {
        "neck_pitch": 0.0,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "head_roll": 0.0,
    }


def test_neutral_head_dry_run_reports_zero_final_target() -> None:
    plan = _registry().create_plan("neutral_head", {"duration_s": 2.0})
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0)

    assert result.executed is False
    assert result.steps == 40
    assert result.target_positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert result.target_positions_by_name["head_yaw"] == pytest.approx(0.0)


def test_neutral_head_live_homes_head_without_requiring_real_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scripted_head_skill, "_load_robot_api_client_class", lambda: _FakeRobotApiClient)
    plan = _registry().create_plan("neutral_head", {"duration_s": 3.0})

    result = execute_scripted_head_plan(plan, dry_run=False, control_hz=10.0)

    assert result.executed is True
    assert result.start_positions_by_name["head_pitch"] == pytest.approx(0.2)
    assert result.start_positions_by_name["head_yaw"] == pytest.approx(-0.3)
    assert result.target_positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert result.target_positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert result.final_positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert result.final_positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert result.fallen is False


def test_neutral_head_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "neutral_head",
            "--args",
            json.dumps({"duration_s": 2.0}),
            "--backend",
            "mujoco",
            "--control-hz",
            "20",
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
    assert payload["plan"]["skill_id"] == "neutral_head"
    assert payload["result"]["target_positions_by_name"]["head_yaw"] == pytest.approx(0.0)
