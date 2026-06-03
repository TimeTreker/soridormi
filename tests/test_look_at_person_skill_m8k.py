from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.scripted_head_skill import (
    HEAD_JOINT_NAMES,
    execute_scripted_head_plan,
    resolve_keyframe_targets_for_execution,
)
import soridormi_runtime.scripted_head_skill as scripted_head_skill
from soridormi_runtime.scripted_social_acceptance import run_acceptance
from soridormi_runtime.skill_execution import SkillExecutionError, SkillExecutionRegistry
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


def _state(*, z: float = 0.30, head_pitch: float = 0.0, head_yaw: float = 0.0) -> RobotState:
    positions = [0.0] * len(JOINT_NAMES)
    positions[JOINT_NAMES.index("head_pitch")] = head_pitch
    positions[JOINT_NAMES.index("head_yaw")] = head_yaw
    return RobotState(
        time=0.0,
        joints=JointState(
            names=list(JOINT_NAMES),
            positions=positions,
            velocities=[0.0] * len(JOINT_NAMES),
            torques=[0.0] * len(JOINT_NAMES),
        ),
        imu=IMUState(),
        base_position_xyz=[0.0, 0.0, z],
        actuator_ctrl=list(positions),
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


def test_look_at_person_manifest_promoted_to_structured_target_skill() -> None:
    skill = _registry().skills["look_at_person"]

    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "scripted_keyframe"
    assert skill["required_actuator_groups"] == ["head_neck"]
    assert skill["safety"]["hardware_enabled"] is False
    assert "structured" in skill["notes"]
    assert "does not run camera perception" in skill["notes"]
    assert skill["parameters"]["end_mode"]["default"] == "hold_target"


def test_look_at_person_plan_consumes_structured_target_direction() -> None:
    plan = _registry().create_plan(
        "look_at_person",
        {
            "target_ref": "person_left",
            "target_yaw_rad": 0.30,
            "target_pitch_rad": -0.06,
            "duration_s": 4.0,
            "hold_fraction": 0.50,
        },
    )

    assert plan.commands == ()
    assert [keyframe.label for keyframe in plan.keyframes] == [
        "look_at_person_neutral_start",
        "look_at_person_acquire_target",
        "look_at_person_hold_target",
    ]
    assert set(plan.keyframes[0].positions_by_name) == set(HEAD_JOINT_NAMES)
    assert plan.keyframes[1].positions_by_name["head_yaw"] == pytest.approx(0.30)
    assert plan.keyframes[1].positions_by_name["head_pitch"] == pytest.approx(-0.06)
    assert plan.keyframes[1].positions_by_name["neck_pitch"] == pytest.approx(0.0)
    assert plan.keyframes[1].positions_by_name["head_roll"] == pytest.approx(0.0)
    assert plan.keyframes[-1].positions_by_name["head_yaw"] == pytest.approx(0.30)
    assert plan.keyframes[-1].positions_by_name["head_pitch"] == pytest.approx(-0.06)
    assert sum(keyframe.duration_s for keyframe in plan.keyframes) == pytest.approx(4.0)
    assert "no perception" in plan.summary
    assert "end_mode=hold_target" in plan.summary


def test_look_at_person_targets_start_neutral_then_hold_target_by_default() -> None:
    plan = _registry().create_plan(
        "look_at_person",
        {"target_yaw_rad": 0.30, "target_pitch_rad": -0.06, "duration_s": 4.0},
    )
    targets = resolve_keyframe_targets_for_execution(
        plan,
        {"neck_pitch": 0.12, "head_pitch": 0.2, "head_yaw": -0.3, "head_roll": 0.1},
    )

    assert targets[0] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
    assert targets[-1]["head_yaw"] == pytest.approx(0.30)
    assert targets[-1]["head_pitch"] == pytest.approx(-0.06)
    assert targets[-1]["neck_pitch"] == pytest.approx(0.0)
    assert targets[-1]["head_roll"] == pytest.approx(0.0)
    assert targets[1]["head_yaw"] == pytest.approx(0.30)
    assert targets[1]["head_pitch"] == pytest.approx(-0.06)
    assert targets[1]["neck_pitch"] == pytest.approx(0.0)
    assert targets[1]["head_roll"] == pytest.approx(0.0)


def test_look_at_person_rejects_out_of_range_target() -> None:
    with pytest.raises(SkillExecutionError, match="above max"):
        _registry().create_plan("look_at_person", {"target_yaw_rad": 2.0})


def test_look_at_person_dry_run_reports_target_and_holds_gaze_by_default() -> None:
    plan = _registry().create_plan(
        "look_at_person",
        {"target_yaw_rad": 0.30, "target_pitch_rad": -0.06, "duration_s": 4.0},
    )
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0)

    assert result.executed is False
    assert result.target_max_positions_by_name["head_yaw"] >= 0.28
    assert result.target_min_positions_by_name["head_pitch"] <= -0.05
    assert result.target_min_positions_by_name["neck_pitch"] == pytest.approx(0.0)
    assert result.target_max_positions_by_name["head_roll"] == pytest.approx(0.0)
    assert result.target_positions_by_name["head_yaw"] == pytest.approx(0.30)
    assert result.target_positions_by_name["head_pitch"] == pytest.approx(-0.06)


def test_look_at_person_live_fake_client_holds_target_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scripted_head_skill, "_load_robot_api_client_class", lambda: _FakeRobotApiClient)
    plan = _registry().create_plan(
        "look_at_person",
        {"target_yaw_rad": 0.30, "target_pitch_rad": -0.06, "duration_s": 4.0},
    )

    result = execute_scripted_head_plan(plan, dry_run=False, control_hz=10.0)

    assert result.executed is True
    assert result.fallen is False
    assert result.observed_max_positions_by_name["head_yaw"] >= 0.28
    assert result.observed_min_positions_by_name["head_pitch"] <= -0.05
    assert result.final_positions_by_name["head_yaw"] == pytest.approx(0.30)
    assert result.final_positions_by_name["head_pitch"] == pytest.approx(-0.06)


def test_look_at_person_can_explicitly_return_to_neutral() -> None:
    plan = _registry().create_plan(
        "look_at_person",
        {
            "target_yaw_rad": 0.30,
            "target_pitch_rad": -0.06,
            "duration_s": 4.0,
            "end_mode": "return_neutral",
        },
    )

    assert [keyframe.label for keyframe in plan.keyframes] == [
        "look_at_person_neutral_start",
        "look_at_person_acquire_target",
        "look_at_person_hold_target",
        "look_at_person_neutral_end",
    ]
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0)

    assert result.target_max_positions_by_name["head_yaw"] >= 0.28
    assert result.target_min_positions_by_name["head_pitch"] <= -0.05
    assert result.target_positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert result.target_positions_by_name["head_pitch"] == pytest.approx(0.0)


def test_look_at_person_is_part_of_acceptance_dry_run() -> None:
    summary = run_acceptance(skill_ids=["look_at_person"], execute=False, control_hz=20.0)

    assert summary.ok is True
    assert [result.skill_id for result in summary.results] == ["look_at_person"]
    result = summary.results[0]
    assert result.commanded_ranges["head_yaw"]["max"] >= 0.28
    assert result.commanded_ranges["head_pitch"]["min"] <= -0.05
    assert result.commanded_ranges["head_roll"]["range"] == pytest.approx(0.0)


def test_look_at_person_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "look_at_person",
            "--args",
            json.dumps({"target_ref": "person_left", "target_yaw_rad": 0.30, "target_pitch_rad": -0.06, "duration_s": 4.0}),
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
    assert payload["plan"]["skill_id"] == "look_at_person"
    assert payload["plan"]["keyframes"][1]["positions_by_name"]["head_yaw"] == pytest.approx(0.30)
    assert payload["result"]["target_positions_by_name"]["head_pitch"] == pytest.approx(-0.06)
