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


def _state(*, z: float = 0.30, head_pitch: float = 0.0, neck_pitch: float = 0.0) -> RobotState:
    positions = [0.0] * len(JOINT_NAMES)
    positions[JOINT_NAMES.index("head_pitch")] = head_pitch
    positions[JOINT_NAMES.index("neck_pitch")] = neck_pitch
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
            neck_pitch=float(by_name.get("neck_pitch", 0.0)),
        )
        return self.last_state

    def close(self) -> None:
        self.closed = True


def test_bow_manifest_promoted_to_sim_experimental_head_neck_only() -> None:
    skill = _registry().skills["bow"]

    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "scripted_keyframe"
    assert skill["required_actuator_groups"] == ["head_neck"]
    assert skill["safety"]["hardware_enabled"] is False
    assert "head/neck-only" in skill["description"].lower()
    assert "arms" not in skill["description"].lower()


def test_bow_plan_is_neutral_home_head_neck_trajectory() -> None:
    plan = _registry().create_plan("bow", {"depth": "small", "duration_s": 5.0})

    assert plan.commands == ()
    assert [keyframe.label for keyframe in plan.keyframes] == [
        "bow_neutral_start",
        "bow_down",
        "bow_hold",
        "bow_neutral_end",
    ]
    assert set(plan.keyframes[0].positions_by_name) == set(HEAD_JOINT_NAMES)
    assert plan.keyframes[0].positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert plan.keyframes[0].positions_by_name["neck_pitch"] == pytest.approx(0.0)
    assert plan.keyframes[1].positions_by_name["head_pitch"] < 0.0
    assert plan.keyframes[1].positions_by_name["neck_pitch"] < 0.0
    assert plan.keyframes[1].positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert plan.keyframes[-1].positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert sum(keyframe.duration_s for keyframe in plan.keyframes) == pytest.approx(5.0)


def test_bow_targets_do_not_preserve_prior_head_drift() -> None:
    plan = _registry().create_plan("bow", {"depth": "small", "duration_s": 5.0})
    targets = resolve_keyframe_targets_for_execution(
        plan,
        {"neck_pitch": 0.12, "head_pitch": 0.2, "head_yaw": -0.3, "head_roll": 0.1},
    )

    assert targets[0] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
    assert targets[-1] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
    assert targets[1]["head_pitch"] < 0.0
    assert targets[1]["neck_pitch"] < 0.0
    assert targets[1]["head_yaw"] == pytest.approx(0.0)


def test_bow_rejects_unknown_depth() -> None:
    with pytest.raises(SkillExecutionError, match="unsupported bow depth"):
        _registry().create_plan("bow", {"depth": "deep"})


def test_bow_dry_run_reports_pitch_only_ranges() -> None:
    plan = _registry().create_plan("bow", {"depth": "small", "duration_s": 5.0})
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0)

    assert result.executed is False
    assert result.target_min_positions_by_name["head_pitch"] <= -0.16
    assert result.target_min_positions_by_name["neck_pitch"] <= -0.05
    assert result.target_min_positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert result.target_max_positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert result.target_positions_by_name["head_pitch"] == pytest.approx(0.0)


def test_bow_live_fake_client_returns_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scripted_head_skill, "_load_robot_api_client_class", lambda: _FakeRobotApiClient)
    plan = _registry().create_plan("bow", {"depth": "small", "duration_s": 5.0})

    result = execute_scripted_head_plan(plan, dry_run=False, control_hz=10.0)

    assert result.executed is True
    assert result.fallen is False
    assert result.observed_min_positions_by_name["head_pitch"] <= -0.16
    assert result.observed_min_positions_by_name["neck_pitch"] <= -0.05
    assert result.final_positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert result.final_positions_by_name["neck_pitch"] == pytest.approx(0.0)


def test_bow_is_part_of_acceptance_dry_run() -> None:
    summary = run_acceptance(skill_ids=["bow"], execute=False, control_hz=20.0)

    assert summary.ok is True
    assert [result.skill_id for result in summary.results] == ["bow"]
    result = summary.results[0]
    assert result.commanded_ranges["head_pitch"]["min"] <= -0.16
    assert result.commanded_ranges["head_yaw"]["range"] == pytest.approx(0.0)


def test_bow_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "bow",
            "--args",
            json.dumps({"depth": "small", "duration_s": 5.0}),
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
    assert payload["plan"]["skill_id"] == "bow"
    assert payload["plan"]["keyframes"][1]["positions_by_name"]["head_pitch"] < 0.0
    assert payload["result"]["target_positions_by_name"]["head_yaw"] == pytest.approx(0.0)
