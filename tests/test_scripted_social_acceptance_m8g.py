from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.scripted_head_skill import execute_scripted_head_plan
import soridormi_runtime.scripted_head_skill as scripted_head_skill
from soridormi_runtime.scripted_social_acceptance import run_acceptance
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


HEAD_NAMES = [
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


def _state(*, z: float = 0.30, head_yaw: float = 0.0, head_pitch: float = 0.0) -> RobotState:
    positions = [0.0] * len(HEAD_NAMES)
    positions[HEAD_NAMES.index("head_yaw")] = head_yaw
    positions[HEAD_NAMES.index("head_pitch")] = head_pitch
    return RobotState(
        time=0.0,
        joints=JointState(
            names=list(HEAD_NAMES),
            positions=positions,
            velocities=[0.0] * len(HEAD_NAMES),
            torques=[0.0] * len(HEAD_NAMES),
        ),
        imu=IMUState(),
        base_position_xyz=[0.0, 0.0, z],
        actuator_ctrl=[0.0] * len(HEAD_NAMES),
    )


class _FakeRobotApiClient:
    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.index = 0
        self.closed = False

    def read_state(self) -> RobotState:
        return _state(z=0.30)

    def step_motor_command(self, command):  # type: ignore[no-untyped-def]
        self.index += 1
        by_name = dict(zip(command.names, command.positions))
        # Drop below the default fall threshold after the first step so the
        # result records fall telemetry without needing a live simulator.
        z = 0.10 if self.index >= 2 else 0.30
        return _state(z=z, head_yaw=float(by_name.get("head_yaw", 0.0)), head_pitch=float(by_name.get("head_pitch", 0.0)))

    def close(self) -> None:
        self.closed = True


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def test_scripted_social_acceptance_dry_run_passes_default_gates() -> None:
    summary = run_acceptance(execute=False, control_hz=20.0)

    assert summary.ok is True
    assert summary.executed is False
    by_skill = {result.skill_id: result for result in summary.results}
    assert set(by_skill) == {"look_direction", "nod_yes", "shake_no"}
    assert by_skill["shake_no"].commanded_ranges["head_yaw"]["min"] <= -0.25
    assert by_skill["shake_no"].commanded_ranges["head_yaw"]["max"] >= 0.25
    assert by_skill["shake_no"].commanded_ranges["head_pitch"]["range"] == pytest.approx(0.0)
    assert by_skill["nod_yes"].commanded_ranges["head_pitch"]["min"] <= -0.16
    assert by_skill["nod_yes"].commanded_ranges["head_yaw"]["range"] == pytest.approx(0.0)


def test_scripted_social_acceptance_can_filter_skill() -> None:
    summary = run_acceptance(skill_ids=["shake_no"], execute=False, control_hz=20.0)

    assert summary.ok is True
    assert [result.skill_id for result in summary.results] == ["shake_no"]


def test_scripted_social_acceptance_rejects_observed_gate_without_execute() -> None:
    summary = run_acceptance(skill_ids=["shake_no"], execute=False, require_observed=True)

    assert summary.ok is False
    assert "--require-observed" in summary.results[0].errors[0]


def test_scripted_head_live_result_reports_base_height_fall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scripted_head_skill, "_load_robot_api_client_class", lambda: _FakeRobotApiClient)
    plan = _registry().create_plan("shake_no", {"count": 2, "amplitude": "small", "duration_s": 4.0})

    result = execute_scripted_head_plan(plan, dry_run=False, control_hz=10.0, fall_height_m=0.14)

    assert result.executed is True
    assert result.fallen is True
    assert result.observed_min_base_height_m == pytest.approx(0.10)
    assert result.final_base_height_m == pytest.approx(0.10)
    assert result.fall_height_m == pytest.approx(0.14)


def test_scripted_social_acceptance_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_social_acceptance",
            "--skill",
            "shake_no",
            "--control-hz",
            "20",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["results"][0]["skill_id"] == "shake_no"
    assert payload["results"][0]["commanded_ranges"]["head_yaw"]["range"] > 0.5


def test_scripted_social_acceptance_shell_wrapper_help() -> None:
    proc = subprocess.run(
        ["bash", "scripts/evaluate_scripted_social_skills.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert "--execute" in proc.stdout
    assert "run_sim_server.sh --backend mujoco" in proc.stdout
