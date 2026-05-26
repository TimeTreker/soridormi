from __future__ import annotations

import json
from pathlib import Path

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.inspect_log import summarize_log
from soridormi_runtime.logging.jsonl_logger import JsonlRuntimeLogger


def _state() -> RobotState:
    return RobotState(
        time=1.23,
        joints=JointState(
            names=["left_hip_pitch", "right_hip_pitch"],
            positions=[0.1, -0.1],
            velocities=[0.0, 0.0],
            torques=[0.0, 0.0],
        ),
        imu=IMUState(),
    )


def _command() -> MotorCommand:
    return MotorCommand(
        names=["left_hip_pitch", "right_hip_pitch"],
        positions=[0.2, -0.2],
        velocities=[0.0, 0.0],
        kp=[10.0, 10.0],
        kd=[0.5, 0.5],
        torques=[0.0, 0.0],
    )


def test_jsonl_runtime_logger_writes_runtime_steps(tmp_path: Path) -> None:
    logger = JsonlRuntimeLogger(log_dir=tmp_path, every_n=1)
    logger.log_step(
        step_index=0,
        state=_state(),
        command=_command(),
        mode="stand",
        backend="sim",
    )
    logger.close()

    assert logger.path is not None
    lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["type"] == "runtime_step"
    assert payload["mode"] == "stand"
    assert payload["backend"] == "sim"
    assert payload["state"]["joints"]["names"] == ["left_hip_pitch", "right_hip_pitch"]
    assert payload["command"]["positions"] == [0.2, -0.2]


def test_inspect_log_summarizes_jsonl(tmp_path: Path) -> None:
    logger = JsonlRuntimeLogger(log_dir=tmp_path, every_n=1)
    logger.log_step(
        step_index=0,
        state=_state(),
        command=_command(),
        mode="stand",
        backend="sim",
    )
    logger.close()

    summary = summarize_log(logger.path)
    assert summary["format"] == "jsonl"
    assert summary["messages"] == 1
    assert summary["topics"] == {"runtime_step": 1}
    assert summary["min_robot_time"] == 1.23
    assert summary["max_robot_time"] == 1.23
