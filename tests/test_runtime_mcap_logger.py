from __future__ import annotations

from pathlib import Path

import pytest

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.inspect_log import summarize_log


def _state() -> RobotState:
    return RobotState(
        time=2.0,
        joints=JointState(
            names=["left_knee", "right_knee"],
            positions=[0.1, -0.1],
            velocities=[0.0, 0.0],
            torques=[0.0, 0.0],
        ),
        imu=IMUState(),
    )


def _command() -> MotorCommand:
    return MotorCommand(
        names=["left_knee", "right_knee"],
        positions=[0.2, -0.2],
        velocities=[0.0, 0.0],
        kp=[10.0, 10.0],
        kd=[0.5, 0.5],
        torques=[0.0, 0.0],
    )


def test_mcap_runtime_logger_writes_robot_state_and_command(tmp_path: Path) -> None:
    pytest.importorskip("mcap")

    from soridormi_runtime.logging.mcap_logger import McapRuntimeLogger

    logger = McapRuntimeLogger(log_dir=tmp_path, every_n=1, mode="stand", backend="sim")
    logger.log_step(
        step_index=0,
        state=_state(),
        command=_command(),
        mode="stand",
        backend="sim",
    )
    logger.close()

    assert logger.path is not None
    assert logger.path.exists()
    assert logger.path.suffix == ".mcap"

    summary = summarize_log(logger.path)
    assert summary["format"] == "mcap"
    assert summary["topics"]["/soridormi/robot_state"] == 1
    assert summary["topics"]["/soridormi/motor_command"] == 1
    assert summary["topics"]["/soridormi/runtime_status"] == 1
