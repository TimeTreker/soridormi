from __future__ import annotations

import json
from pathlib import Path

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.inspect_log import summarize_log
from soridormi_runtime.logging.jsonl_logger import JsonlRuntimeLogger


def _state() -> RobotState:
    return RobotState(
        time=1.0,
        joints=JointState(
            names=["joint_0", "joint_1"],
            positions=[0.1, -0.1],
            velocities=[0.0, 0.0],
            torques=[0.0, 0.0],
        ),
        imu=IMUState(),
    )


def _command() -> MotorCommand:
    return MotorCommand(
        names=["joint_0", "joint_1"],
        positions=[0.2, -0.2],
        velocities=[0.0, 0.0],
        kp=[1.0, 1.0],
        kd=[0.1, 0.1],
        torques=[0.0, 0.0],
    )


def test_jsonl_runtime_logger_writes_policy_debug_payloads(tmp_path: Path) -> None:
    logger = JsonlRuntimeLogger(log_dir=tmp_path, every_n=1)
    logger.log_step(
        step_index=0,
        state=_state(),
        command=_command(),
        mode="onnx_policy",
        backend="sim",
        policy_action=[0.1] * 14,
        policy_debug={
            "command": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "phase": [1.0, 0.0],
            "action_min": 0.1,
            "action_max": 0.1,
        },
        policy_observation_stats={"shape": [1, 101], "min": -1.0, "max": 1.0},
    )
    logger.close()

    assert logger.path is not None
    payload = json.loads(logger.path.read_text(encoding="utf-8").strip())
    assert payload["policy_action"] == [0.1] * 14
    assert payload["policy_debug"]["command"][0] == 0.01
    assert payload["policy_observation_stats"]["shape"] == [1, 101]

    summary = summarize_log(logger.path)
    assert summary["topics"]["runtime_step"] == 1
    assert summary["topics"]["policy_debug"] == 1
    assert summary["topics"]["policy_action"] == 1
    assert summary["topics"]["policy_observation_stats"] == 1
