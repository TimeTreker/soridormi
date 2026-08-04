from __future__ import annotations

from pathlib import Path

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.analyze_policy_log import analyze_policy_log


JOINTS = [f"joint_{i}" for i in range(14)]


def test_robot_state_accepts_optional_base_pose() -> None:
    state = RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
        base_position_xyz=[1.0, 2.0, 0.3],
        base_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )

    assert state.base_position_xyz == [1.0, 2.0, 0.3]
    assert state.base_quat_wxyz == [1.0, 0.0, 0.0, 0.0]


def test_analyze_jsonl_reports_base_displacement(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    path.write_text(
        """
{"type":"/soridormi/robot_state","step_index":0,"time_wall_ns":0,"state":{"time":0.0,"joints":{"names":["j"],"positions":[0.0],"velocities":[0.0],"torques":[0.0]},"imu":{"quat_wxyz":[1,0,0,0],"gyro_xyz":[0,0,0],"accel_xyz":[0,0,9.81]},"base_position_xyz":[0.0,0.0,0.3]}}
{"type":"/soridormi/policy_debug","step_index":0,"time_wall_ns":1,"debug":{"robot_time":0.0,"command":[0.05,0,0,0,0,0,0],"phase":[1,0],"speed_limit_enabled":true}}
{"type":"/soridormi/policy_action","step_index":0,"time_wall_ns":2,"action":[0.1,0.0]}
{"type":"/soridormi/robot_state","step_index":1,"time_wall_ns":1000000000,"state":{"time":1.0,"joints":{"names":["j"],"positions":[0.0],"velocities":[0.0],"torques":[0.0]},"imu":{"quat_wxyz":[1,0,0,0],"gyro_xyz":[0,0,0],"accel_xyz":[0,0,9.81]},"base_position_xyz":[0.12,0.01,0.28]}}
""".strip() + "\n",
        encoding="utf-8",
    )

    summary = analyze_policy_log(path)

    assert summary["base_displacement"]["available"] is True
    assert summary["base_displacement"]["forward_x"] == 0.12
    assert summary["base_displacement"]["lateral_y"] == 0.01
    assert any("Base displacement" in item for item in summary["diagnosis"])
