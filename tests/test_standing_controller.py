from __future__ import annotations

from pathlib import Path

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.standing_controller import StandingPoseController


def test_standing_controller_uses_default_pose(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: left_hip_pitch
  - name: right_hip_pitch
default_pose:
  positions:
    left_hip_pitch: -0.15
    right_hip_pitch: 0.15
  gains:
    kp_default: 12.0
    kd_default: 0.7
  torque_default: 0.0
""",
        encoding="utf-8",
    )

    controller = StandingPoseController(config_path=config)

    state = RobotState(
        time=0.0,
        joints=JointState(
            names=["left_hip_pitch", "right_hip_pitch", "unknown_joint"],
            positions=[0.0, 0.0, 0.42],
            velocities=[0.0, 0.0, 0.0],
            torques=[0.0, 0.0, 0.0],
        ),
        imu=IMUState(),
    )

    command = controller.compute(state)

    assert command.names == ["left_hip_pitch", "right_hip_pitch", "unknown_joint"]
    assert command.positions == [-0.15, 0.15, 0.42]
    assert command.kp == [12.0, 12.0, 12.0]
    assert command.kd == [0.7, 0.7, 0.7]
