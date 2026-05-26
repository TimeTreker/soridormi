from __future__ import annotations

from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_robot_config_loads_optional_reset_pose(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: left_hip_pitch
  - name: right_hip_pitch
reset_pose:
  base:
    position_xyz: [0.0, 0.0, 0.12]
    quat_wxyz: [1.0, 0.0, 0.0, 0.0]
  joints:
    left_hip_pitch: -0.1
    right_hip_pitch: 0.1
default_pose:
  positions:
    left_hip_pitch: -0.1
    right_hip_pitch: 0.1
  gains:
    kp_default: 10.0
    kd_default: 0.5
  torque_default: 0.0
""",
        encoding="utf-8",
    )

    robot_config = load_robot_config(config)

    assert robot_config.reset_pose is not None
    assert robot_config.reset_pose.base is not None
    assert robot_config.reset_pose.base.position_xyz == [0.0, 0.0, 0.12]
    assert robot_config.reset_pose.base.quat_wxyz == [1.0, 0.0, 0.0, 0.0]
    assert robot_config.reset_pose.joints["left_hip_pitch"] == -0.1
    assert robot_config.default_pose is not None
    assert robot_config.default_pose.gains.kp_default == 10.0
