from __future__ import annotations

from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.observation_builder import ObservationBuilder, ObservationBuilderConfig


JOINTS = [
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

JOINT_OFFSET_START = 3 + 3 + 7
MOTOR_TARGET_START = JOINT_OFFSET_START + 14 + 14 + 14 + 14 + 14


def _state_at_defaults(defaults: dict[str, float]) -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[float(defaults[name]) for name in JOINTS],
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            accel_xyz=[0.0, 0.0, 0.0],
        ),
    )


def test_motor_target_updates_do_not_mutate_joint_offset_defaults() -> None:
    defaults = {name: 0.0 for name in JOINTS}

    # Reproduce the old alias-prone construction path: the same object is passed
    # as default_positions_by_name and motor_targets_by_name.
    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=JOINTS,
            default_positions_by_name=defaults,
            motor_targets_by_name=defaults,
        )
    )

    builder.set_motor_targets_by_name({"left_ankle": 0.1048, "neck_pitch": 0.1048})
    obs = builder.build(_state_at_defaults({name: 0.0 for name in JOINTS}))

    joint_offsets = obs[JOINT_OFFSET_START : JOINT_OFFSET_START + 14]
    motor_targets = obs[MOTOR_TARGET_START : MOTOR_TARGET_START + 14]

    np.testing.assert_allclose(joint_offsets, np.zeros(14, dtype=np.float32))
    assert motor_targets[JOINTS.index("left_ankle")] == np.float32(0.1048)
    assert motor_targets[JOINTS.index("neck_pitch")] == np.float32(0.1048)
    assert builder.config.default_positions_by_name["left_ankle"] == 0.0
    assert builder.config.default_positions_by_name["neck_pitch"] == 0.0


def test_from_robot_config_uses_distinct_default_and_motor_target_dicts(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    actuator_lines = "\n".join(f"  - name: {name}" for name in JOINTS)
    default_lines = "\n".join(f"    {name}: 0.0" for name in JOINTS)
    robot_config.write_text(
        f"""
robot_name: alias_test
model:
  path: /tmp/fake.xml
actuators:
{actuator_lines}
default_pose:
  positions:
{default_lines}
""",
        encoding="utf-8",
    )

    builder = ObservationBuilder.from_robot_config(robot_config)

    assert builder.config.default_positions_by_name is not builder.config.motor_targets_by_name

    builder.set_motor_targets_by_name({"left_ankle": 0.1048})

    assert builder.config.default_positions_by_name["left_ankle"] == 0.0
    assert builder.config.motor_targets_by_name["left_ankle"] == 0.1048
