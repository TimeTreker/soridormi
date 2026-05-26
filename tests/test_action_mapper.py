from __future__ import annotations

from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.action_mapper import PolicyActionMapper
from soridormi_runtime.observation_builder import ObservationBuilder, ObservationBuilderConfig


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


def make_state() -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=list(JOINT_NAMES),
            positions=[0.0] * len(JOINT_NAMES),
            velocities=[0.0] * len(JOINT_NAMES),
            torques=[0.0] * len(JOINT_NAMES),
        ),
        imu=IMUState(),
    )


def write_robot_config(path: Path) -> None:
    actuator_block = "\n".join(f"  - name: {name}" for name in JOINT_NAMES)
    position_block = "\n".join(f"    {name}: 0.0" for name in JOINT_NAMES)
    path.write_text(
        f"""
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
{actuator_block}
default_pose:
  positions:
{position_block}
  gains:
    kp_default: 12.0
    kd_default: 0.7
action_mapping:
  action_scale: 0.25
  kp_default: 11.0
  kd_default: 0.6
  torque_default: 0.0
  clip_to_limits: true
""",
        encoding="utf-8",
    )


def test_action_mapper_converts_action_to_motor_command(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    write_robot_config(config)

    mapper = PolicyActionMapper.from_robot_config(config)
    action = np.ones(14, dtype=np.float32)

    command = mapper.action_to_command(action, state=make_state())

    assert command.names == JOINT_NAMES
    assert command.positions == [0.25] * 14
    assert command.kp == [11.0] * 14
    assert command.kd == [0.6] * 14
    assert command.torques == [0.0] * 14


def test_action_mapper_accepts_batched_action(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    write_robot_config(config)

    mapper = PolicyActionMapper.from_robot_config(config)
    action = np.ones((1, 14), dtype=np.float32) * 2.0

    targets = mapper.action_to_targets(action)

    assert list(targets) == JOINT_NAMES
    assert all(value == 0.5 for value in targets.values())


def test_observation_builder_can_update_motor_targets() -> None:
    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=list(JOINT_NAMES),
            default_positions_by_name={name: 0.0 for name in JOINT_NAMES},
        )
    )

    builder.set_motor_targets(JOINT_NAMES, np.ones(14, dtype=np.float32) * 0.123)
    obs = builder.build(make_state())

    # motor_targets start after:
    # gyro 3, accel 3, command 7, joint offsets 14, joint velocities 14,
    # last_action 14, last_last_action 14, last_last_last_action 14
    motor_targets_start = 3 + 3 + 7 + 14 + 14 + 14 + 14 + 14

    np.testing.assert_allclose(
        obs[motor_targets_start : motor_targets_start + 14],
        np.ones(14, dtype=np.float32) * 0.123,
    )
