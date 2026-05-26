from __future__ import annotations

from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.observation_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    load_default_pose_positions,
)


def make_state() -> RobotState:
    names = [
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

    positions = [0.1 * i for i in range(len(names))]
    velocities = [0.2 * i for i in range(len(names))]
    torques = [0.0 for _ in names]

    return RobotState(
        time=1.23,
        joints=JointState(
            names=names,
            positions=positions,
            velocities=velocities,
            torques=torques,
        ),
        imu=IMUState(
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.1, 0.2, 0.3],
            accel_xyz=[0.0, 0.0, 9.81],
        ),
    )


def test_observation_builder_returns_101_float32_values() -> None:
    state = make_state()

    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=list(state.joints.names),
            default_positions_by_name={name: 0.0 for name in state.joints.names},
        )
    )

    obs = builder.build(state)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (101,)
    assert obs.dtype == np.float32


def test_observation_builder_joint_offsets_and_velocity_scale() -> None:
    state = make_state()

    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=list(state.joints.names),
            default_positions_by_name={name: 0.0 for name in state.joints.names},
            dof_vel_scale=0.05,
        )
    )

    obs = builder.build(state)

    # Layout:
    # gyro_xyz: 3
    # accel_xyz: 3
    # command: 7
    # joint_offsets: 14
    # joint_velocities_scaled: 14
    joint_offset_start = 3 + 3 + 7
    joint_velocity_start = joint_offset_start + 14

    expected_offsets = np.array(state.joints.positions, dtype=np.float32)
    expected_velocities = np.array(state.joints.velocities, dtype=np.float32) * 0.05

    np.testing.assert_allclose(
        obs[joint_offset_start : joint_offset_start + 14],
        expected_offsets,
    )
    np.testing.assert_allclose(
        obs[joint_velocity_start : joint_velocity_start + 14],
        expected_velocities,
    )


def test_observation_builder_tracks_action_history() -> None:
    state = make_state()

    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=list(state.joints.names),
            default_positions_by_name={name: 0.0 for name in state.joints.names},
        )
    )

    first = builder.build(state)
    first_action = np.arange(14, dtype=np.float32) * 0.01
    builder.update_action_history(first_action)

    second = builder.build(state)

    # Layout:
    # gyro: 3
    # accel: 3
    # command: 7
    # joint offsets: 14
    # joint velocities: 14
    # last_action: 14
    last_action_start = 3 + 3 + 7 + 14 + 14

    np.testing.assert_allclose(
        first[last_action_start : last_action_start + 14],
        np.zeros(14, dtype=np.float32),
    )
    np.testing.assert_allclose(
        second[last_action_start : last_action_start + 14],
        first_action,
    )


def test_load_default_pose_positions_from_config(tmp_path: Path) -> None:
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
""",
        encoding="utf-8",
    )

    positions = load_default_pose_positions(config)

    assert positions == {
        "left_hip_pitch": -0.15,
        "right_hip_pitch": 0.15,
    }
