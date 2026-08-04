from __future__ import annotations

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


def make_state() -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            accel_xyz=[0.0, 0.0, 9.81],
        ),
        feet_contacts=[1.0, 0.0],
    )


def test_observation_builder_applies_accel_bias_and_state_feet_contacts() -> None:
    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=JOINTS,
            accelerometer_bias_xyz=[1.3, 0.0, 0.0],
            use_state_feet_contacts=True,
        )
    )

    obs = builder.build(make_state())

    np.testing.assert_allclose(obs[3:6], np.array([1.3, 0.0, 9.81], dtype=np.float32))
    np.testing.assert_allclose(obs[97:99], np.array([1.0, 0.0], dtype=np.float32))


def test_robot_state_validates_two_feet_contacts() -> None:
    state = make_state()
    assert state.feet_contacts == [1.0, 0.0]
