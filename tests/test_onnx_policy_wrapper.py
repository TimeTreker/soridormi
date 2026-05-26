from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.observation_builder import ObservationBuilder, ObservationBuilderConfig
from soridormi_runtime.onnx_policy import OnnxPolicy


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


class FakeIo:
    def __init__(self, name: str, shape: list[int], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers
        self.last_feed = None

    def get_inputs(self):
        return [FakeIo("obs", [1, 101])]

    def get_outputs(self):
        return [FakeIo("continuous_actions", [1, 14])]

    def run(self, output_names, feed):
        self.last_feed = feed
        obs = feed["obs"]
        assert obs.shape == (1, 101)
        return [np.arange(14, dtype=np.float32).reshape(1, 14) * 0.01]


def make_state() -> RobotState:
    n = len(JOINT_NAMES)
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINT_NAMES,
            positions=[0.0] * n,
            velocities=[0.0] * n,
            torques=[0.0] * n,
        ),
        imu=IMUState(
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            accel_xyz=[0.0, 0.0, 9.81],
        ),
    )


def test_onnx_policy_compute_action_updates_history() -> None:
    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=JOINT_NAMES,
            default_positions_by_name={name: 0.0 for name in JOINT_NAMES},
        )
    )

    policy = OnnxPolicy(
        policy_path="/tmp/fake.onnx",
        providers=["CPUExecutionProvider"],
        observation_builder=builder,
        session_factory=lambda path, providers: FakeSession(path, providers),
    )

    action = policy.compute_action(make_state())

    assert action.shape == (14,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, np.arange(14, dtype=np.float32) * 0.01)
    np.testing.assert_allclose(policy.observation_builder.last_action, action)


def test_onnx_policy_describe_contains_metadata() -> None:
    builder = ObservationBuilder(
        ObservationBuilderConfig(
            joint_names=JOINT_NAMES,
            default_positions_by_name={name: 0.0 for name in JOINT_NAMES},
        )
    )

    policy = OnnxPolicy(
        policy_path="/tmp/fake.onnx",
        providers=["CPUExecutionProvider"],
        observation_builder=builder,
        session_factory=lambda path, providers: FakeSession(path, providers),
    )

    info = policy.describe()

    assert info["input_name"] == "obs"
    assert info["input_shape"] == [1, 101]
    assert info["output_name"] == "continuous_actions"
    assert info["output_shape"] == [1, 14]
    assert info["joint_names"] == JOINT_NAMES
