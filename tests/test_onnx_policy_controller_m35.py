from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController
from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand


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


class FakePolicy:
    def __init__(self) -> None:
        self.command_vector = None
        self.phase_vector = None
        self.motor_targets = None

    def set_command_vector(self, command_vector):
        self.command_vector = list(command_vector)

    def set_imitation_phase(self, phase_vector):
        self.phase_vector = list(phase_vector)

    def set_motor_targets(self, joint_names, positions):
        self.motor_targets = dict(zip(joint_names, positions))

    def compute_action(self, state):
        return np.ones(14, dtype=np.float32) * 0.5


class FakeMapper:
    def __init__(self) -> None:
        self.last_motor_targets_by_name = {}
        self.last_dt = None

    def action_to_command(self, action, state=None, dt=None):
        from soridormi_api import MotorCommand

        self.last_dt = dt
        names = list(state.joints.names)
        positions = [0.1 for _ in names]
        self.last_motor_targets_by_name = dict(zip(names, positions))
        return MotorCommand(
            names=names,
            positions=positions,
            velocities=[0.0] * len(names),
            kp=[1.0] * len(names),
            kd=[0.1] * len(names),
            torques=[0.0] * len(names),
        )


def make_state() -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
    )


def test_onnx_policy_controller_sets_command_phase_and_motor_targets() -> None:
    policy = FakePolicy()
    mapper = FakeMapper()

    controller = OnnxPolicyController(
        policy_path="/tmp/fake.onnx",
        policy=policy,
        mapper=mapper,
        command=PolicyCommand(x_velocity=0.05, yaw_velocity=0.1),
        phase_generator=GaitPhaseGenerator(frequency_hz=1.0, start_time=10.0),
        control_hz=50.0,
    )

    command = controller.compute(make_state())

    assert policy.command_vector == [0.05, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    assert policy.phase_vector is not None
    assert mapper.last_dt == 0.02
    assert policy.motor_targets is not None
    assert command.positions == [0.1] * 14
