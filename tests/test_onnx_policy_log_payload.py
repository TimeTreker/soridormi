from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController
from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand


JOINTS = [f"joint_{i}" for i in range(14)]


class FakePolicy:
    def __init__(self) -> None:
        self.last_observation_stats = {"shape": [1, 101], "min": -0.5, "max": 0.5}

    def set_command_vector(self, command_vector):
        self.command_vector = list(command_vector)

    def set_imitation_phase(self, phase_vector):
        self.phase_vector = list(phase_vector)

    def set_motor_targets(self, joint_names, positions):
        self.motor_targets = dict(zip(joint_names, positions))

    def compute_action(self, state):
        return np.linspace(-0.7, 0.6, 14, dtype=np.float32)

    def get_observation_stats(self):
        return dict(self.last_observation_stats)


class FakeMapper:
    def __init__(self) -> None:
        self.last_motor_targets_by_name = {}

        class Config:
            action_scale = 0.1
            max_motor_velocity = 3.0
            speed_limit_enabled = True

        self.config = Config()

    def action_to_command(self, action, state=None, dt=None):
        positions = [float(x) * 0.1 for x in action]
        self.last_motor_targets_by_name = dict(zip(state.joints.names, positions))
        return MotorCommand(
            names=list(state.joints.names),
            positions=positions,
            velocities=[0.0] * 14,
            kp=[1.0] * 14,
            kd=[0.1] * 14,
            torques=[0.0] * 14,
        )


def make_state() -> RobotState:
    return RobotState(
        time=1.25,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
    )


def test_onnx_policy_controller_exposes_policy_log_payload() -> None:
    controller = OnnxPolicyController(
        policy_path="/tmp/fake.onnx",
        policy=FakePolicy(),
        mapper=FakeMapper(),
        command=PolicyCommand(x_velocity=0.01),
        phase_generator=GaitPhaseGenerator(frequency_hz=1.0, start_time=10.0),
        control_hz=50.0,
    )

    controller.compute(make_state())
    payload = controller.get_policy_log_payload()

    assert payload["policy_action"] is not None
    assert len(payload["policy_action"]) == 14
    assert payload["policy_debug"]["command"][0] == 0.01
    assert payload["policy_debug"]["action_scale"] == 0.1
    assert payload["policy_debug"]["max_motor_velocity"] == 3.0
    assert payload["policy_debug"]["speed_limit_enabled"] is True
    assert payload["policy_observation_stats"] == {"shape": [1, 101], "min": -0.5, "max": 0.5}
