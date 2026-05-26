from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController


JOINTS = [f"joint_{i}" for i in range(14)]


class ResettablePolicy:
    def __init__(self) -> None:
        self.reset_count = 0

    def compute_action(self, state: RobotState) -> np.ndarray:
        return np.zeros(14, dtype=np.float32)

    def set_command_vector(self, command):
        pass

    def set_imitation_phase(self, phase):
        pass

    def set_motor_targets(self, names, positions):
        pass

    def get_observation_stats(self):
        return None

    def reset_state(self) -> None:
        self.reset_count += 1


class ResettableMapper:
    def __init__(self) -> None:
        self.reset_count = 0

    def action_to_command(self, action, state=None, dt=None):
        return MotorCommand(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            kp=[0.0] * 14,
            kd=[0.0] * 14,
            torques=[0.0] * 14,
        )

    def reset_targets(self) -> None:
        self.reset_count += 1


def make_state(t: float) -> RobotState:
    return RobotState(
        time=t,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
        feet_contacts=[0.0, 0.0],
    )


def test_controller_resets_policy_state_when_sim_time_rewinds() -> None:
    policy = ResettablePolicy()
    mapper = ResettableMapper()
    controller = OnnxPolicyController(
        policy_path="/tmp/fake.onnx",
        policy=policy,
        mapper=mapper,
    )

    controller.compute(make_state(1.0))
    controller.compute(make_state(0.02))

    assert policy.reset_count == 1
    assert mapper.reset_count == 1
    assert controller.last_policy_debug is not None
    assert controller.last_policy_debug["sim_time_rewind_reset"] is True
