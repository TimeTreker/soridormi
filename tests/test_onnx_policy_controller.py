from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController


def make_state() -> RobotState:
    names = [f"joint_{i}" for i in range(14)]
    return RobotState(
        time=0.0,
        joints=JointState(
            names=names,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
    )


class FakePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def compute_action(self, state: RobotState) -> np.ndarray:
        self.calls += 1
        return np.arange(14, dtype=np.float32) * 0.01


class FakeMapper:
    def __init__(self) -> None:
        self.last_action: np.ndarray | None = None

    def action_to_command(self, action, state: RobotState | None = None) -> MotorCommand:
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_action = action_arr.copy()
        assert state is not None
        return MotorCommand(
            names=list(state.joints.names),
            positions=[float(x) for x in action_arr],
            velocities=[0.0] * 14,
            kp=[1.0] * 14,
            kd=[0.1] * 14,
            torques=[0.0] * 14,
        )


def test_onnx_policy_controller_maps_policy_action_to_motor_command() -> None:
    policy = FakePolicy()
    mapper = FakeMapper()
    controller = OnnxPolicyController(
        policy_path="/tmp/fake.onnx",
        policy=policy,
        mapper=mapper,
    )

    state = make_state()
    command = controller.compute(state)

    assert policy.calls == 1
    assert mapper.last_action is not None
    assert command.names == state.joints.names
    assert command.positions[0] == 0.0
    assert command.positions[-1] == float(np.float32(0.13))
    assert controller.step_count == 1
    assert controller.last_action is not None
    assert controller.last_command is command
