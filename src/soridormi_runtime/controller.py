from __future__ import annotations

from soridormi_api import MotorCommand, RobotState


class HoldPositionController:
    """Safe starter controller.

    It commands the current joint positions with PD gains. Replace this with your walking
    controller/policy loop once the API is stable.
    """

    def __init__(self, kp: float = 5.0, kd: float = 0.1) -> None:
        self.kp = kp
        self.kd = kd

    def compute(self, state: RobotState) -> MotorCommand:
        n = len(state.joints.names)
        return MotorCommand(
            names=state.joints.names,
            positions=list(state.joints.positions),
            velocities=[0.0] * n,
            kp=[self.kp] * n,
            kd=[self.kd] * n,
            torques=[0.0] * n,
        )
