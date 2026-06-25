from __future__ import annotations

from soridormi_api import MotorCommand, RobotState, VisualExpressionCommand


class HardwareRobot:
    """Real hardware backend placeholder.

    Replace this with motor bus, IMU, encoder, and power-board integration.
    Keep this class API identical to SimRobot.
    """

    def read_state(self) -> RobotState:
        raise NotImplementedError("HardwareRobot.read_state is not implemented yet")

    def send_motor_command(self, command: MotorCommand) -> None:
        raise NotImplementedError("HardwareRobot.send_motor_command is not implemented yet")

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        raise NotImplementedError("HardwareRobot.set_visual_expression is not implemented yet")
