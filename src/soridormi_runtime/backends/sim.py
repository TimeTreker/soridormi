from __future__ import annotations

from soridormi_api import MotorCommand, RobotApiClient, RobotState, VisualExpressionCommand


class SimRobot:
    def __init__(self, host: str, port: int) -> None:
        self.client = RobotApiClient(host=host, port=port)

    def read_state(self) -> RobotState:
        return self.client.read_state()

    def send_motor_command(self, command: MotorCommand) -> None:
        self.client.send_motor_command(command)

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        return self.client.step_motor_command(command)

    def reset(self) -> str:
        return self.client.reset()

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        return self.client.set_visual_expression(command)
