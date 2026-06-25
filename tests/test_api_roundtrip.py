from __future__ import annotations

import threading
import time

from soridormi_api.client import RobotApiClient
from soridormi_api.server import RobotApiServer
from soridormi_api.types import IMUState, JointState, MotorCommand, RobotState, VisualExpressionCommand


class FakeBackend:
    def __init__(self) -> None:
        self.step_count = 0
        self.last_command: MotorCommand | None = None
        self.last_visual_expression: VisualExpressionCommand | None = None

    def step(self) -> None:
        self.step_count += 1

    def get_state(self) -> RobotState:
        return RobotState(
            time=float(self.step_count) * 0.02,
            joints=JointState(
                names=["left_hip", "right_hip"],
                positions=[0.1, -0.1],
                velocities=[0.0, 0.0],
                torques=[0.0, 0.0],
            ),
            imu=IMUState(
                quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                gyro_xyz=[0.0, 0.0, 0.0],
                accel_xyz=[0.0, 0.0, 9.81],
            ),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command

    def apply_visual_expression(self, command: VisualExpressionCommand) -> None:
        self.last_visual_expression = command


def test_api_roundtrip_read_state_and_send_command() -> None:
    backend = FakeBackend()
    port = 5566

    server = RobotApiServer(backend=backend, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    time.sleep(0.2)

    client = RobotApiClient(host="127.0.0.1", port=port, timeout_ms=1000)

    try:
        assert client.ping() == "soridormi-sim alive"

        state = client.read_state()
        assert state.joints.names == ["left_hip", "right_hip"]
        assert state.joints.positions == [0.1, -0.1]
        assert backend.step_count >= 1

        command = MotorCommand(
            names=["left_hip", "right_hip"],
            positions=[0.2, -0.2],
            velocities=[0.0, 0.0],
            kp=[20.0, 20.0],
            kd=[1.0, 1.0],
            torques=[0.0, 0.0],
        )

        client.send_motor_command(command)

        assert backend.last_command is not None
        assert backend.last_command.positions == [0.2, -0.2]
        assert backend.last_command.kp == [20.0, 20.0]

        message = client.set_visual_expression(
            VisualExpressionCommand(expression="eyes_closed", intensity=0.75)
        )

        assert message == "visual expression applied: eyes_closed"
        assert backend.last_visual_expression is not None
        assert backend.last_visual_expression.expression == "eyes_closed"
        assert backend.last_visual_expression.intensity == 0.75

    finally:
        client.close()
