from __future__ import annotations

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_api.types import ApiRequest, ApiResponse
from soridormi_api.server import RobotApiServer
from soridormi_runtime.policy_profiles import PolicyProfile


class FakeBackend:
    def __init__(self) -> None:
        self.command_count = 0
        self.step_count = 0
        self.last_command = None

    def get_state(self) -> RobotState:
        return RobotState(
            time=float(self.step_count),
            joints=JointState(names=["j"], positions=[0.0], velocities=[0.0], torques=[0.0]),
            imu=IMUState(),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.command_count += 1
        self.last_command = command

    def step(self) -> None:
        self.step_count += 1


def make_command() -> MotorCommand:
    return MotorCommand(
        names=["j"],
        positions=[0.1],
        velocities=[0.0],
        kp=[0.0],
        kd=[0.0],
        torques=[0.0],
    )


def test_api_request_accepts_step_command() -> None:
    request = ApiRequest(kind="step_command", command=make_command())
    assert request.kind == "step_command"


def test_server_step_command_applies_steps_and_returns_state() -> None:
    backend = FakeBackend()
    server = RobotApiServer(backend=backend)

    response = server._handle(ApiRequest(kind="step_command", command=make_command()))

    assert isinstance(response, ApiResponse)
    assert response.ok
    assert response.state is not None
    assert response.state.time == 1.0
    assert backend.command_count == 1
    assert backend.step_count == 1


def test_policy_profile_exports_sync_step() -> None:
    profile = PolicyProfile.load("open_duck_forward")
    env = profile.env()
    assert env["SORIDORMI_SIM_SYNC_STEP"] == "1"
