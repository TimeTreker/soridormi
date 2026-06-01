from __future__ import annotations

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.sync_preroll import make_hold_current_command, preroll_sync_simulator
from soridormi_runtime.policy_profiles import PolicyProfile


def _state(*, time: float = 0.0, ctrl: list[float] | None = None) -> RobotState:
    return RobotState(
        time=time,
        joints=JointState(
            names=["left_hip", "right_hip"],
            positions=[0.1, -0.2],
            velocities=[0.0, 0.0],
            torques=[0.0, 0.0],
        ),
        imu=IMUState(),
        actuator_ctrl=ctrl,
    )


class FakeSyncRobot:
    def __init__(self) -> None:
        self.calls = 0
        self.commands = []

    def step_motor_command(self, command):
        self.calls += 1
        self.commands.append(command)
        return _state(time=float(self.calls), ctrl=list(command.positions))


def test_hold_current_command_prefers_actuator_ctrl() -> None:
    command = make_hold_current_command(_state(ctrl=[0.3, -0.4]))

    assert command.names == ["left_hip", "right_hip"]
    assert command.positions == [0.3, -0.4]
    assert command.kp == [10.0, 10.0]
    assert command.kd == [0.5, 0.5]


def test_hold_current_command_falls_back_to_joint_positions() -> None:
    command = make_hold_current_command(_state(ctrl=None))

    assert command.positions == [0.1, -0.2]


def testpreroll_sync_simulator_steps_hold_current_command() -> None:
    robot = FakeSyncRobot()

    result = preroll_sync_simulator(robot, _state(ctrl=[0.3, -0.4]), 2)

    assert robot.calls == 2
    assert result.time == 2.0
    assert robot.commands[0].positions == [0.3, -0.4]
    assert robot.commands[1].positions == [0.3, -0.4]


def test_open_duck_forward_exports_one_sync_preroll_step() -> None:
    env = PolicyProfile.load("open_duck_forward").env()

    assert env["SORIDORMI_SIM_SYNC_STEP"] == "1"
    assert env["SORIDORMI_SIM_PREROLL_STEPS"] == "1"
