from __future__ import annotations

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.joint_sweep_controller import JointSweepController


def test_joint_sweep_controller_outputs_motor_command(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_JOINT_SWEEP_AMPLITUDE", "0.1")
    monkeypatch.setenv("SORIDORMI_JOINT_SWEEP_PERIOD_SECONDS", "2.0")
    monkeypatch.setenv("SORIDORMI_JOINT_SWEEP_HOLD_SECONDS", "0.0")
    monkeypatch.setenv("SORIDORMI_JOINT_SWEEP_KP", "9.0")
    monkeypatch.setenv("SORIDORMI_JOINT_SWEEP_KD", "0.4")

    controller = JointSweepController()

    state = RobotState(
        time=0.0,
        joints=JointState(
            names=["left_hip_pitch", "right_hip_pitch"],
            positions=[0.2, -0.2],
            velocities=[0.0, 0.0],
            torques=[0.0, 0.0],
        ),
        imu=IMUState(),
    )

    command = controller.compute(state)

    assert command.names == ["left_hip_pitch", "right_hip_pitch"]
    assert len(command.positions) == 2
    assert command.kp == [9.0, 9.0]
    assert command.kd == [0.4, 0.4]
    assert command.velocities == [0.0, 0.0]
    assert command.torques == [0.0, 0.0]
