from __future__ import annotations

import math

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController
from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand


JOINTS = [f"joint_{i}" for i in range(14)]


def make_state(time: float = 0.0) -> RobotState:
    return RobotState(
        time=time,
        joints=JointState(
            names=JOINTS,
            positions=[0.1 * i for i in range(14)],
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
    )


class BootstrapPolicy:
    def __init__(self) -> None:
        self.defaults = None
        self.commands = []
        self.phases = []

    def bootstrap_defaults_from_state(self, state):
        self.defaults = dict(zip(state.joints.names, state.joints.positions))
        return self.defaults

    def set_command_vector(self, command):
        self.commands.append(list(command))

    def set_imitation_phase(self, phase):
        self.phases.append(list(phase))

    def compute_action(self, state):
        return np.zeros(14, dtype=np.float32)


class BootstrapMapper:
    def __init__(self) -> None:
        self.defaults = None

    def set_default_positions_by_name(self, defaults):
        self.defaults = dict(defaults)

    def action_to_command(self, action, state=None, dt=None):
        return MotorCommand(
            names=list(state.joints.names),
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            kp=[1.0] * 14,
            kd=[0.1] * 14,
            torques=[0.0] * 14,
        )


def test_controller_bootstraps_policy_defaults_from_first_state(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_BOOTSTRAP_POLICY_DEFAULTS_FROM_STATE", "1")
    policy = BootstrapPolicy()
    mapper = BootstrapMapper()
    controller = OnnxPolicyController(
        policy_path="/tmp/fake.onnx",
        policy=policy,
        mapper=mapper,
        command=PolicyCommand(x_velocity=0.06),
        phase_generator=GaitPhaseGenerator(mode="step", period_steps=50),
    )

    controller.compute(make_state())

    assert policy.defaults is not None
    assert mapper.defaults is not None
    assert policy.defaults["joint_13"] == 1.3
    assert mapper.defaults["joint_13"] == 1.3
    assert controller.last_policy_debug["policy_defaults_bootstrapped"] is True
    assert controller.last_policy_debug["bootstrapped_default_count"] == 14


def test_step_phase_advances_once_per_compute() -> None:
    gen = GaitPhaseGenerator(mode="step", period_steps=4, step_increment=1.0)

    p0 = gen.advance_and_as_list()
    p1 = gen.advance_and_as_list()
    p2 = gen.advance_and_as_list()

    # Official Open Duck increments imitation_i before building the
    # observation, so the first policy phase is 1 / period, not 0 / period.
    assert abs(p0[0]) < 1e-6
    assert math.isclose(p0[1], 1.0, abs_tol=1e-6)
    assert math.isclose(p1[0], -1.0, abs_tol=1e-6)
    assert abs(p1[1]) < 1e-6
    assert abs(p2[0]) < 1e-6
    assert math.isclose(p2[1], -1.0, abs_tol=1e-6)
