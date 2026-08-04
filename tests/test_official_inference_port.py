from __future__ import annotations

from types import SimpleNamespace

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.onnx_policy import OnnxPolicy
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController
from soridormi_runtime.policy_profiles import PolicyProfile


class DummyBuilder:
    def __init__(self) -> None:
        self.defaults = {}
        self.targets = {}
        self.config = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(14)])

    def set_default_positions_by_name(self, values):
        self.defaults.update(values)

    def set_motor_targets_by_name(self, values):
        self.targets.update(values)


class DummySessionItem:
    name = "obs"
    shape = [1, 101]
    type = "tensor(float)"


def make_state() -> RobotState:
    names = [f"joint_{i}" for i in range(14)]
    return RobotState(
        time=0.0,
        joints=JointState(
            names=names,
            positions=[10.0 + i for i in range(14)],
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
        actuator_ctrl=[0.1 * i for i in range(14)],
    )


def test_onnx_policy_bootstrap_prefers_actuator_ctrl() -> None:
    policy = object.__new__(OnnxPolicy)
    policy.observation_builder = DummyBuilder()
    defaults = OnnxPolicy.bootstrap_defaults_from_state(policy, make_state())

    assert defaults["joint_0"] == 0.0
    assert defaults["joint_13"] == 1.3
    assert policy.observation_builder.defaults["joint_3"] == 0.30000000000000004
    assert policy.observation_builder.targets["joint_3"] == 0.30000000000000004


def test_controller_fallback_bootstrap_prefers_actuator_ctrl() -> None:
    controller = object.__new__(OnnxPolicyController)
    controller.bootstrap_policy_defaults_from_state = True
    controller.policy_defaults_bootstrapped = False
    controller.bootstrapped_defaults = {}
    controller.policy = object()
    controller.mapper = object()

    OnnxPolicyController._bootstrap_policy_defaults_from_state_once(controller, make_state())

    assert controller.bootstrapped_defaults["joint_1"] == 0.1
    assert controller.bootstrapped_defaults["joint_10"] == 1.0
    assert controller.policy_defaults_bootstrapped is True


def test_open_duck_forward_profile_requests_home_keyframe() -> None:
    profile = PolicyProfile.load("open_duck_forward")
    env = profile.env()

    assert env["SORIDORMI_MUJOCO_USE_HOME_KEYFRAME"] == "1"
    assert env["SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE"] == "1"
    assert env["SORIDORMI_SIM_BACKEND"] == "mujoco"
    assert env["SORIDORMI_COMMAND_X"] == "0.15"
    assert env["SORIDORMI_COMMAND_RAMP_SECONDS"] == "0"
