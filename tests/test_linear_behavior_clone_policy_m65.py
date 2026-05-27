from __future__ import annotations

from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.check_policy_model import check_profile_model
from soridormi_runtime.create_linear_bc_profile import create_linear_bc_profile
from soridormi_runtime.linear_behavior_clone_policy import LinearBehaviorClonePolicy, load_linear_behavior_clone_model
from soridormi_runtime.observation_builder import ObservationBuilder, ObservationBuilderConfig
from soridormi_runtime.policy_profiles import PolicyProfile


JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def _state() -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
    )


def _write_linear_model(path: Path, *, action_value: float = 0.25) -> None:
    np.savez(
        path,
        weights=np.zeros((101, 14), dtype=np.float32),
        bias=np.ones(14, dtype=np.float32) * action_value,
        observation_mean=np.zeros(101, dtype=np.float32),
        observation_std=np.ones(101, dtype=np.float32),
        action_mean=np.zeros(14, dtype=np.float32),
        action_std=np.ones(14, dtype=np.float32),
    )


def test_linear_behavior_clone_policy_computes_action_and_updates_history(tmp_path: Path) -> None:
    model = tmp_path / "linear_behavior_clone.npz"
    _write_linear_model(model, action_value=0.125)
    builder = ObservationBuilder(ObservationBuilderConfig(joint_names=JOINTS))
    policy = LinearBehaviorClonePolicy(model, observation_builder=builder)

    action = policy.compute_action(_state())

    assert action.shape == (14,)
    assert np.allclose(action, 0.125)
    assert policy.get_observation() is not None
    assert np.allclose(builder.last_action, 0.125)


def test_linear_behavior_clone_model_validation_reports_bad_shapes(tmp_path: Path) -> None:
    model = tmp_path / "bad.npz"
    np.savez(
        model,
        weights=np.zeros((100, 14), dtype=np.float32),
        bias=np.zeros(14, dtype=np.float32),
        observation_mean=np.zeros(101, dtype=np.float32),
        observation_std=np.ones(101, dtype=np.float32),
        action_mean=np.zeros(14, dtype=np.float32),
        action_std=np.ones(14, dtype=np.float32),
    )

    result = load_linear_behavior_clone_model(model)

    assert not result.ok
    assert any("weights" in error for error in result.errors)


def test_create_linear_behavior_clone_profile_and_check_model(tmp_path: Path) -> None:
    model = tmp_path / "linear_behavior_clone.npz"
    _write_linear_model(model)
    profile_path = tmp_path / "linear_bc.yaml"

    created = create_linear_bc_profile(
        name="linear_bc_ci",
        model=model,
        output_path=profile_path,
        description="Linear BC test profile",
        robot_config_path="configs/robots/open_duck_mini_v2.yaml",
    )

    assert created.path == profile_path
    profile = PolicyProfile.load(profile_path)
    assert profile.model.kind == "linear_behavior_clone"
    assert profile.env()["SORIDORMI_POLICY_BACKEND"] == "linear_behavior_clone"

    checked = check_profile_model(profile, robot_config_path="configs/robots/open_duck_mini_v2.yaml")
    assert checked.ok
    assert checked.input_shape == [1, 101]
    assert checked.output_shape == [1, 14]
