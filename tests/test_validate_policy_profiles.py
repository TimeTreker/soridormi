from __future__ import annotations

from pathlib import Path
from typing import Any

from soridormi_runtime.check_policy_model import PolicyCheckResult
from soridormi_runtime.validate_policy_profiles import validate_policy_profiles


JOINT_NAMES = [
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


def write_robot_config(path: Path) -> None:
    actuators = "\n".join(f"  - name: {name}\n    ctrlrange: [-2.0, 2.0]" for name in JOINT_NAMES)
    positions = "\n".join(f"    {name}: 0.0" for name in JOINT_NAMES)
    path.write_text(
        f"""
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
{actuators}
default_pose:
  positions:
{positions}
  gains:
    kp_default: 12.0
    kd_default: 0.7
policy_observation:
  accelerometer_bias_xyz: [1.3, 0.0, 0.0]
  use_state_feet_contacts: true
action_mapping:
  action_scale: 0.2
  max_motor_velocity: 4.0
  kp_default: 11.0
  kd_default: 0.6
  torque_default: 0.0
  clip_to_limits: true
""",
        encoding="utf-8",
    )


def write_profile(path: Path, *, name: str, observation_size: int = 101) -> None:
    path.write_text(
        f"""
name: {name}
description: profile-suite validation test
contract:
  observation_size: {observation_size}
  action_size: 14
model:
  path: /models/{name}.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: [1, 101]
  output_shape: [1, 14]
  input_type: tensor(float)
  output_type: tensor(float)
action_mapping:
  action_scale: 0.25
  max_motor_velocity: 5.24
phase:
  mode: step
  period_steps: auto
""",
        encoding="utf-8",
    )


def test_validate_policy_profiles_static_suite_passes_for_valid_profiles(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_a = tmp_path / "a.yaml"
    profile_b = tmp_path / "b.yaml"
    write_robot_config(robot_config)
    write_profile(profile_a, name="profile_a")
    write_profile(profile_b, name="profile_b")

    result = validate_policy_profiles(
        [profile_a, profile_b],
        robot_config_path=robot_config,
    )

    assert result.ok
    assert result.profile_count == 2
    assert result.model_checked_count == 0
    assert [item.name for item in result.results] == ["profile_a", "profile_b"]
    assert all(item.contract_ok for item in result.results)
    assert all(not item.model_checked for item in result.results)


def test_validate_policy_profiles_reports_static_contract_failures(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "bad.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="bad_profile", observation_size=100)

    result = validate_policy_profiles([profile], robot_config_path=robot_config)

    assert not result.ok
    assert result.results[0].name == "bad_profile"
    assert not result.results[0].contract_ok
    assert any("observation_size" in error for error in result.results[0].errors)


def test_validate_policy_profiles_detects_duplicate_profile_names(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_a = tmp_path / "a.yaml"
    profile_b = tmp_path / "b.yaml"
    write_robot_config(robot_config)
    write_profile(profile_a, name="duplicate")
    write_profile(profile_b, name="duplicate")

    result = validate_policy_profiles([profile_a, profile_b], robot_config_path=robot_config)

    assert not result.ok
    assert "Duplicate policy profile name: duplicate" in result.errors


def test_validate_policy_profiles_optionally_runs_model_gate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from soridormi_runtime import validate_policy_profiles as validator

    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="model_profile")
    captured: dict[str, Any] = {}

    def fake_check_profile_model(*args: Any, **kwargs: Any) -> PolicyCheckResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PolicyCheckResult(
            ok=True,
            policy_path="/models/model_profile.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            input_name="obs",
            input_shape=[1, 101],
            input_type="tensor(float)",
            output_name="continuous_actions",
            output_shape=[1, 14],
            output_type="tensor(float)",
            errors=[],
            warnings=[],
            profile_name="model_profile",
            profile_path=str(profile),
            robot_config_path=str(robot_config),
            contract_ok=True,
            contract_errors=[],
            contract_warnings=[],
        )

    monkeypatch.setattr(validator, "check_profile_model", fake_check_profile_model)

    result = validate_policy_profiles(
        [profile],
        robot_config_path=robot_config,
        check_models=True,
        require_providers=["CUDAExecutionProvider"],
    )

    assert result.ok
    assert result.model_checked_count == 1
    assert result.results[0].model_checked
    assert result.results[0].providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert captured["kwargs"]["require_providers"] == ["CUDAExecutionProvider"]
