from __future__ import annotations

from pathlib import Path
from typing import Any

from soridormi_runtime.policy_profiles import PolicyProfile


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


def write_profile(path: Path, *, observation_size: int = 101, input_shape: str = "[1, 101]") -> None:
    path.write_text(
        f"""
name: replacement_test
description: replacement profile contract gate unit test
contract:
  observation_size: {observation_size}
  action_size: 14
model:
  path: /models/replacement.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: {input_shape}
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


def successful_fake_model_result(checker: Any, policy_path: str = "/models/replacement.onnx") -> Any:
    return checker.PolicyCheckResult(
        ok=True,
        policy_path=policy_path,
        providers=["CPUExecutionProvider"],
        input_name="obs",
        input_shape=[1, 101],
        input_type="tensor(float)",
        output_name="continuous_actions",
        output_shape=[1, 14],
        output_type="tensor(float)",
        errors=[],
        warnings=[],
    )


def test_profile_model_check_fails_when_static_contract_fails(tmp_path: Path, monkeypatch: Any) -> None:
    from soridormi_runtime import check_policy_model as checker

    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile_path, observation_size=100)
    monkeypatch.setattr(checker, "check_policy_model", lambda *args, **kwargs: successful_fake_model_result(checker))

    result = checker.check_profile_model(
        PolicyProfile.load(profile_path),
        robot_config_path=robot_config,
    )

    assert not result.ok
    assert result.contract_ok is False
    assert any("observation_size" in error for error in result.contract_errors or [])
    assert any("observation_size" in error for error in result.errors)


def test_profile_model_check_runs_onnx_gate_with_profile_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    from soridormi_runtime import check_policy_model as checker

    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile_path)
    captured: dict[str, Any] = {}

    def fake_check_policy_model(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return successful_fake_model_result(checker, policy_path=str(args[0]))

    monkeypatch.setattr(checker, "check_policy_model", fake_check_policy_model)

    result = checker.check_profile_model(
        PolicyProfile.load(profile_path),
        robot_config_path=robot_config,
    )

    assert result.ok
    assert result.contract_ok is True
    assert captured["args"] == ("/models/replacement.onnx",)
    assert captured["kwargs"]["expected_input_name"] == "obs"
    assert captured["kwargs"]["expected_output_name"] == "continuous_actions"
    assert captured["kwargs"]["expected_input_shape"] == [1, 101]
    assert captured["kwargs"]["expected_output_shape"] == [1, 14]
    assert result.profile_name == "replacement_test"
    assert result.robot_config_path == str(robot_config)


def test_check_policy_model_reports_missing_onnxruntime_without_import_failure(tmp_path: Path, monkeypatch: Any) -> None:
    from soridormi_runtime import check_policy_model as checker

    fake_model = tmp_path / "fake.onnx"
    fake_model.write_bytes(b"not a real onnx file")
    monkeypatch.setattr(checker, "ort", None)

    result = checker.check_policy_model(fake_model)

    assert not result.ok
    assert any("onnxruntime is not installed" in error for error in result.errors)
