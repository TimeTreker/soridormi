from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.policy_acceptance import accept_policy_profile


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


def write_profile(path: Path, *, name: str, model_path: str = "/models/replacement.onnx") -> None:
    path.write_text(
        f"""
name: {name}
description: acceptance test profile
contract:
  observation_size: 101
  action_size: 14
model:
  path: {model_path}
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


def test_accept_policy_profile_writes_artifact_bundle(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    output_dir = tmp_path / "acceptance"
    write_robot_config(robot_config)
    write_profile(profile, name="accepted_profile")

    result = accept_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=output_dir,
        profile_only=True,
    )

    assert result.ok
    assert result.profile_name == "accepted_profile"
    assert Path(result.artifacts.directory).is_dir()
    assert Path(result.artifacts.contract_json).is_file()
    assert Path(result.artifacts.manifest_json).is_file()
    assert Path(result.artifacts.suite_validation_json).is_file()
    assert Path(result.artifacts.report_markdown).is_file()
    assert (Path(result.artifacts.directory) / "acceptance.json").is_file()

    manifest = json.loads(Path(result.artifacts.manifest_json).read_text(encoding="utf-8"))
    assert manifest["ok"] is True
    assert manifest["model_artifact"]["exists"] is False
    assert any("Model artifact not found" in warning for warning in result.warnings)

    report = Path(result.artifacts.report_markdown).read_text(encoding="utf-8")
    assert "Soridormi policy acceptance report" in report
    assert "accepted_profile" in report
    assert "Static contract: OK" in report


def test_accept_policy_profile_can_require_model(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="missing_model_profile")

    result = accept_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=tmp_path / "acceptance",
        require_model=True,
        profile_only=True,
    )

    assert not result.ok
    assert not result.manifest_ok
    assert any("Model artifact not found" in error for error in result.errors)


def test_accept_policy_profile_hashes_present_model(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    model = tmp_path / "replacement.onnx"
    model.write_bytes(b"replacement bytes")
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="present_model_profile", model_path=str(model))

    result = accept_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=tmp_path / "acceptance",
        require_model=True,
        profile_only=True,
    )

    assert result.ok
    manifest = json.loads(Path(result.artifacts.manifest_json).read_text(encoding="utf-8"))
    assert manifest["model_artifact"]["exists"] is True
    assert manifest["model_artifact"]["size_bytes"] == len(b"replacement bytes")
    assert manifest["model_artifact"]["sha256"]
