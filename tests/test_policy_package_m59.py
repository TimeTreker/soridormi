from __future__ import annotations

import json
import tarfile
from pathlib import Path

from soridormi_runtime.policy_package import package_policy_profile, verify_policy_package


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
description: package test profile
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


def test_package_policy_profile_creates_verifiable_handoff_without_model(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="package_profile")

    result = package_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=tmp_path / "packages",
    )

    assert result.ok
    package_path = Path(result.package_path)
    assert package_path.is_file()
    assert not result.include_model

    with tarfile.open(package_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "profile.yaml" in names
    assert "package_manifest.json" in names
    assert "artifacts/contract.json" in names
    assert "artifacts/manifest.json" in names
    assert "artifacts/acceptance.json" in names
    assert "artifacts/acceptance_report.md" in names
    assert not any(name.startswith("model/") for name in names)

    verification = verify_policy_package(package_path)
    assert verification.ok
    assert verification.profile_name == "package_profile"
    assert verification.files_checked >= 5


def test_package_policy_profile_can_embed_model_bytes(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    model = tmp_path / "replacement.onnx"
    model.write_bytes(b"replacement package bytes")
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="model_package_profile", model_path=str(model))

    result = package_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=tmp_path / "packages",
        include_model=True,
        require_model=True,
    )

    assert result.ok
    with tarfile.open(result.package_path, "r:gz") as tar:
        names = set(tar.getnames())
        manifest = json.loads(tar.extractfile("package_manifest.json").read().decode("utf-8"))  # type: ignore[union-attr]
    assert "model/replacement.onnx" in names
    assert manifest["model_artifact"]["packaged_path"] == "model/replacement.onnx"
    assert manifest["model_artifact"]["size_bytes"] == len(b"replacement package bytes")
    assert verify_policy_package(result.package_path).ok


def test_verify_policy_package_reports_hash_tampering(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="tamper_profile")

    result = package_policy_profile(profile, robot_config_path=robot_config, output_dir=tmp_path / "packages")
    original = Path(result.package_path)
    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    with tarfile.open(original, "r:gz") as tar:
        tar.extractall(tampered_root, filter="data")
    (tampered_root / "profile.yaml").write_text("name: tampered\n", encoding="utf-8")
    tampered = tmp_path / "tampered.policy.tar.gz"
    with tarfile.open(tampered, "w:gz") as tar:
        for path in sorted(tampered_root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(tampered_root).as_posix())

    verification = verify_policy_package(tampered)
    assert not verification.ok
    assert any("sha256 mismatch" in error for error in verification.errors)


def test_package_policy_profile_fails_when_required_model_missing(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="missing_model_package")

    result = package_policy_profile(
        profile,
        robot_config_path=robot_config,
        output_dir=tmp_path / "packages",
        include_model=True,
        require_model=True,
    )

    assert not result.ok
    assert any("Model artifact not found" in error for error in result.errors)
