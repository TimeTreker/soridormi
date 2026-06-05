from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from soridormi_runtime.create_policy_profile import (
    build_replacement_profile_payload,
    create_replacement_profile,
)
from soridormi_runtime.policy_contract import build_policy_contract
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


def write_template_profile(path: Path) -> None:
    path.write_text(
        """
name: template_forward
description: template profile
model:
  path: /models/template.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: [1, 101]
  output_shape: [1, 14]
  input_type: tensor(float)
  output_type: tensor(float)
runtime:
  mode: onnx_policy
  backend: sim
  control_hz: 50
  reset_at_start: true
  sync_step: true
command:
  x: 0.15
  y: 0.0
  yaw: 0.0
phase:
  mode: step
  period_steps: auto
action_mapping:
  action_scale: 0.25
  max_motor_velocity: 5.24
logging:
  enabled: true
  format: mcap
  every_n: 1
  prefix: policy_template_forward
""",
        encoding="utf-8",
    )


def test_build_replacement_profile_payload_stamps_contract(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    payload = build_replacement_profile_payload(
        name="my_replacement",
        model_path="/models/my_replacement.onnx",
        template=template_path,
        description="My replacement policy",
        robot_config_path=robot_config,
    )

    assert payload["name"] == "my_replacement"
    assert payload["description"] == "My replacement policy"
    assert payload["model"]["path"] == "/models/my_replacement.onnx"
    assert payload["metadata"]["derived_from_profile"] == "template_forward"
    assert payload["contract"]["observation_size"] == 101
    assert payload["contract"]["action_size"] == 14
    assert payload["contract"]["joint_names"] == JOINT_NAMES
    assert payload["logging"]["prefix"] == "policy_my_replacement"


def test_build_replacement_profile_payload_stamps_context_input_mode(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    payload = build_replacement_profile_payload(
        name="context_model",
        model_path="/models/context.onnx",
        template=template_path,
        robot_config_path=robot_config,
        input_mode="context_stage1_command",
    )

    assert payload["contract"]["input_mode"] == "context_stage1_command"
    assert payload["contract"]["policy_input_size"] == 104
    assert payload["model"]["input_mode"] == "context_stage1_command"
    assert payload["model"]["input_shape"] == [1, 104]


def test_build_replacement_profile_rejects_mismatched_policy_input_size(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    with pytest.raises(ValueError, match="policy_input_size"):
        build_replacement_profile_payload(
            name="bad_context_model",
            model_path="/models/context.onnx",
            template=template_path,
            robot_config_path=robot_config,
            input_mode="context_stage1_command",
            policy_input_size=105,
        )


def test_create_replacement_profile_writes_loadable_valid_profile(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    output_dir = tmp_path / "profiles"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    result = create_replacement_profile(
        name="drop_in_model",
        model_path="/models/drop_in.onnx",
        template=template_path,
        output_dir=output_dir,
        robot_config_path=robot_config,
    )

    assert result.path == output_dir / "drop_in_model.yaml"
    assert result.path.exists()
    profile = PolicyProfile.load(result.path)
    assert profile.name == "drop_in_model"
    contract = build_policy_contract(profile, robot_config_path=robot_config)
    assert contract.ok
    assert contract.model["path"] == "/models/drop_in.onnx"


def test_create_replacement_profile_refuses_overwrite_without_force(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    output_dir = tmp_path / "profiles"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    create_replacement_profile(
        name="drop_in_model",
        model_path="/models/drop_in.onnx",
        template=template_path,
        output_dir=output_dir,
        robot_config_path=robot_config,
    )

    with pytest.raises(FileExistsError):
        create_replacement_profile(
            name="drop_in_model",
            model_path="/models/drop_in_v2.onnx",
            template=template_path,
            output_dir=output_dir,
            robot_config_path=robot_config,
        )

    overwritten = create_replacement_profile(
        name="drop_in_model",
        model_path="/models/drop_in_v2.onnx",
        template=template_path,
        output_dir=output_dir,
        robot_config_path=robot_config,
        force=True,
    )
    payload = yaml.safe_load(overwritten.path.read_text(encoding="utf-8"))
    assert payload["model"]["path"] == "/models/drop_in_v2.onnx"


def test_create_replacement_profile_supports_stdout_without_writing(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    output_dir = tmp_path / "profiles"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    result = create_replacement_profile(
        name="stdout_model",
        model_path="/models/stdout.onnx",
        template=template_path,
        output_dir=output_dir,
        robot_config_path=robot_config,
        stdout=True,
    )

    assert result.path is None
    assert not output_dir.exists()
    payload = yaml.safe_load(result.yaml_text)
    assert payload["name"] == "stdout_model"
    assert payload["model"]["path"] == "/models/stdout.onnx"


def test_create_replacement_profile_rejects_unsafe_profile_name(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    template_path = tmp_path / "template.yaml"
    write_robot_config(robot_config)
    write_template_profile(template_path)

    with pytest.raises(ValueError):
        build_replacement_profile_payload(
            name="../bad",
            model_path="/models/bad.onnx",
            template=template_path,
            robot_config_path=robot_config,
        )
