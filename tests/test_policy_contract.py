from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from soridormi_runtime.policy_contract import build_policy_contract, observation_segments
from soridormi_runtime.policy_input_features import INPUT_MODE_CONTEXT_COMMAND_V1


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


def write_profile(
    path: Path,
    *,
    output_shape: str = "[1, 14]",
    joint_names: list[str] | None = None,
    input_shape: str = "[1, 101]",
    input_mode: str = "observation",
    policy_input_size: int = 101,
) -> None:
    joint_names_block = ""
    if joint_names is not None:
        joint_names_block = "  joint_names:\n" + "\n".join(f"    - {name}" for name in joint_names) + "\n"
    path.write_text(
        f"""
name: replacement_test
description: replacement contract unit test
metadata:
  format_version: 1
  policy_family: test
contract:
  observation_size: 101
  input_mode: {input_mode}
  policy_input_size: {policy_input_size}
  action_size: 14
{joint_names_block}model:
  path: /models/replacement.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: {input_shape}
  output_shape: {output_shape}
  input_type: tensor(float)
  output_type: tensor(float)
  input_mode: {input_mode}
action_mapping:
  action_scale: 0.25
  max_motor_velocity: 5.24
phase:
  mode: step
  period_steps: auto
""",
        encoding="utf-8",
    )


def test_observation_segments_are_canonical_101d_layout() -> None:
    segments = observation_segments()

    assert sum(item.size for item in segments) == 101
    assert segments[0].name == "gyro_xyz"
    assert segments[0].start == 0
    assert segments[-1].name == "imitation_phase"
    assert segments[-1].end == 101


def test_policy_contract_exports_replacement_interface(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile_path, joint_names=JOINT_NAMES)

    result = build_policy_contract(profile_path, robot_config_path=robot_config)

    assert result.ok
    assert result.observation["size"] == 101
    assert result.policy_input["mode"] == "observation"
    assert result.policy_input["size"] == 101
    assert result.action["size"] == 14
    assert result.model["input_shape"] == [1, 101]
    assert result.model["output_shape"] == [1, 14]
    assert result.action["joint_order"] == JOINT_NAMES
    assert result.action["action_scale"] == 0.25
    assert result.action["max_motor_velocity"] == 5.24
    assert result.action["kp_default"] == 11.0
    assert result.command["fields"] == [
        "x",
        "y",
        "yaw",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    ]
    assert asdict(result)["ok"] is True


def test_policy_contract_reports_model_shape_mismatch(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile_path, output_shape="[1, 12]")

    result = build_policy_contract(profile_path, robot_config_path=robot_config)

    assert not result.ok
    assert any("Model output last dimension" in error for error in result.errors)


def test_policy_contract_accepts_command_context_policy_input(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(
        profile_path,
        input_shape="[1, 104]",
        input_mode=INPUT_MODE_CONTEXT_COMMAND_V1,
        policy_input_size=104,
        joint_names=JOINT_NAMES,
    )

    result = build_policy_contract(profile_path, robot_config_path=robot_config)

    assert result.ok
    assert result.observation["size"] == 101
    assert result.policy_input["mode"] == INPUT_MODE_CONTEXT_COMMAND_V1
    assert result.policy_input["size"] == 104
    assert result.policy_input["segments"][-1]["name"] == "desired_command.vx_vy_yaw"
    assert result.model["input_shape"] == [1, 104]


def test_policy_contract_reports_declared_joint_order_mismatch(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile_path = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    wrong_order = list(JOINT_NAMES)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    write_profile(profile_path, joint_names=wrong_order)

    result = build_policy_contract(profile_path, robot_config_path=robot_config)

    assert not result.ok
    assert any("joint_names" in error for error in result.errors)
