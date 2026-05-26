from __future__ import annotations

from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_robot_config_loads_fixed_base_debug_mode(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: left_hip_pitch
  - name: right_hip_pitch
debug:
  zero_gravity:
    enabled_env: TEST_ZERO_GRAVITY
  fixed_base:
    enabled_env: TEST_FIXED_BASE
""",
        encoding="utf-8",
    )

    robot_config = load_robot_config(config)

    assert robot_config.debug.zero_gravity.enabled_env == "TEST_ZERO_GRAVITY"
    assert robot_config.debug.fixed_base.enabled_env == "TEST_FIXED_BASE"


def test_robot_config_uses_default_fixed_base_debug_env(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: left_hip_pitch
""",
        encoding="utf-8",
    )

    robot_config = load_robot_config(config)

    assert robot_config.debug.fixed_base.enabled_env == "SORIDORMI_MUJOCO_FIXED_BASE"
