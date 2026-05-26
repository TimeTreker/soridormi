from __future__ import annotations

from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_robot_config_auto_reset_safety_defaults(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: joint_a
""",
        encoding="utf-8",
    )

    robot_config = load_robot_config(config)

    assert robot_config.safety.auto_reset.enabled_env == "SORIDORMI_AUTO_RESET"
    assert robot_config.safety.auto_reset.min_base_height == 0.05
    assert robot_config.safety.auto_reset.max_tilt_rad == 1.2
    assert robot_config.safety.auto_reset.cooldown_seconds == 1.0


def test_robot_config_auto_reset_safety_values(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: joint_a
safety:
  auto_reset:
    enabled_env: TEST_AUTO_RESET
    min_base_height: 0.12
    max_tilt_rad: 0.9
    cooldown_seconds: 2.5
""",
        encoding="utf-8",
    )

    robot_config = load_robot_config(config)

    assert robot_config.safety.auto_reset.enabled_env == "TEST_AUTO_RESET"
    assert robot_config.safety.auto_reset.min_base_height == 0.12
    assert robot_config.safety.auto_reset.max_tilt_rad == 0.9
    assert robot_config.safety.auto_reset.cooldown_seconds == 2.5
