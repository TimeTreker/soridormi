from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_robot_config_has_default_zero_gravity_env(tmp_path: Path) -> None:
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

    assert robot_config.debug.zero_gravity.enabled_env == "SORIDORMI_MUJOCO_ZERO_GRAVITY"
