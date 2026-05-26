from __future__ import annotations

from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_robot_config_policy_observation_defaults(tmp_path: Path) -> None:
    config = tmp_path / "robot.yaml"
    config.write_text(
        """
robot_name: test_robot
model:
  path: /tmp/fake.xml
actuators:
  - name: j0
""",
        encoding="utf-8",
    )

    robot = load_robot_config(config)

    assert robot.policy_observation.accelerometer_bias_xyz == [1.3, 0.0, 0.0]
    assert robot.policy_observation.foot_contact.left_body == "foot_assembly"
