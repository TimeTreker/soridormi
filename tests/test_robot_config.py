from __future__ import annotations

from pathlib import Path

from soridormi_sim.robot_config import load_robot_config


def test_open_duck_mini_v2_robot_config_loads() -> None:
    config = load_robot_config(Path("configs/robots/open_duck_mini_v2.yaml"))

    assert config.robot_name == "open_duck_mini_v2"
    assert config.model.path.endswith("scene_flat_terrain.xml")
    assert config.simulation.substeps_per_api_step == 10
    assert config.base.free_joint_name == "floating_base"
    assert config.actuator_names == [
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


def test_open_duck_mini_v2_viewer_config_loads() -> None:
    config = load_robot_config(Path("configs/robots/open_duck_mini_v2.yaml"))

    assert config.viewer.enabled_env == "SORIDORMI_MUJOCO_VIEWER"
    assert config.viewer.sync_every_api_step is True
    assert config.viewer.show_left_ui is True
    assert config.viewer.show_right_ui is True
