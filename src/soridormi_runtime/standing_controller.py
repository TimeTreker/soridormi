from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from soridormi_api import MotorCommand, RobotState


DEFAULT_ROBOT_CONFIG_PATH = Path("/app/configs/robots/open_duck_mini_v2.yaml")


def resolve_robot_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_ROBOT_CONFIG")
    return Path(explicit) if explicit else DEFAULT_ROBOT_CONFIG_PATH


def load_default_pose(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = resolve_robot_config_path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Robot config file not found: {config_path}. "
            "Set SORIDORMI_ROBOT_CONFIG or mount configs/ into /app/configs."
        )

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Robot config must be a YAML mapping: {config_path}")

    default_pose = payload.get("default_pose", {})
    if default_pose is None:
        default_pose = {}

    if not isinstance(default_pose, dict):
        raise ValueError("default_pose must be a YAML mapping")

    return default_pose


class StandingPoseController:
    """Command a configured default standing pose.

    Unknown joints are commanded to hold their current positions. This lets the
    same controller work even if the model contains extra joints.
    """

    def __init__(self, config_path: str | os.PathLike[str] | None = None) -> None:
        default_pose = load_default_pose(config_path)

        positions = default_pose.get("positions", {})
        if positions is None:
            positions = {}
        if not isinstance(positions, dict):
            raise ValueError("default_pose.positions must be a mapping of joint name to position")

        gains = default_pose.get("gains", {})
        if gains is None:
            gains = {}
        if not isinstance(gains, dict):
            raise ValueError("default_pose.gains must be a mapping")

        self.positions_by_name = {
            str(name): float(value) for name, value in positions.items()
        }

        self.kp_default = float(gains.get("kp_default", 10.0))
        self.kd_default = float(gains.get("kd_default", 0.5))
        self.torque_default = float(default_pose.get("torque_default", 0.0))

    def compute(self, state: RobotState) -> MotorCommand:
        names = list(state.joints.names)
        n = len(names)

        target_positions: list[float] = []
        for i, name in enumerate(names):
            if name in self.positions_by_name:
                target_positions.append(self.positions_by_name[name])
            else:
                # Unknown joint: hold current position.
                target_positions.append(float(state.joints.positions[i]))

        return MotorCommand(
            names=names,
            positions=target_positions,
            velocities=[0.0] * n,
            kp=[self.kp_default] * n,
            kd=[self.kd_default] * n,
            torques=[self.torque_default] * n,
        )
