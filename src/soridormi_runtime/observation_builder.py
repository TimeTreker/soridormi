from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from soridormi_api import RobotState


DEFAULT_ROBOT_CONFIG_PATH = Path("/app/configs/robots/open_duck_mini_v2.yaml")


def resolve_robot_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_ROBOT_CONFIG")
    return Path(explicit) if explicit else DEFAULT_ROBOT_CONFIG_PATH


def load_default_pose_positions(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, float]:
    config_path = resolve_robot_config_path(path)

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        return {}

    default_pose = payload.get("default_pose", {})
    if not isinstance(default_pose, dict):
        return {}

    positions = default_pose.get("positions", {})
    if not isinstance(positions, dict):
        return {}

    return {str(name): float(value) for name, value in positions.items()}


def _array(values: list[float] | tuple[float, ...], size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


@dataclass
class ObservationBuilderConfig:
    """Configuration for the Open Duck Mini v2 ONNX observation vector.

    Observation layout:

      gyro_xyz                       3
      accel_xyz                      3
      command                        7
      joint_offsets                 14
      joint_velocities_scaled       14
      last_action                   14
      last_last_action              14
      last_last_last_action         14
      motor_targets                 14
      feet_contacts                  2
      imitation_phase                2
      --------------------------------
      total                        101
    """

    joint_names: list[str]
    default_positions_by_name: dict[str, float] = field(default_factory=dict)
    command: list[float] = field(default_factory=lambda: [0.0] * 7)
    motor_targets_by_name: dict[str, float] = field(default_factory=dict)
    feet_contacts: list[float] = field(default_factory=lambda: [0.0, 0.0])
    imitation_phase: list[float] = field(default_factory=lambda: [0.0, 0.0])
    dof_vel_scale: float = 0.05

    def __post_init__(self) -> None:
        if len(self.joint_names) != 14:
            raise ValueError(f"joint_names must contain 14 joints, got {len(self.joint_names)}")

        _array(self.command, 7, "command")
        _array(self.feet_contacts, 2, "feet_contacts")
        _array(self.imitation_phase, 2, "imitation_phase")


class ObservationBuilder:
    """Build the 101-dimensional observation expected by BEST_WALK_ONNX_2.onnx."""

    OBS_SIZE = 101
    ACTION_SIZE = 14

    def __init__(self, config: ObservationBuilderConfig) -> None:
        self.config = config

        self.last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self.last_last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self.last_last_last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)

    @classmethod
    def from_robot_config(
        cls,
        path: str | os.PathLike[str] | None = None,
        joint_names: list[str] | None = None,
    ) -> ObservationBuilder:
        config_path = resolve_robot_config_path(path)

        if joint_names is None:
            joint_names = _load_actuator_names_from_config(config_path)

        default_positions = load_default_pose_positions(config_path)

        return cls(
            ObservationBuilderConfig(
                joint_names=joint_names,
                default_positions_by_name=default_positions,
                motor_targets_by_name=default_positions,
            )
        )

    def build(self, state: RobotState) -> np.ndarray:
        state_positions = {
            name: float(value)
            for name, value in zip(state.joints.names, state.joints.positions)
        }
        state_velocities = {
            name: float(value)
            for name, value in zip(state.joints.names, state.joints.velocities)
        }

        gyro = _array(state.imu.gyro_xyz, 3, "state.imu.gyro_xyz")
        accel = _array(state.imu.accel_xyz, 3, "state.imu.accel_xyz")
        command = _array(self.config.command, 7, "command")

        joint_offsets: list[float] = []
        joint_velocities: list[float] = []
        motor_targets: list[float] = []

        for name in self.config.joint_names:
            position = state_positions.get(name, 0.0)
            velocity = state_velocities.get(name, 0.0)
            default_position = self.config.default_positions_by_name.get(name, 0.0)

            joint_offsets.append(position - default_position)
            joint_velocities.append(velocity * self.config.dof_vel_scale)
            motor_targets.append(self.config.motor_targets_by_name.get(name, default_position))

        parts = [
            gyro,
            accel,
            command,
            np.asarray(joint_offsets, dtype=np.float32),
            np.asarray(joint_velocities, dtype=np.float32),
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            np.asarray(motor_targets, dtype=np.float32),
            _array(self.config.feet_contacts, 2, "feet_contacts"),
            _array(self.config.imitation_phase, 2, "imitation_phase"),
        ]

        obs = np.concatenate(parts).astype(np.float32)

        if obs.shape != (self.OBS_SIZE,):
            raise RuntimeError(f"Observation must have shape ({self.OBS_SIZE},), got {obs.shape}")

        return obs

    def build_batch(self, state: RobotState) -> np.ndarray:
        return self.build(state)[None, :]

    def update_action_history(self, action: np.ndarray | list[float]) -> None:
        action_arr = np.asarray(action, dtype=np.float32)

        if action_arr.shape == (1, self.ACTION_SIZE):
            action_arr = action_arr.reshape(self.ACTION_SIZE)

        if action_arr.shape != (self.ACTION_SIZE,):
            raise ValueError(
                f"action must have shape ({self.ACTION_SIZE},) or (1, {self.ACTION_SIZE}), "
                f"got {action_arr.shape}"
            )

        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action_arr.copy()


def _load_actuator_names_from_config(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Robot config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload: Any = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Robot config must be a YAML mapping: {path}")

    actuators = payload.get("actuators")
    if not isinstance(actuators, list):
        raise ValueError("Robot config must contain an actuators list")

    names: list[str] = []
    for item in actuators:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item

        if not name:
            raise ValueError(f"Invalid actuator entry: {item!r}")

        names.append(str(name))

    if len(names) != 14:
        raise ValueError(f"Expected 14 actuator names, got {len(names)}")

    return names

OpenDuckObservationBuilder = ObservationBuilder
