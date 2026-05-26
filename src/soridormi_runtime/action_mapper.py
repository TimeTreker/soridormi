from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.observation_builder import (
    load_default_pose_positions,
    resolve_robot_config_path,
)


DEFAULT_ACTION_SCALE = 0.25
DEFAULT_KP = 10.0
DEFAULT_KD = 0.5
DEFAULT_TORQUE = 0.0
DEFAULT_MAX_MOTOR_VELOCITY = 5.24


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass
class ActionMapperConfig:
    """Convert a 14D policy action vector into a MotorCommand.

    The Open Duck ONNX integration follows this convention:

        raw_target = default_pose + action_scale * action

    M3.5 adds a motor target speed limit:

        target = clip(raw_target, previous_target ± max_motor_velocity * dt)

    MuJoCoBackend still clips final controls to actuator ctrlrange as a second
    safety layer.
    """

    joint_names: list[str]
    default_positions_by_name: dict[str, float] = field(default_factory=dict)
    action_scale: float = DEFAULT_ACTION_SCALE
    kp_default: float = DEFAULT_KP
    kd_default: float = DEFAULT_KD
    torque_default: float = DEFAULT_TORQUE
    clip_to_limits: bool = True
    limits_by_name: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_motor_velocity: float = DEFAULT_MAX_MOTOR_VELOCITY
    speed_limit_enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.joint_names) != 14:
            raise ValueError(f"joint_names must contain 14 joints, got {len(self.joint_names)}")


class PolicyActionMapper:
    """Map ONNX policy actions to Soridormi MotorCommand messages."""

    ACTION_SIZE = 14

    def __init__(self, config: ActionMapperConfig) -> None:
        self.config = config
        self.last_motor_targets_by_name = {
            name: float(self.config.default_positions_by_name.get(name, 0.0))
            for name in self.config.joint_names
        }

    @classmethod
    def from_robot_config(
        cls,
        path: str | os.PathLike[str] | None = None,
    ) -> PolicyActionMapper:
        config_path = resolve_robot_config_path(path)
        payload = _load_yaml_mapping(config_path)

        joint_names = _load_actuator_names(payload)
        limits_by_name = _load_actuator_limits(payload)
        default_positions = load_default_pose_positions(config_path)

        action_mapping = payload.get("action_mapping", {})
        if action_mapping is None:
            action_mapping = {}
        if not isinstance(action_mapping, dict):
            raise ValueError("action_mapping must be a YAML mapping")

        default_pose = payload.get("default_pose", {})
        gains = {}
        if isinstance(default_pose, dict):
            gains = default_pose.get("gains", {}) or {}
        if not isinstance(gains, dict):
            gains = {}

        action_scale = _env_float(
            "SORIDORMI_ACTION_SCALE",
            float(action_mapping.get("action_scale", DEFAULT_ACTION_SCALE)),
        )
        max_motor_velocity = _env_float(
            "SORIDORMI_MAX_MOTOR_VELOCITY",
            float(action_mapping.get("max_motor_velocity", DEFAULT_MAX_MOTOR_VELOCITY)),
        )

        return cls(
            ActionMapperConfig(
                joint_names=joint_names,
                default_positions_by_name=default_positions,
                action_scale=action_scale,
                kp_default=float(action_mapping.get("kp_default", gains.get("kp_default", DEFAULT_KP))),
                kd_default=float(action_mapping.get("kd_default", gains.get("kd_default", DEFAULT_KD))),
                torque_default=float(action_mapping.get("torque_default", DEFAULT_TORQUE)),
                clip_to_limits=bool(action_mapping.get("clip_to_limits", True)),
                limits_by_name=limits_by_name,
                max_motor_velocity=max_motor_velocity,
                speed_limit_enabled=bool(action_mapping.get("speed_limit_enabled", True)),
            )
        )

    def action_to_targets(
        self,
        action: np.ndarray | list[float],
        dt: float | None = None,
    ) -> dict[str, float]:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape == (1, self.ACTION_SIZE):
            action_arr = action_arr.reshape(self.ACTION_SIZE)
        if action_arr.shape != (self.ACTION_SIZE,):
            raise ValueError(
                f"action must have shape ({self.ACTION_SIZE},) or (1, {self.ACTION_SIZE}), "
                f"got {action_arr.shape}"
            )

        targets: dict[str, float] = {}
        for i, name in enumerate(self.config.joint_names):
            default = float(self.config.default_positions_by_name.get(name, 0.0))
            raw_target = default + self.config.action_scale * float(action_arr[i])
            target = self._apply_speed_limit(name=name, raw_target=raw_target, dt=dt)

            if self.config.clip_to_limits and name in self.config.limits_by_name:
                lo, hi = self.config.limits_by_name[name]
                target = float(np.clip(target, lo, hi))

            targets[name] = float(target)

        self.last_motor_targets_by_name = dict(targets)
        return targets

    def action_to_command(
        self,
        action: np.ndarray | list[float],
        state: RobotState | None = None,
        dt: float | None = None,
    ) -> MotorCommand:
        targets_by_name = self.action_to_targets(action, dt=dt)

        if state is None:
            names = list(self.config.joint_names)
        else:
            # Preserve the robot state's joint ordering when possible.
            names = [name for name in state.joints.names if name in targets_by_name]
            if len(names) != len(self.config.joint_names):
                names = list(self.config.joint_names)

        positions = [targets_by_name[name] for name in names]
        n = len(names)

        return MotorCommand(
            names=names,
            positions=positions,
            velocities=[0.0] * n,
            kp=[self.config.kp_default] * n,
            kd=[self.config.kd_default] * n,
            torques=[self.config.torque_default] * n,
        )

    def _apply_speed_limit(self, name: str, raw_target: float, dt: float | None) -> float:
        if not self.config.speed_limit_enabled:
            return float(raw_target)
        if dt is None or dt <= 0.0:
            return float(raw_target)
        if self.config.max_motor_velocity <= 0.0:
            return float(raw_target)

        prev = float(self.last_motor_targets_by_name.get(name, raw_target))
        max_delta = float(self.config.max_motor_velocity * dt)
        return float(np.clip(raw_target, prev - max_delta, prev + max_delta))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Robot config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Robot config must be a YAML mapping: {path}")

    return payload


def _load_actuator_names(payload: dict[str, Any]) -> list[str]:
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


def _load_actuator_limits(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    actuators = payload.get("actuators")
    if not isinstance(actuators, list):
        return {}

    limits: dict[str, tuple[float, float]] = {}
    for item in actuators:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        ctrlrange = item.get("ctrlrange") or item.get("ctrl_range")
        if not name or ctrlrange is None:
            continue
        if not isinstance(ctrlrange, list | tuple) or len(ctrlrange) != 2:
            raise ValueError(f"Invalid ctrlrange for actuator {name!r}: {ctrlrange!r}")
        limits[str(name)] = (float(ctrlrange[0]), float(ctrlrange[1]))
    return limits


# Backward-compatible alias for alternate naming.
ActionToMotorCommandMapper = PolicyActionMapper
