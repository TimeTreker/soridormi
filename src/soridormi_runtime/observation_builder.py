from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from soridormi_api import RobotState


DEFAULT_ROBOT_CONFIG_PATH = Path("configs/robots/open_duck_mini_v2.yaml")
CONTAINER_ROBOT_CONFIG_PATH = Path("/app/configs/robots/open_duck_mini_v2.yaml")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def resolve_robot_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_ROBOT_CONFIG")
    if explicit:
        return Path(explicit)
    if DEFAULT_ROBOT_CONFIG_PATH.exists():
        return DEFAULT_ROBOT_CONFIG_PATH
    return CONTAINER_ROBOT_CONFIG_PATH


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


def load_policy_observation_options(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Load Open Duck policy-observation compatibility options from YAML/env.

    The original Open Duck Mini v2 MuJoCo inference adds +1.3 to accelerometer x
    and includes real [left, right] foot contacts in the 101D observation. We keep
    constructor defaults conservative, but from_robot_config() enables these
    compatibility options by default for first-walk experiments.
    """
    config_path = resolve_robot_config_path(path)
    payload: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            payload = loaded

    policy_observation = payload.get("policy_observation", {})
    if policy_observation is None:
        policy_observation = {}
    if not isinstance(policy_observation, dict):
        raise ValueError("policy_observation must be a YAML mapping")

    accel_bias = policy_observation.get("accelerometer_bias_xyz", [1.3, 0.0, 0.0])
    use_state_feet_contacts = bool(policy_observation.get("use_state_feet_contacts", True))

    accel_bias_arr = _array(accel_bias, 3, "policy_observation.accelerometer_bias_xyz")
    accel_bias_arr[0] = _env_float("SORIDORMI_POLICY_ACCEL_BIAS_X", float(accel_bias_arr[0]))
    accel_bias_arr[1] = _env_float("SORIDORMI_POLICY_ACCEL_BIAS_Y", float(accel_bias_arr[1]))
    accel_bias_arr[2] = _env_float("SORIDORMI_POLICY_ACCEL_BIAS_Z", float(accel_bias_arr[2]))

    return {
        "accelerometer_bias_xyz": [float(x) for x in accel_bias_arr.tolist()],
        "use_state_feet_contacts": _env_bool(
            "SORIDORMI_USE_STATE_FEET_CONTACTS", use_state_feet_contacts
        ),
    }


def _array(values: list[float] | tuple[float, ...] | np.ndarray, size: int, name: str) -> np.ndarray:
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
    accelerometer_bias_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    use_state_feet_contacts: bool = False

    def __post_init__(self) -> None:
        if len(self.joint_names) != 14:
            raise ValueError(f"joint_names must contain 14 joints, got {len(self.joint_names)}")

        _array(self.command, 7, "command")
        _array(self.feet_contacts, 2, "feet_contacts")
        _array(self.imitation_phase, 2, "imitation_phase")
        _array(self.accelerometer_bias_xyz, 3, "accelerometer_bias_xyz")


class ObservationBuilder:
    """Build the 101-dimensional observation expected by BEST_WALK_ONNX_2.onnx."""

    OBS_SIZE = 101
    ACTION_SIZE = 14

    def __init__(self, config: ObservationBuilderConfig) -> None:
        self.config = config

        # Keep policy defaults and live motor targets as independent mappings.
        # from_robot_config() used to pass the same dict for both, and external
        # callers may do the same. Updating motor targets after inference must
        # not mutate default_positions_by_name, because joint_offsets are
        # computed as joint_position - default_position. A shared dict causes a
        # one-step offset drift equal to the speed-limited target update.
        self.config.default_positions_by_name = {
            str(name): float(value)
            for name, value in self.config.default_positions_by_name.items()
        }
        self.config.motor_targets_by_name = {
            str(name): float(value)
            for name, value in self.config.motor_targets_by_name.items()
        }

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
        policy_options = load_policy_observation_options(config_path)

        return cls(
            ObservationBuilderConfig(
                joint_names=joint_names,
                default_positions_by_name=default_positions,
                motor_targets_by_name=dict(default_positions),
                accelerometer_bias_xyz=list(policy_options["accelerometer_bias_xyz"]),
                use_state_feet_contacts=bool(policy_options["use_state_feet_contacts"]),
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
        accel = _array(state.imu.accel_xyz, 3, "state.imu.accel_xyz") + _array(
            self.config.accelerometer_bias_xyz, 3, "accelerometer_bias_xyz"
        )
        command = _array(self.config.command, 7, "command")
        feet_contacts = self._feet_contacts_from_state_or_config(state)

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
            feet_contacts,
            _array(self.config.imitation_phase, 2, "imitation_phase"),
        ]

        obs = np.concatenate(parts).astype(np.float32)

        if obs.shape != (self.OBS_SIZE,):
            raise RuntimeError(f"Observation must have shape ({self.OBS_SIZE},), got {obs.shape}")

        return obs

    def build_batch(self, state: RobotState) -> np.ndarray:
        return self.build(state)[None, :]

    def reset_action_history(self) -> None:
        self.last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self.last_last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        self.last_last_last_action = np.zeros(self.ACTION_SIZE, dtype=np.float32)

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

    def set_command(self, command: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.config.command = [float(x) for x in _array(command, 7, "command").tolist()]

    def set_imitation_phase(self, imitation_phase: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.config.imitation_phase = [
            float(x) for x in _array(imitation_phase, 2, "imitation_phase").tolist()
        ]

    def set_feet_contacts(self, feet_contacts: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.config.feet_contacts = [float(x) for x in _array(feet_contacts, 2, "feet_contacts").tolist()]

    def _feet_contacts_from_state_or_config(self, state: RobotState) -> np.ndarray:
        state_contacts = getattr(state, "feet_contacts", None)
        if self.config.use_state_feet_contacts and state_contacts is not None:
            return _array(state_contacts, 2, "state.feet_contacts")
        return _array(self.config.feet_contacts, 2, "feet_contacts")


    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        """Update the policy default actuator pose used for joint offsets.

        This is useful for first-walk experiments: the Open Duck policy expects
        joint_angles - default_actuator, where default_actuator comes from the
        MuJoCo home keyframe ctrl. If the YAML default pose is stale, we can
        bootstrap these defaults from the first simulator state.
        """
        clean = {str(name): float(value) for name, value in positions_by_name.items()}
        self.config.default_positions_by_name.update(clean)
        for name, value in clean.items():
            self.config.motor_targets_by_name.setdefault(name, float(value))

    def set_motor_targets_by_name(self, targets_by_name: dict[str, float]) -> None:
        """Update motor target values used in future observations."""
        clean_targets = {str(name): float(value) for name, value in targets_by_name.items()}
        self.config.motor_targets_by_name.update(clean_targets)

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        positions_arr = np.asarray(positions, dtype=np.float32)
        if positions_arr.shape != (len(joint_names),):
            raise ValueError(
                f"positions must have shape ({len(joint_names)},), got {positions_arr.shape}"
            )
        self.set_motor_targets_by_name(
            {name: float(value) for name, value in zip(joint_names, positions_arr)}
        )


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


# Backward-compatible alias for older probe/controller code.
OpenDuckObservationBuilder = ObservationBuilder
