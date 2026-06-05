from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soridormi_api import RobotState
from soridormi_runtime.observation_builder import ObservationBuilder
from soridormi_runtime.policy_input_features import (
    INPUT_MODE_OBSERVATION,
    build_policy_input_batch,
    normalize_policy_input_mode,
)
LINEAR_BEHAVIOR_CLONE_KIND = "linear_behavior_clone"


@dataclass(frozen=True)
class LinearBehaviorCloneModel:
    path: Path
    weights: np.ndarray
    bias: np.ndarray
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    observation_size: int = 101
    action_size: int = 14
    input_mode: str = INPUT_MODE_OBSERVATION
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _array_stats(name: str, values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "l2_norm": float(np.linalg.norm(arr)) if arr.size else 0.0,
    }


def _npz_array(payload: Any, name: str, shape: tuple[int, ...], errors: list[str]) -> np.ndarray:
    if name not in payload:
        errors.append(f"Missing array {name!r}")
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(payload[name], dtype=np.float32)
    if arr.shape != shape:
        errors.append(f"Array {name!r} shape {arr.shape} != expected {shape}")
        return np.zeros(shape, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        errors.append(f"Array {name!r} contains non-finite values")
    return arr


def _npz_scalar_int(payload: Any, name: str, default: int) -> int:
    if name not in payload:
        return int(default)
    try:
        values = np.asarray(payload[name]).reshape(-1)
        if values.size:
            return int(values[0])
    except Exception:
        return int(default)
    return int(default)


def _npz_scalar_str(payload: Any, name: str, default: str) -> str:
    if name not in payload:
        return str(default)
    try:
        values = np.asarray(payload[name]).reshape(-1)
        if values.size:
            return str(values[0])
    except Exception:
        return str(default)
    return str(default)


def load_linear_behavior_clone_model(
    path: str | os.PathLike[str],
    *,
    observation_size: int = 101,
    action_size: int = 14,
) -> LinearBehaviorCloneModel:
    model_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    if not model_path.exists():
        return LinearBehaviorCloneModel(
            path=model_path,
            weights=np.zeros((observation_size, action_size), dtype=np.float32),
            bias=np.zeros(action_size, dtype=np.float32),
            observation_mean=np.zeros(observation_size, dtype=np.float32),
            observation_std=np.ones(observation_size, dtype=np.float32),
            action_mean=np.zeros(action_size, dtype=np.float32),
            action_std=np.ones(action_size, dtype=np.float32),
            observation_size=observation_size,
            action_size=action_size,
            input_mode=INPUT_MODE_OBSERVATION,
            errors=[f"Linear behavior-clone model not found: {model_path}"],
        )

    try:
        with np.load(model_path) as payload:
            observation_size = _npz_scalar_int(payload, "observation_size", observation_size)
            action_size = _npz_scalar_int(payload, "action_size", action_size)
            input_mode = normalize_policy_input_mode(_npz_scalar_str(payload, "input_mode", INPUT_MODE_OBSERVATION))
            weights = _npz_array(payload, "weights", (observation_size, action_size), errors)
            bias = _npz_array(payload, "bias", (action_size,), errors)
            observation_mean = _npz_array(payload, "observation_mean", (observation_size,), errors)
            observation_std = _npz_array(payload, "observation_std", (observation_size,), errors)
            action_mean = _npz_array(payload, "action_mean", (action_size,), errors)
            action_std = _npz_array(payload, "action_std", (action_size,), errors)
    except Exception as exc:
        return LinearBehaviorCloneModel(
            path=model_path,
            weights=np.zeros((observation_size, action_size), dtype=np.float32),
            bias=np.zeros(action_size, dtype=np.float32),
            observation_mean=np.zeros(observation_size, dtype=np.float32),
            observation_std=np.ones(observation_size, dtype=np.float32),
            action_mean=np.zeros(action_size, dtype=np.float32),
            action_std=np.ones(action_size, dtype=np.float32),
            observation_size=observation_size,
            action_size=action_size,
            input_mode=INPUT_MODE_OBSERVATION,
            errors=[f"Failed to load linear behavior-clone model {model_path}: {exc!r}"],
        )

    if np.any(observation_std <= 0.0):
        errors.append("Array 'observation_std' must be strictly positive")
    if np.any(action_std <= 0.0):
        errors.append("Array 'action_std' must be strictly positive")

    return LinearBehaviorCloneModel(
        path=model_path,
        weights=weights,
        bias=bias,
        observation_mean=observation_mean,
        observation_std=observation_std,
        action_mean=action_mean,
        action_std=action_std,
        observation_size=observation_size,
        action_size=action_size,
        input_mode=input_mode,
        errors=errors,
        warnings=warnings,
    )


class LinearBehaviorClonePolicy:
    """Runtime policy wrapper for M6 linear behavior-cloning baseline artifacts.

    This is intentionally a lightweight deployment path for M6 sanity checks. It
    does not replace the long-term ONNX path; it lets a trained
    ``linear_behavior_clone.npz`` run through the same ObservationBuilder,
    action-history update, ActionPostprocessor, and PolicyActionMapper as the
    pretrained ONNX policy.
    """

    ACTION_SIZE = 14
    OBS_BATCH_SHAPE = (1, 101)

    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        robot_config_path: str | os.PathLike[str] | None = None,
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        explicit = policy_path or os.environ.get("SORIDORMI_POLICY_PATH")
        if explicit is None or str(explicit).strip() == "":
            raise ValueError("Linear behavior-clone policy requires a policy_path or SORIDORMI_POLICY_PATH")
        self.policy_path = Path(explicit)
        self.model = load_linear_behavior_clone_model(self.policy_path)
        if not self.model.ok:
            raise RuntimeError("; ".join(self.model.errors))
        self.input_mode = normalize_policy_input_mode(os.environ.get("SORIDORMI_POLICY_INPUT_MODE") or self.model.input_mode)
        self.last_command_vector: list[float] = [0.0] * 7
        self.observation_builder = observation_builder or ObservationBuilder.from_robot_config(
            path=robot_config_path
        )
        self.last_observation: np.ndarray | None = None
        self.last_observation_stats: dict[str, object] | None = None
        self.last_action: np.ndarray | None = None
        self.last_action_stats: dict[str, object] | None = None

    @property
    def joint_names(self) -> list[str]:
        return list(self.observation_builder.config.joint_names)

    def set_command_vector(self, command: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.last_command_vector = [float(x) for x in np.asarray(command, dtype=np.float32).reshape(-1).tolist()]
        self.observation_builder.set_command(command)

    def set_imitation_phase(self, imitation_phase: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.observation_builder.set_imitation_phase(imitation_phase)

    def set_motor_targets_by_name(self, targets_by_name: dict[str, float]) -> None:
        self.observation_builder.set_motor_targets_by_name(targets_by_name)

    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        self.observation_builder.set_default_positions_by_name(positions_by_name)

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        source_values = state.actuator_ctrl
        if source_values is not None and len(source_values) == len(state.joints.names):
            values = source_values
        else:
            values = state.joints.positions

        defaults = {
            str(name): float(value)
            for name, value in zip(state.joints.names, values)
            if name in self.joint_names
        }
        self.set_default_positions_by_name(defaults)
        self.set_motor_targets_by_name(defaults)
        return defaults

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        self.observation_builder.set_motor_targets(joint_names, positions)

    def reset_state(self) -> None:
        self.observation_builder.reset_action_history()
        self.last_observation = None
        self.last_observation_stats = None
        self.last_action = None
        self.last_action_stats = None

    def compute_action(self, state: RobotState) -> np.ndarray:
        robot_observation = self.observation_builder.build_batch(state)
        if robot_observation.shape != self.OBS_BATCH_SHAPE:
            raise RuntimeError(
                f"Policy robot observation must have shape {self.OBS_BATCH_SHAPE}, got {robot_observation.shape}"
            )
        obs = build_policy_input_batch(
            robot_observation,
            input_mode=self.input_mode,
            command_vector=self.last_command_vector,
        )
        expected_shape = (1, self.model.observation_size)
        if obs.shape != expected_shape:
            raise RuntimeError(f"Policy input must have shape {expected_shape}, got {obs.shape}")
        self.last_observation = obs.copy()
        self.last_observation_stats = _array_stats("observation", obs)

        x = (obs.astype(np.float32) - self.model.observation_mean.reshape((1, -1))) / self.model.observation_std.reshape((1, -1))
        y_norm = x @ self.model.weights + self.model.bias.reshape((1, -1))
        action = y_norm * self.model.action_std.reshape((1, -1)) + self.model.action_mean.reshape((1, -1))
        action = np.asarray(action, dtype=np.float32).reshape(self.ACTION_SIZE)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Linear behavior-clone policy produced non-finite action values")

        self.last_action = action.copy()
        self.last_action_stats = _array_stats("action", action)
        self.observation_builder.update_action_history(action)
        return action

    def get_observation(self) -> list[float] | None:
        if self.last_observation is None:
            return None
        return [float(x) for x in np.asarray(self.last_observation, dtype=np.float32).reshape(-1)]

    def get_observation_stats(self) -> dict[str, object] | None:
        return None if self.last_observation_stats is None else dict(self.last_observation_stats)

    def get_action_stats(self) -> dict[str, object] | None:
        return None if self.last_action_stats is None else dict(self.last_action_stats)

    def describe(self) -> dict[str, Any]:
        return {
            "policy_backend": LINEAR_BEHAVIOR_CLONE_KIND,
            "policy_path": str(self.policy_path),
            "input_name": "obs",
            "input_shape": [1, self.model.observation_size],
            "input_mode": self.input_mode,
            "output_name": "continuous_actions",
            "output_shape": [1, self.model.action_size],
            "joint_names": self.joint_names,
        }
