from __future__ import annotations

from typing import Any, Sequence

import numpy as np

INPUT_MODE_OBSERVATION = "observation"
INPUT_MODE_CONTEXT_COMMAND_V1 = "context_command_v1"
INPUT_MODES = {
    INPUT_MODE_OBSERVATION,
    INPUT_MODE_CONTEXT_COMMAND_V1,
}
CONTEXT_COMMAND_V1_FIELDS = ("vx_mps", "vy_mps", "yaw_radps")


def normalize_policy_input_mode(value: Any) -> str:
    text = str(value or INPUT_MODE_OBSERVATION).strip().lower().replace("-", "_")
    if text in {"", "legacy", "robot_state", "robot_observation"}:
        return INPUT_MODE_OBSERVATION
    if text in INPUT_MODES:
        return text
    raise ValueError(f"Unsupported policy input mode {value!r}; use one of: {', '.join(sorted(INPUT_MODES))}")


def input_size_for(input_mode: str, *, robot_observation_size: int = 101) -> int:
    mode = normalize_policy_input_mode(input_mode)
    if mode == INPUT_MODE_OBSERVATION:
        return int(robot_observation_size)
    if mode == INPUT_MODE_CONTEXT_COMMAND_V1:
        return int(robot_observation_size) + len(CONTEXT_COMMAND_V1_FIELDS)
    raise AssertionError(f"unhandled policy input mode: {mode}")


def command_context_from_policy_command(command_vector: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if command_vector is None:
        return np.zeros((len(CONTEXT_COMMAND_V1_FIELDS),), dtype=np.float32)
    values = np.asarray(list(command_vector), dtype=np.float32).reshape(-1)
    if values.shape[0] < len(CONTEXT_COMMAND_V1_FIELDS):
        raise ValueError(
            f"Policy command vector must contain at least {len(CONTEXT_COMMAND_V1_FIELDS)} values for "
            f"{INPUT_MODE_CONTEXT_COMMAND_V1}"
        )
    return values[: len(CONTEXT_COMMAND_V1_FIELDS)].astype(np.float32, copy=True)


def build_policy_input_batch(
    robot_observation_batch: np.ndarray,
    *,
    input_mode: str = INPUT_MODE_OBSERVATION,
    command_vector: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    mode = normalize_policy_input_mode(input_mode)
    observation = np.asarray(robot_observation_batch, dtype=np.float32)
    if observation.ndim != 2 or observation.shape[0] != 1:
        raise ValueError(f"robot_observation_batch must have shape (1, N), got {observation.shape}")

    if mode == INPUT_MODE_OBSERVATION:
        return observation.astype(np.float32, copy=True)

    if mode == INPUT_MODE_CONTEXT_COMMAND_V1:
        command = command_context_from_policy_command(command_vector).reshape((1, -1))
        return np.concatenate([observation, command], axis=1).astype(np.float32, copy=False)

    raise AssertionError(f"unhandled policy input mode: {mode}")
