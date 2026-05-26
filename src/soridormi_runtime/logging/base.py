from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol

from soridormi_api import MotorCommand, RobotState

TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def now_ns() -> int:
    return time.time_ns()


def default_log_dir() -> Path:
    return Path(os.environ.get("SORIDORMI_RUNTIME_LOG_DIR", "/data/logs"))


def model_to_json_dict(model: Any) -> dict[str, Any]:
    """Return a JSON-safe dict for Pydantic v2 or v1 models."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Object is not a Pydantic model: {type(model)!r}")


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/Pydantic-ish values into JSON-safe values."""
    if hasattr(value, "model_dump") or hasattr(value, "dict"):
        return model_to_json_dict(value)

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


class RuntimeLogger(Protocol):
    path: Path | None

    def log_step(
        self,
        *,
        step_index: int,
        state: RobotState,
        command: MotorCommand,
        mode: str,
        backend: str,
        policy_action: list[float] | None = None,
        policy_debug: dict[str, Any] | None = None,
        policy_observation_stats: dict[str, Any] | None = None,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class NullRuntimeLogger:
    path: Path | None = None

    def log_step(
        self,
        *,
        step_index: int,
        state: RobotState,
        command: MotorCommand,
        mode: str,
        backend: str,
        policy_action: list[float] | None = None,
        policy_debug: dict[str, Any] | None = None,
        policy_observation_stats: dict[str, Any] | None = None,
    ) -> None:
        return

    def close(self) -> None:
        return
