from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

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


def model_to_json_dict(model):
    """Return a JSON-safe dict for Pydantic v2 or v1 models."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"Object is not a Pydantic model: {type(model)!r}")


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
    ) -> None:
        return

    def close(self) -> None:
        return
