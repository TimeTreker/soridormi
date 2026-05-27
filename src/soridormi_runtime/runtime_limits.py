from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeLimits:
    max_steps: int | None = None
    max_seconds: float | None = None


def _env_optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _env_optional_positive_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if parsed > 0.0 else None


def runtime_limits_from_env() -> RuntimeLimits:
    return RuntimeLimits(
        max_steps=_env_optional_positive_int("SORIDORMI_MAX_STEPS"),
        max_seconds=_env_optional_positive_float("SORIDORMI_MAX_SECONDS"),
    )


def runtime_limit_reached(
    *,
    completed_steps: int,
    started_at: float,
    now: float,
    limits: RuntimeLimits,
) -> bool:
    if limits.max_steps is not None and completed_steps >= limits.max_steps:
        return True
    if limits.max_seconds is not None and now - started_at >= limits.max_seconds:
        return True
    return False
