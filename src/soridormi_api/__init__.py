from __future__ import annotations

from .types import IMUState, JointState, MotorCommand, RobotState, VisualExpressionCommand

__all__ = [
    "IMUState",
    "JointState",
    "MotorCommand",
    "RobotState",
    "VisualExpressionCommand",
    "RobotApiClient",
]


def __getattr__(name: str):
    """Lazily expose API client so type-only imports do not require pyzmq."""

    if name == "RobotApiClient":
        from .client import RobotApiClient

        return RobotApiClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
