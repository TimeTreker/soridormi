from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class PolicyCommand:
    """Seven-dimensional command vector used by the Open Duck ONNX observation.

    Layout:
      [x_velocity, y_velocity, yaw_velocity, neck_pitch, head_pitch, head_yaw, head_roll]
    """

    x_velocity: float = 0.0
    y_velocity: float = 0.0
    yaw_velocity: float = 0.0
    neck_pitch: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

    @classmethod
    def from_env(cls) -> PolicyCommand:
        return cls(
            x_velocity=env_float("SORIDORMI_COMMAND_X", 0.0),
            y_velocity=env_float("SORIDORMI_COMMAND_Y", 0.0),
            yaw_velocity=env_float("SORIDORMI_COMMAND_YAW", 0.0),
            neck_pitch=env_float("SORIDORMI_NECK_PITCH", 0.0),
            head_pitch=env_float("SORIDORMI_HEAD_PITCH", 0.0),
            head_yaw=env_float("SORIDORMI_HEAD_YAW", 0.0),
            head_roll=env_float("SORIDORMI_HEAD_ROLL", 0.0),
        )

    def scaled(self, scale: float) -> PolicyCommand:
        return PolicyCommand(
            x_velocity=float(self.x_velocity) * scale,
            y_velocity=float(self.y_velocity) * scale,
            yaw_velocity=float(self.yaw_velocity) * scale,
            neck_pitch=float(self.neck_pitch) * scale,
            head_pitch=float(self.head_pitch) * scale,
            head_yaw=float(self.head_yaw) * scale,
            head_roll=float(self.head_roll) * scale,
        )

    def as_vector(self) -> np.ndarray:
        return np.asarray(self.as_list(), dtype=np.float32)

    def as_list(self) -> list[float]:
        return [
            float(self.x_velocity),
            float(self.y_velocity),
            float(self.yaw_velocity),
            float(self.neck_pitch),
            float(self.head_pitch),
            float(self.head_yaw),
            float(self.head_roll),
        ]

    def describe(self) -> dict[str, float]:
        return {
            "x_velocity": float(self.x_velocity),
            "y_velocity": float(self.y_velocity),
            "yaw_velocity": float(self.yaw_velocity),
            "neck_pitch": float(self.neck_pitch),
            "head_pitch": float(self.head_pitch),
            "head_yaw": float(self.head_yaw),
            "head_roll": float(self.head_roll),
        }


@dataclass
class GaitPhaseGenerator:
    """Phase oscillator for the ONNX observation imitation_phase field.

    Open Duck's original MuJoCo inference advances an integer imitation index
    once per policy/control step, then normalizes it by the reference-motion
    period length. Soridormi supports that step-based mode as the default for
    first-walk policy experiments. The older wall-clock frequency mode remains
    available for tests and manual probing.
    """

    frequency_hz: float = 0.0
    enabled: bool = True
    start_time: float = field(default_factory=time.monotonic)
    phase_offset: float = 0.0
    mode: str = "time"
    period_steps: int = 50
    step_increment: float = 1.0
    step_index: float = 0.0

    @classmethod
    def from_env(cls) -> GaitPhaseGenerator:
        # For ONNX walking, default to Open Duck-like step mode. Existing unit
        # tests and explicit construction still use time mode unless requested.
        mode = os.environ.get("SORIDORMI_PHASE_MODE", "step").strip().lower()
        frequency_hz = env_float("SORIDORMI_PHASE_FREQUENCY", 0.0)
        enabled = env_bool("SORIDORMI_PHASE_ENABLED", True)
        phase_offset = env_float("SORIDORMI_PHASE_OFFSET", 0.0)
        period_steps = max(1, env_int("SORIDORMI_PHASE_PERIOD_STEPS", 50))
        step_increment = env_float("SORIDORMI_PHASE_STEP_INCREMENT", 1.0)
        return cls(
            frequency_hz=frequency_hz,
            enabled=enabled,
            phase_offset=phase_offset,
            mode=mode,
            period_steps=period_steps,
            step_increment=step_increment,
        )

    def reset(self, now: float | None = None) -> None:
        self.start_time = time.monotonic() if now is None else float(now)
        self.step_index = 0.0

    def phase(self, now: float | None = None) -> float:
        if not self.enabled:
            return float(self.phase_offset % 1.0)

        if self.mode == "step":
            return float((self.phase_offset + self.step_index / max(1, self.period_steps)) % 1.0)

        if self.frequency_hz == 0.0:
            return float(self.phase_offset % 1.0)

        t = time.monotonic() if now is None else now
        elapsed = max(0.0, t - self.start_time)
        return float((self.phase_offset + elapsed * self.frequency_hz) % 1.0)

    def advance(self) -> None:
        if self.enabled and self.mode == "step":
            self.step_index = (self.step_index + self.step_increment) % max(1, self.period_steps)

    def vector(self, now: float | None = None) -> np.ndarray:
        phase = self.phase(now)
        angle = 2.0 * math.pi * phase
        return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)

    def as_list(self, now: float | None = None) -> list[float]:
        return [float(x) for x in self.vector(now).tolist()]

    def advance_and_as_list(self) -> list[float]:
        values = self.as_list()
        self.advance()
        return values

    def describe(self) -> dict[str, float | bool | str]:
        return {
            "frequency_hz": float(self.frequency_hz),
            "enabled": bool(self.enabled),
            "phase_offset": float(self.phase_offset),
            "mode": str(self.mode),
            "period_steps": int(self.period_steps),
            "step_increment": float(self.step_increment),
            "step_index": float(self.step_index),
        }
