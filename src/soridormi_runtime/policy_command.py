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

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.x_velocity,
                self.y_velocity,
                self.yaw_velocity,
                self.neck_pitch,
                self.head_pitch,
                self.head_yaw,
                self.head_roll,
            ],
            dtype=np.float32,
        )

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
    """Simple phase oscillator for the ONNX observation imitation_phase field."""

    frequency_hz: float = 0.0
    enabled: bool = True
    start_time: float = field(default_factory=time.monotonic)
    phase_offset: float = 0.0

    @classmethod
    def from_env(cls) -> GaitPhaseGenerator:
        frequency_hz = env_float("SORIDORMI_PHASE_FREQUENCY", 0.0)
        enabled = env_bool("SORIDORMI_PHASE_ENABLED", True)
        phase_offset = env_float("SORIDORMI_PHASE_OFFSET", 0.0)
        return cls(frequency_hz=frequency_hz, enabled=enabled, phase_offset=phase_offset)

    def phase(self, now: float | None = None) -> float:
        if not self.enabled or self.frequency_hz == 0.0:
            return float(self.phase_offset % 1.0)

        t = time.monotonic() if now is None else now
        elapsed = max(0.0, t - self.start_time)
        return float((self.phase_offset + elapsed * self.frequency_hz) % 1.0)

    def vector(self, now: float | None = None) -> np.ndarray:
        phase = self.phase(now)
        angle = 2.0 * math.pi * phase
        return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)

    def as_list(self, now: float | None = None) -> list[float]:
        return [float(x) for x in self.vector(now).tolist()]

    def describe(self) -> dict[str, float | bool]:
        return {
            "frequency_hz": float(self.frequency_hz),
            "enabled": bool(self.enabled),
            "phase_offset": float(self.phase_offset),
        }
