from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
DEFAULT_OPEN_DUCK_REFERENCE_DATA = Path(
    "/workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
)



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


def _env_text(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _load_open_duck_reference_period_steps(path: str | os.PathLike[str] | None) -> int | None:
    """Read Open Duck PolyReferenceMotion period_steps without importing JAX/Numpy code.

    Official Open Duck phase uses:

        imitation_i += phase_frequency_factor
        imitation_i %= PRM.nb_steps_in_period
        phase = [cos(imitation_i / PRM.nb_steps_in_period * 2π),
                 sin(imitation_i / PRM.nb_steps_in_period * 2π)]

    PRM.nb_steps_in_period is computed as int(period * fps) from the
    polynomial_coefficients.pkl file. Loading those two scalar fields directly
    keeps Soridormi runtime lightweight and avoids importing the full training
    package just to get the phase period.
    """

    if path is None or str(path).strip() == "":
        return None

    reference_path = Path(path)
    if not reference_path.exists():
        return None

    with reference_path.open("rb") as f:
        payload: Any = pickle.load(f)

    if not isinstance(payload, dict) or not payload:
        return None

    first = next(iter(payload.values()))
    if not isinstance(first, dict):
        return None

    period = first.get("period")
    fps = first.get("fps")
    if period is None or fps is None:
        return None

    steps = int(float(period) * float(fps))
    return steps if steps > 0 else None


def _resolve_phase_period_steps(
    raw_value: str,
    reference_data: str | os.PathLike[str] | None,
    *,
    require_reference_data: bool = False,
) -> tuple[int, str]:
    text = str(raw_value or "").strip().lower()
    wants_reference = text in {"", "auto", "reference", "open_duck", "0"}

    if wants_reference:
        loaded = _load_open_duck_reference_period_steps(reference_data)
        if loaded is not None:
            return loaded, "reference_data"
        if require_reference_data:
            raise FileNotFoundError(
                "Open Duck phase period was requested from reference data, but Soridormi could not "
                f"load it from {reference_data!r}. Make sure the runtime container mounts "
                "./workspace/Open_Duck_Playground at /workspaces/Open_Duck_Playground, or set "
                "SORIDORMI_PHASE_PERIOD_STEPS explicitly."
            )
        return 50, "fallback_50"

    steps = int(float(text))
    if steps <= 0:
        loaded = _load_open_duck_reference_period_steps(reference_data)
        if loaded is not None:
            return loaded, "reference_data"
        if require_reference_data:
            raise FileNotFoundError(
                "Open Duck phase period was requested from reference data, but Soridormi could not "
                f"load it from {reference_data!r}."
            )
        return 50, "fallback_50"
    return steps, "env"


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
    reference_data: str = ""
    period_source: str = "default"
    require_reference_data: bool = False

    @classmethod
    def from_env(cls) -> GaitPhaseGenerator:
        # For ONNX walking, default to Open Duck-like step mode. Existing unit
        # tests and explicit construction still use time mode unless requested.
        mode = os.environ.get("SORIDORMI_PHASE_MODE", "step").strip().lower()
        frequency_hz = env_float("SORIDORMI_PHASE_FREQUENCY", 0.0)
        enabled = env_bool("SORIDORMI_PHASE_ENABLED", True)
        phase_offset = env_float("SORIDORMI_PHASE_OFFSET", 0.0)
        reference_data = _env_text(
            "SORIDORMI_PHASE_REFERENCE_DATA",
            str(DEFAULT_OPEN_DUCK_REFERENCE_DATA),
        )
        raw_period_steps = _env_text("SORIDORMI_PHASE_PERIOD_STEPS", "auto")
        require_reference_data = env_bool("SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA", False)
        period_steps, period_source = _resolve_phase_period_steps(
            raw_period_steps,
            reference_data,
            require_reference_data=require_reference_data,
        )
        step_increment = env_float("SORIDORMI_PHASE_STEP_INCREMENT", 1.0)
        return cls(
            frequency_hz=frequency_hz,
            enabled=enabled,
            phase_offset=phase_offset,
            mode=mode,
            period_steps=period_steps,
            step_increment=step_increment,
            reference_data=reference_data,
            period_source=period_source,
            require_reference_data=require_reference_data,
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
        # Official Open Duck increments imitation_i before building the policy
        # observation for a policy step. Matching that order removes a one-step
        # phase lag relative to the reference runner.
        if self.enabled and self.mode == "step":
            self.advance()
        return self.as_list()

    def describe(self) -> dict[str, float | bool | str]:
        return {
            "frequency_hz": float(self.frequency_hz),
            "enabled": bool(self.enabled),
            "phase_offset": float(self.phase_offset),
            "mode": str(self.mode),
            "period_steps": int(self.period_steps),
            "step_increment": float(self.step_increment),
            "step_index": float(self.step_index),
            "reference_data": str(self.reference_data),
            "period_source": str(self.period_source),
            "require_reference_data": bool(self.require_reference_data),
        }
