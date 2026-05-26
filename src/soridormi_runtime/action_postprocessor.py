from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


TRUE_VALUES = {"1", "true", "yes", "on", "y"}

LEG_JOINTS = {
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
}
HEAD_NECK_JOINTS = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}

GROUP_BY_SUFFIX = {
    "hip_yaw": ("left_hip_yaw", "right_hip_yaw"),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle": ("left_ankle", "right_ankle"),
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in TRUE_VALUES


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass
class ActionPostprocessorConfig:
    """Optional ONNX action compatibility postprocessing.

    This does not replace the ONNX policy. It is a runtime compatibility shim for
    early first-walk integration: if the policy produces mostly body wiggle with
    planted feet, we can safely test whether larger leg actions and damped
    head/neck actions produce stepping attempts before training a new model.
    """

    enabled: bool = False
    leg_gain: float = 1.0
    head_gain: float = 1.0
    hip_yaw_gain: float = 1.0
    hip_roll_gain: float = 1.0
    hip_pitch_gain: float = 1.0
    knee_gain: float = 1.0
    ankle_gain: float = 1.0
    clip_abs: float = 0.0
    mode: str = "identity"


@dataclass
class ActionPostprocessor:
    config: ActionPostprocessorConfig = field(default_factory=ActionPostprocessorConfig)
    last_input_stats: dict[str, Any] | None = None
    last_output_stats: dict[str, Any] | None = None
    last_joint_gains: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ActionPostprocessor":
        return cls(
            ActionPostprocessorConfig(
                enabled=_env_bool("SORIDORMI_ACTION_POSTPROCESS", False),
                mode=os.environ.get("SORIDORMI_ACTION_POSTPROCESS_MODE", "locomotion_boost"),
                leg_gain=_env_float("SORIDORMI_LEG_ACTION_GAIN", 1.0),
                head_gain=_env_float("SORIDORMI_HEAD_ACTION_GAIN", 1.0),
                hip_yaw_gain=_env_float("SORIDORMI_HIP_YAW_ACTION_GAIN", 1.0),
                hip_roll_gain=_env_float("SORIDORMI_HIP_ROLL_ACTION_GAIN", 1.0),
                hip_pitch_gain=_env_float("SORIDORMI_HIP_PITCH_ACTION_GAIN", 1.0),
                knee_gain=_env_float("SORIDORMI_KNEE_ACTION_GAIN", 1.0),
                ankle_gain=_env_float("SORIDORMI_ANKLE_ACTION_GAIN", 1.0),
                clip_abs=_env_float("SORIDORMI_ACTION_CLIP_ABS", 0.0),
            )
        )

    def apply(self, action: np.ndarray | list[float], joint_names: list[str]) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape == (1, 14):
            arr = arr.reshape(14)
        if arr.shape != (14,):
            raise ValueError(f"action must have shape (14,) or (1, 14), got {arr.shape}")
        if len(joint_names) != 14:
            raise ValueError(f"joint_names must have 14 entries, got {len(joint_names)}")

        self.last_input_stats = _stats(arr)
        gains = self._joint_gains(joint_names)
        out = arr.copy()

        if self.config.enabled:
            gain_arr = np.asarray([gains[name] for name in joint_names], dtype=np.float32)
            out = out * gain_arr
            if self.config.clip_abs and self.config.clip_abs > 0.0:
                out = np.clip(out, -float(self.config.clip_abs), float(self.config.clip_abs))

        self.last_joint_gains = gains
        self.last_output_stats = _stats(out)
        return out.astype(np.float32)

    def _joint_gains(self, joint_names: list[str]) -> dict[str, float]:
        gains: dict[str, float] = {}
        for name in joint_names:
            if name in LEG_JOINTS:
                gains[name] = float(self.config.leg_gain)
            elif name in HEAD_NECK_JOINTS:
                gains[name] = float(self.config.head_gain)
            else:
                gains[name] = 1.0

        for group_name, names in GROUP_BY_SUFFIX.items():
            group_gain = float(getattr(self.config, f"{group_name}_gain"))
            for name in names:
                if name in gains:
                    gains[name] *= group_gain
        return gains

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "mode": str(self.config.mode),
            "leg_gain": float(self.config.leg_gain),
            "head_gain": float(self.config.head_gain),
            "hip_yaw_gain": float(self.config.hip_yaw_gain),
            "hip_roll_gain": float(self.config.hip_roll_gain),
            "hip_pitch_gain": float(self.config.hip_pitch_gain),
            "knee_gain": float(self.config.knee_gain),
            "ankle_gain": float(self.config.ankle_gain),
            "clip_abs": float(self.config.clip_abs),
        }


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "abs_max": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "abs_max": float(np.max(np.abs(arr))),
    }
