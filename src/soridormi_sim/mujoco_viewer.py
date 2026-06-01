from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-like environment variable.

    Accepted true values are: 1, true, yes, on, y.
    Anything else is treated as false.
    """

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_float(name: str, default: float) -> float:
    """Read a float environment variable with a clear error on bad input."""

    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


@dataclass
class MujocoViewerHandle:
    """Small wrapper around mujoco.viewer.launch_passive.

    The backend owns physics stepping. The viewer only displays the current
    model/data state. Because launch_passive is non-blocking, the backend must
    call sync() after physics steps.
    """

    model: Any
    data: Any
    enabled: bool = False
    show_left_ui: bool = True
    show_right_ui: bool = True
    follow_camera: bool = False
    camera_distance: float = 1.4
    camera_azimuth: float = 135.0
    camera_elevation: float = -20.0

    def __post_init__(self) -> None:
        self._viewer = None

        if not self.enabled:
            return

        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=self.show_left_ui,
            show_right_ui=self.show_right_ui,
        )
        self._apply_follow_camera()
        print("MuJoCo passive viewer launched.")
        if self.follow_camera:
            print(
                "MuJoCo follow camera enabled "
                f"(distance={self.camera_distance}, "
                f"azimuth={self.camera_azimuth}, "
                f"elevation={self.camera_elevation})."
            )

    @property
    def is_enabled(self) -> bool:
        return self._viewer is not None

    @property
    def is_running(self) -> bool:
        if self._viewer is None:
            return False
        return bool(self._viewer.is_running())

    def sync(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            return
        self._apply_follow_camera()
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None

    def _apply_follow_camera(self) -> None:
        """Keep the passive viewer camera centered on the floating base."""

        if self._viewer is None or not self.follow_camera:
            return

        cam = getattr(self._viewer, "cam", None)
        if cam is None:
            return

        qpos = getattr(self.data, "qpos", None)
        if qpos is None or len(qpos) < 3:
            return

        lock_factory = getattr(self._viewer, "lock", None)
        lock_context = lock_factory() if callable(lock_factory) else nullcontext()
        with lock_context:
            lookat = getattr(cam, "lookat", None)
            if lookat is not None:
                for index in range(3):
                    lookat[index] = float(qpos[index])
            if hasattr(cam, "distance"):
                cam.distance = float(self.camera_distance)
            if hasattr(cam, "azimuth"):
                cam.azimuth = float(self.camera_azimuth)
            if hasattr(cam, "elevation"):
                cam.elevation = float(self.camera_elevation)
