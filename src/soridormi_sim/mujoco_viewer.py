from __future__ import annotations

import os
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
        print("MuJoCo passive viewer launched.")

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
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None
