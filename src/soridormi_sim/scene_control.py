"""Simulator-only control for explicitly dynamic MuJoCo scene objects.

The static scene belongs to MuJoCo. An external scenario runner may change
declared mocap objects through this separate local socket; robot commands and
model-facing capabilities cannot reach this interface.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class SceneControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["list_objects", "set_object_pose"]
    object_name: str | None = None
    position_xyz: list[float] | None = None
    quat_wxyz: list[float] | None = None

    @field_validator("position_xyz", "quat_wxyz")
    @classmethod
    def _finite_vector(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and not all(math.isfinite(item) for item in value):
            raise ValueError("scene pose must contain finite numbers")
        return value


class SceneControl:
    def __init__(self, backend: object) -> None:
        self.backend = backend

    def handle(self, payload: dict[str, object]) -> dict[str, object]:
        request = SceneControlRequest.model_validate(payload)
        backend = self.backend
        if request.kind == "list_objects":
            if any(
                value is not None
                for value in (request.object_name, request.position_xyz, request.quat_wxyz)
            ):
                raise ValueError("list_objects takes no pose fields")
            objects: list[dict[str, object]] = []
            for name, mocap_id in self._dynamic_objects():
                objects.append(self._object_state(name, mocap_id))
            return {"ok": True, "objects": objects, "robot_time_s": float(backend.data.time)}

        name = request.object_name
        position = request.position_xyz
        if name is None or not name.startswith("scenario_"):
            raise ValueError("object is not declared dynamic in this scene")
        if position is None or len(position) != 3:
            raise ValueError("set_object_pose requires position_xyz with three values")
        quat = request.quat_wxyz
        if quat is not None and (
            len(quat) != 4
            or not math.isclose(math.sqrt(sum(item * item for item in quat)), 1.0, abs_tol=1e-4)
        ):
            raise ValueError("quat_wxyz must be a unit quaternion with four values")
        mocap_id = self._mocap_id(name)
        if mocap_id < 0:
            raise ValueError("dynamic object is absent from the loaded scene")
        backend.data.mocap_pos[mocap_id] = position
        if quat is not None:
            backend.data.mocap_quat[mocap_id] = quat
        backend.mujoco.mj_forward(backend.model, backend.data)
        return {
            "ok": True,
            "object": self._object_state(name, mocap_id),
            "robot_time_s": float(backend.data.time),
        }

    def _mocap_id(self, name: str) -> int:
        backend = self.backend
        body_id = backend.mujoco.mj_name2id(
            backend.model, backend.mujoco.mjtObj.mjOBJ_BODY, name
        )
        return int(backend.model.body_mocapid[body_id]) if body_id >= 0 else -1

    def _dynamic_objects(self) -> list[tuple[str, int]]:
        backend = self.backend
        result: list[tuple[str, int]] = []
        for body_id in range(backend.model.nbody):
            mocap_id = int(backend.model.body_mocapid[body_id])
            if mocap_id < 0:
                continue
            name = backend.mujoco.mj_id2name(
                backend.model, backend.mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name is not None and name.startswith("scenario_"):
                result.append((name, mocap_id))
        return result

    def _object_state(self, name: str, mocap_id: int) -> dict[str, object]:
        data = self.backend.data
        return {
            "name": name,
            "position_xyz": [float(value) for value in data.mocap_pos[mocap_id]],
            "quat_wxyz": [float(value) for value in data.mocap_quat[mocap_id]],
        }
