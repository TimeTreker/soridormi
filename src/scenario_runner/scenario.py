"""Scenario file contract for a world map and scenario-owned dynamic elements."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorldMap(StrictModel):
    base_xml: Path
    static_layout: Literal["none", "table_chairs"] = "none"


class SceneGeom(StrictModel):
    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: tuple[float, float, float]
    local_position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    collidable: bool = False

    @field_validator("size")
    @classmethod
    def _positive_size(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not math.isfinite(item) or item < 0 for item in value) or value[0] <= 0:
            raise ValueError("geom size must be finite and positive")
        return value

    @field_validator("local_position_xyz", "rgba")
    @classmethod
    def _finite_components(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("geom components must be finite")
        return value

    @model_validator(mode="after")
    def _shape_size(self) -> SceneGeom:
        required = 3 if self.shape == "box" else 2 if self.shape == "cylinder" else 1
        if any(item <= 0 for item in self.size[:required]):
            raise ValueError(f"{self.shape} requires {required} positive size components")
        if any(item < 0 or item > 1 for item in self.rgba):
            raise ValueError("geom rgba components must be between 0 and 1")
        return self


class DynamicElement(StrictModel):
    name: str
    position_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    geoms: list[SceneGeom] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _scenario_name(cls, value: str) -> str:
        if not value.startswith("scenario_") or not value.removeprefix("scenario_").replace("_", "").isalnum():
            raise ValueError("dynamic element name must start with scenario_ and use letters, digits, or underscores")
        return value

    @field_validator("position_xyz", "quat_wxyz")
    @classmethod
    def _finite_pose(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("element pose must be finite")
        return value

    @model_validator(mode="after")
    def _unit_quaternion(self) -> DynamicElement:
        norm = math.sqrt(sum(item * item for item in self.quat_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise ValueError("element quaternion must have unit length")
        return self


class PoseEvent(StrictModel):
    at_sim_time_s: float = Field(ge=0)
    element_name: str
    position_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] | None = None

    @field_validator("at_sim_time_s")
    @classmethod
    def _finite_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("event time must be finite")
        return value

    @field_validator("position_xyz", "quat_wxyz")
    @classmethod
    def _finite_pose(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is not None and any(not math.isfinite(item) for item in value):
            raise ValueError("event pose must be finite")
        return value

    @model_validator(mode="after")
    def _unit_quaternion(self) -> PoseEvent:
        if self.quat_wxyz is not None:
            norm = math.sqrt(sum(item * item for item in self.quat_wxyz))
            if not math.isclose(norm, 1.0, abs_tol=1e-4):
                raise ValueError("event quaternion must have unit length")
        return self


class Scenario(StrictModel):
    name: str
    world_map: WorldMap
    dynamic_elements: list[DynamicElement] = Field(default_factory=list)
    events: list[PoseEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> Scenario:
        names = [element.name for element in self.dynamic_elements]
        geom_names = [geom.name for element in self.dynamic_elements for geom in element.geoms]
        if len(names) != len(set(names)) or len(geom_names) != len(set(geom_names)):
            raise ValueError("scenario dynamic body and geom names must be unique")
        if any(event.element_name not in names for event in self.events):
            raise ValueError("scenario event refers to an undeclared dynamic element")
        if any(
            self.events[index].at_sim_time_s > self.events[index + 1].at_sim_time_s
            for index in range(len(self.events) - 1)
        ):
            raise ValueError("scenario events must be ordered by simulation time")
        return self


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))
