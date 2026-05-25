from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


DEFAULT_ROBOT_CONFIG_PATH = Path("/app/configs/robots/open_duck_mini_v2.yaml")


class ModelConfig(BaseModel):
    path: str


class SimulationConfig(BaseModel):
    substeps_per_api_step: int = Field(default=10, ge=1)


class BaseConfig(BaseModel):
    free_joint_name: str = "floating_base"
    qpos_xyz_slice: tuple[int, int] = (0, 3)
    qpos_quat_wxyz_slice: tuple[int, int] = (3, 7)
    qvel_linear_slice: tuple[int, int] = (0, 3)
    qvel_angular_slice: tuple[int, int] = (3, 6)

    @field_validator(
        "qpos_xyz_slice",
        "qpos_quat_wxyz_slice",
        "qvel_linear_slice",
        "qvel_angular_slice",
        mode="before",
    )
    @classmethod
    def _slice_to_tuple(cls, value: list[int] | tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("slice config must contain exactly [start, stop]")
        start, stop = int(value[0]), int(value[1])
        if start < 0 or stop <= start:
            raise ValueError("slice config must satisfy 0 <= start < stop")
        return (start, stop)


class ActuatorConfig(BaseModel):
    name: str


class ControlConfig(BaseModel):
    mode: Literal["position", "torque"] = "position"
    clip_to_ctrlrange: bool = True


class ImuConfig(BaseModel):
    accel_xyz_default: list[float] = Field(default_factory=lambda: [0.0, 0.0, 9.81])

    @field_validator("accel_xyz_default")
    @classmethod
    def _three_values(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("accel_xyz_default must contain exactly 3 values")
        return value


class ViewerConfig(BaseModel):
    enabled_env: str = "SORIDORMI_MUJOCO_VIEWER"
    sync_every_api_step: bool = True
    show_left_ui: bool = True
    show_right_ui: bool = True


class RobotConfig(BaseModel):
    robot_name: str
    model: ModelConfig
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    base: BaseConfig = Field(default_factory=BaseConfig)
    actuators: list[ActuatorConfig]
    control: ControlConfig = Field(default_factory=ControlConfig)
    imu: ImuConfig = Field(default_factory=ImuConfig)
    viewer: ViewerConfig = Field(default_factory=ViewerConfig)

    @field_validator("actuators")
    @classmethod
    def _actuators_non_empty_and_unique(
        cls,
        value: list[ActuatorConfig],
    ) -> list[ActuatorConfig]:
        if not value:
            raise ValueError("at least one actuator is required")
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("actuator names must be unique")
        return value

    @property
    def actuator_names(self) -> list[str]:
        return [item.name for item in self.actuators]


def resolve_robot_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_ROBOT_CONFIG")
    return Path(explicit) if explicit else DEFAULT_ROBOT_CONFIG_PATH


def load_robot_config(path: str | os.PathLike[str] | None = None) -> RobotConfig:
    config_path = resolve_robot_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Robot config file not found: {config_path}. "
            "Set SORIDORMI_ROBOT_CONFIG or mount configs/ into /app/configs."
        )

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Robot config must be a YAML mapping: {config_path}")

    return RobotConfig.model_validate(payload)
