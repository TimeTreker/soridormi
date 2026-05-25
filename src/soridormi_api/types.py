from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JointState(BaseModel):
    names: list[str]
    positions: list[float]
    velocities: list[float]
    torques: list[float]

    @field_validator("positions", "velocities", "torques")
    @classmethod
    def _non_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("joint vectors must not be empty")
        return value


class IMUState(BaseModel):
    quat_wxyz: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyro_xyz: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    accel_xyz: list[float] = Field(default_factory=lambda: [0.0, 0.0, 9.81])


class BatteryState(BaseModel):
    voltage: float | None = None
    current: float | None = None
    percent: float | None = None


class RobotState(BaseModel):
    time: float
    joints: JointState
    imu: IMUState
    battery: BatteryState | None = None


class MotorCommand(BaseModel):
    names: list[str]
    positions: list[float]
    velocities: list[float]
    kp: list[float]
    kd: list[float]
    torques: list[float]


class ApiRequest(BaseModel):
    kind: Literal["ping", "get_state", "send_command"]
    command: MotorCommand | None = None


class ApiResponse(BaseModel):
    ok: bool
    message: str = ""
    state: RobotState | None = None
