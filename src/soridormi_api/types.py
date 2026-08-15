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
    # Optional policy-observation metadata. MuJoCo fills this with
    # [left_foot_contact, right_foot_contact] as floats 0.0/1.0.
    # Hardware backends can leave it unset until contact sensing exists.
    feet_contacts: list[float] | None = None
    # Optional floating-base pose. The simulator backend fills this so runnable policy engine
    # analysis can report forward/lateral displacement. Real hardware backends
    # can leave it unset until state estimation is available.
    base_position_xyz: list[float] | None = None
    base_quat_wxyz: list[float] | None = None
    # Optional [left, right] foot world positions, each [x, y, z].
    # MuJoCo fills this for stepping diagnostics. Real hardware can leave it
    # unset until kinematics/state estimation is available.
    feet_position_xyz: list[list[float]] | None = None
    # Optional actuator control vector in the backend's actuator order.
    # MuJoCo fills this with data.ctrl, which lets the runtime bootstrap the
    # policy's default_actuator/motor_targets from model.keyframe("home").ctrl
    # instead of guessing from joint qpos. Hardware backends can leave it unset.
    actuator_ctrl: list[float] | None = None

    @field_validator("feet_contacts")
    @classmethod
    def _feet_contacts_shape(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 2:
            raise ValueError("feet_contacts must contain exactly [left, right]")
        return value

    @field_validator("base_position_xyz")
    @classmethod
    def _base_position_shape(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 3:
            raise ValueError("base_position_xyz must contain exactly [x, y, z]")
        return value

    @field_validator("base_quat_wxyz")
    @classmethod
    def _base_quat_shape(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("base_quat_wxyz must contain exactly [w, x, y, z]")
        return value

    @field_validator("feet_position_xyz")
    @classmethod
    def _feet_position_shape(
        cls,
        value: list[list[float]] | None,
    ) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("feet_position_xyz must contain exactly [left_xyz, right_xyz]")
        for item in value:
            if len(item) != 3:
                raise ValueError("each feet_position_xyz item must contain exactly [x, y, z]")
        return value

    @field_validator("actuator_ctrl")
    @classmethod
    def _actuator_ctrl_non_empty(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) == 0:
            raise ValueError("actuator_ctrl must be unset or contain at least one value")
        return value


class MotorCommand(BaseModel):
    names: list[str]
    positions: list[float]
    velocities: list[float]
    kp: list[float]
    kd: list[float]
    torques: list[float]


class VisualExpressionCommand(BaseModel):
    expression: Literal["eyes_open", "eyes_closed"]
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)


class VisualArmPoseCommand(BaseModel):
    pose: Literal["rest", "reach", "hold", "place"]


class ApiRequest(BaseModel):
    kind: Literal[
        "ping",
        "get_state",
        "send_command",
        "step_command",
        "set_visual_expression",
        "set_visual_arm_pose",
        "reset",
    ]
    command: MotorCommand | None = None
    visual_expression: VisualExpressionCommand | None = None
    visual_arm_pose: VisualArmPoseCommand | None = None


class ApiResponse(BaseModel):
    ok: bool
    message: str = ""
    state: RobotState | None = None
