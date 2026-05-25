from __future__ import annotations

import time
from dataclasses import dataclass, field

from soridormi_api import IMUState, JointState, MotorCommand, RobotState

DEFAULT_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]


@dataclass
class FakeMujocoBackend:
    """Safe starter backend for API testing.

    Replace this with a real MuJoCo-backed implementation. The public methods should remain
    unchanged so runtime code does not change between fake sim, real MuJoCo, and hardware.
    """

    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_NAMES))
    start_time: float = field(default_factory=time.monotonic)
    positions: list[float] = field(init=False)
    velocities: list[float] = field(init=False)
    torques: list[float] = field(init=False)
    last_command: MotorCommand | None = None

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        self.positions = [0.0] * n
        self.velocities = [0.0] * n
        self.torques = [0.0] * n

    def step(self) -> None:
        # Starter fake dynamics: slowly move toward commanded positions.
        if self.last_command is None:
            return
        alpha = 0.05
        for i, target in enumerate(self.last_command.positions[: len(self.positions)]):
            old = self.positions[i]
            self.positions[i] = old + alpha * (target - old)
            self.velocities[i] = self.positions[i] - old

    def get_state(self) -> RobotState:
        return RobotState(
            time=time.monotonic() - self.start_time,
            joints=JointState(
                names=self.joint_names,
                positions=self.positions,
                velocities=self.velocities,
                torques=self.torques,
            ),
            imu=IMUState(),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command


class MujocoBackend(FakeMujocoBackend):
    """TODO: connect to the actual Open Duck MuJoCo XML.

    Suggested next steps:
      1. Load playground/open_duck_mini_v2/xmls/scene.xml.
      2. Map MuJoCo joints/actuators to soridormi_api joint names.
      3. Convert MotorCommand into ctrl targets/torques.
      4. Convert qpos/qvel/sensors into RobotState.
    """

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self.model_path = model_path
        try:
            import mujoco  # noqa: F401
        except Exception as exc:
            print(f"MuJoCo import failed; using fake backend. Reason: {exc!r}")
