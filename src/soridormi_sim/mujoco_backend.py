from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState

from .robot_config import RobotConfig, load_robot_config


@dataclass
class FakeMujocoBackend:
    """Safe starter backend for API testing.

    The fake backend can also be config-driven, so API tests and fake sim use the
    same joint names as the real robot model.
    """

    config: RobotConfig = field(default_factory=load_robot_config)
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

    @property
    def joint_names(self) -> list[str]:
        return self.config.actuator_names

    def step(self) -> None:
        # Starter fake dynamics: slowly move toward commanded positions by name.
        if self.last_command is None:
            return

        alpha = 0.05
        command_by_name = {
            name: target for name, target in zip(self.last_command.names, self.last_command.positions)
        }

        for i, name in enumerate(self.joint_names):
            if name not in command_by_name:
                continue
            old = self.positions[i]
            target = command_by_name[name]
            self.positions[i] = old + alpha * (target - old)
            self.velocities[i] = self.positions[i] - old

    def get_state(self) -> RobotState:
        return RobotState(
            time=time.monotonic() - self.start_time,
            joints=JointState(
                names=list(self.joint_names),
                positions=list(self.positions),
                velocities=list(self.velocities),
                torques=list(self.torques),
            ),
            imu=IMUState(accel_xyz=list(self.config.imu.accel_xyz_default)),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command


class MujocoBackend:
    """Config-driven MuJoCo backend.

    Robot-specific details live in configs/robots/*.yaml. This backend should stay
    generic so changing robot models does not require changing Python code.
    """

    def __init__(
        self,
        config_path: str | None = None,
        model_path: str | None = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.config = load_robot_config(config_path)
        self.model_path = Path(model_path or self.config.model.path)
        self.substeps_per_api_step = self.config.simulation.substeps_per_api_step

        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model XML not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.last_command: MotorCommand | None = None

        self.actuator_names = self._load_and_validate_actuator_names()
        self.joint_ids = self._load_actuator_joint_ids()
        self.joint_names = [self._joint_name(joint_id) for joint_id in self.joint_ids]
        self.qpos_addrs = [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids]
        self.qvel_addrs = [int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids]

        self.ctrl_min = np.array(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.ctrl_max = np.array(self.model.actuator_ctrlrange[:, 1], dtype=float)

        mujoco.mj_forward(self.model, self.data)

        print(f"Loaded robot config: {self.config.robot_name}")
        print(f"Loaded MuJoCo model: {self.model_path}")
        print(f"MuJoCo timestep: {self.model.opt.timestep}")
        print(f"API step substeps: {self.substeps_per_api_step}")
        print(f"Actuators: {self.actuator_names}")

    def _load_and_validate_actuator_names(self) -> list[str]:
        model_names: list[str] = []
        for i in range(self.model.nu):
            name = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_ACTUATOR,
                i,
            )
            model_names.append(name if name is not None else f"actuator_{i}")

        expected = self.config.actuator_names
        missing = [name for name in expected if name not in model_names]
        extra = [name for name in model_names if name not in expected]
        if missing or extra:
            raise ValueError(
                "MuJoCo actuator names do not match robot config. "
                f"missing={missing}, extra={extra}, model_names={model_names}, expected={expected}"
            )

        # Preserve config ordering for API consistency. Here the model order already
        # matches Open Duck Mini v2, but explicit ordering is safer for future robots.
        if model_names != expected:
            raise ValueError(
                "Config actuator order must match MuJoCo actuator order for now. "
                f"model_names={model_names}, expected={expected}"
            )
        return model_names

    def _load_actuator_joint_ids(self) -> list[int]:
        joint_ids: list[int] = []
        for i in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[i][0])
            if joint_id < 0:
                raise RuntimeError(f"Actuator {i} is not attached to a joint")
            joint_ids.append(joint_id)
        return joint_ids

    def _joint_name(self, joint_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.model,
            self.mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        return name if name is not None else f"joint_{joint_id}"

    def step(self) -> None:
        if self.last_command is not None:
            self._apply_last_command_to_ctrl()

        for _ in range(self.substeps_per_api_step):
            self.mujoco.mj_step(self.model, self.data)

    def get_state(self) -> RobotState:
        positions = [float(self.data.qpos[addr]) for addr in self.qpos_addrs]
        velocities = [float(self.data.qvel[addr]) for addr in self.qvel_addrs]
        torques = [float(x) for x in self.data.actuator_force[: self.model.nu]]

        return RobotState(
            time=float(self.data.time),
            joints=JointState(
                names=list(self.actuator_names),
                positions=positions,
                velocities=velocities,
                torques=torques,
            ),
            imu=self._read_base_imu(),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command

    def _apply_last_command_to_ctrl(self) -> None:
        assert self.last_command is not None

        if self.config.control.mode == "torque":
            command_values = {
                name: value for name, value in zip(self.last_command.names, self.last_command.torques)
            }
        else:
            command_values = {
                name: value for name, value in zip(self.last_command.names, self.last_command.positions)
            }

        for actuator_index, actuator_name in enumerate(self.actuator_names):
            if actuator_name not in command_values:
                continue

            target = float(command_values[actuator_name])
            if self.config.control.clip_to_ctrlrange:
                target = float(np.clip(target, self.ctrl_min[actuator_index], self.ctrl_max[actuator_index]))
            self.data.ctrl[actuator_index] = target

    def _slice(self, vector: np.ndarray, bounds: tuple[int, int]) -> list[float]:
        start, stop = bounds
        return [float(x) for x in vector[start:stop]]

    def _read_base_imu(self) -> IMUState:
        base = self.config.base
        return IMUState(
            quat_wxyz=self._slice(self.data.qpos, base.qpos_quat_wxyz_slice),
            gyro_xyz=self._slice(self.data.qvel, base.qvel_angular_slice),
            accel_xyz=list(self.config.imu.accel_xyz_default),
        )
