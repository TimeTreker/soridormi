from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState, VisualExpressionCommand
from .social_eye_scene import (
    LEFT_EYE_CLOSED_NAME,
    LEFT_EYE_NAME,
    RIGHT_EYE_CLOSED_NAME,
    RIGHT_EYE_NAME,
)
from .mujoco_viewer import MujocoViewerHandle, env_flag, env_float
from .robot_config import RobotConfig, load_robot_config


@dataclass
class FakeMujocoBackend:
    """Safe starter backend for API testing.

    The fake backend can also be config-driven, so API tests and fake sim use
    the same joint names as the real robot model.
    """

    config: RobotConfig = field(default_factory=load_robot_config)
    start_time: float = field(default_factory=time.monotonic)
    positions: list[float] = field(init=False)
    velocities: list[float] = field(init=False)
    torques: list[float] = field(init=False)
    last_command: MotorCommand | None = None
    last_visual_expression: VisualExpressionCommand | None = None

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
            feet_contacts=[0.0, 0.0],
            feet_position_xyz=None,
            actuator_ctrl=list(self.positions),
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command

    def apply_visual_expression(self, command: VisualExpressionCommand) -> None:
        self.last_visual_expression = command

    def reset(self) -> None:
        self.start_time = time.monotonic()
        n = len(self.joint_names)
        self.positions = [0.0] * n
        self.velocities = [0.0] * n
        self.torques = [0.0] * n
        self.last_command = None
        self.last_visual_expression = None


class MujocoBackend:
    """Config-driven MuJoCo backend.

    Robot-specific details live in configs/robots/*.yaml. This backend should
    stay generic so changing robot models does not require changing Python code.
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

        self.official_reset_sequence = env_flag(
            "SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE",
            default=False,
        )
        self.official_sensor_mode = env_flag(
            "SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE",
            default=False,
        )
        self.official_contact_mode = env_flag(
            "SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE",
            default=False,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        # Official Open Duck inference forces 2 ms MuJoCo steps. Keep this as
        # an opt-in compatibility mode so generic robot configs remain generic.
        if self.official_reset_sequence:
            self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.last_command: MotorCommand | None = None
        self._original_gravity = np.array(self.model.opt.gravity, dtype=float)
        self.use_home_keyframe = env_flag(
            "SORIDORMI_MUJOCO_USE_HOME_KEYFRAME",
            default=False,
        )
        self.home_keyframe_name = "home"
        self.last_reset_used_home_keyframe = False

        self.actuator_names = self._load_and_validate_actuator_names()
        self.joint_ids = self._load_actuator_joint_ids()
        self.joint_names = [self._joint_name(joint_id) for joint_id in self.joint_ids]
        self.qpos_addrs = [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids]
        self.qvel_addrs = [int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids]
        self.ctrl_min = np.array(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.ctrl_max = np.array(self.model.actuator_ctrlrange[:, 1], dtype=float)
        self.gyro_sensor_id = self._sensor_id("gyro")
        self.accelerometer_sensor_id = self._sensor_id("accelerometer")

        self._apply_home_keyframe_if_enabled()
        self._apply_configured_reset_pose()
        self._apply_startup_debug_options()
        if not self.last_reset_used_home_keyframe:
            self._initialize_ctrl_from_current_qpos()
        if not self.official_reset_sequence:
            mujoco.mj_forward(self.model, self.data)

        self.fixed_base_enabled = env_flag(
            self.config.debug.fixed_base.enabled_env,
            default=False,
        )
        self.fixed_base_qpos_slices = self._capture_fixed_base_qpos_slices()
        if self.fixed_base_enabled:
            print(
                "MuJoCo fixed-base debug mode enabled via "
                f"{self.config.debug.fixed_base.enabled_env}=1"
            )

        auto_reset = self.config.safety.auto_reset
        self.auto_reset_enabled = env_flag(auto_reset.enabled_env, default=False)
        self.auto_reset_min_base_height = auto_reset.min_base_height
        self.auto_reset_max_tilt_rad = auto_reset.max_tilt_rad
        self.auto_reset_cooldown_seconds = auto_reset.cooldown_seconds
        self.last_reset_time = time.monotonic()
        self.reset_count = 0
        if self.auto_reset_enabled:
            print(
                "MuJoCo auto-reset enabled via "
                f"{auto_reset.enabled_env}=1 "
                f"(min_base_height={self.auto_reset_min_base_height}, "
                f"max_tilt_rad={self.auto_reset_max_tilt_rad}, "
                f"cooldown_seconds={self.auto_reset_cooldown_seconds})"
            )

        self.viewer = MujocoViewerHandle(
            model=self.model,
            data=self.data,
            enabled=env_flag(self.config.viewer.enabled_env, default=False),
            show_left_ui=self.config.viewer.show_left_ui,
            show_right_ui=self.config.viewer.show_right_ui,
            follow_camera=env_flag("SORIDORMI_MUJOCO_FOLLOW_CAMERA", default=False),
            camera_distance=env_float("SORIDORMI_MUJOCO_CAMERA_DISTANCE", 1.4),
            camera_azimuth=env_float("SORIDORMI_MUJOCO_CAMERA_AZIMUTH", 135.0),
            camera_elevation=env_float("SORIDORMI_MUJOCO_CAMERA_ELEVATION", -20.0),
        )

        print(f"Loaded robot config: {self.config.robot_name}")
        print(f"Loaded MuJoCo model: {self.model_path}")
        print(f"MuJoCo timestep: {self.model.opt.timestep}")
        print(f"API step substeps: {self.substeps_per_api_step}")
        print(f"Actuators: {self.actuator_names}")
        if self.official_reset_sequence:
            print("Official Open Duck reset sequence enabled.")
        if self.official_sensor_mode:
            print("Official Open Duck sensor mode enabled.")
        if self.official_contact_mode:
            print("Official Open Duck body-contact mode enabled.")

    def reset(self) -> None:
        """Reset MuJoCo data to the configured reset_pose.

        The model stays loaded and the viewer stays open. The runtime can keep
        sending commands while the simulation returns to a known safe pose.
        """
        self.mujoco.mj_resetData(self.model, self.data)
        self._apply_home_keyframe_if_enabled()
        self._apply_configured_reset_pose()
        self._apply_startup_debug_options()
        if not self.last_reset_used_home_keyframe:
            self._initialize_ctrl_from_current_qpos()
        if not self.official_reset_sequence:
            self.mujoco.mj_forward(self.model, self.data)
        self.last_command = None
        self.reset_count += 1

        # If fixed-base mode is enabled, the fixed pose should match the new
        # reset pose rather than the previous fallen pose.
        if hasattr(self, "fixed_base_qpos_slices"):
            self.fixed_base_qpos_slices = self._capture_fixed_base_qpos_slices()

    def _apply_home_keyframe_if_enabled(self) -> None:
        """Apply MuJoCo keyframe("home") qpos and ctrl like official Open Duck.

        In official-compatibility mode we intentionally reproduce the upstream
        startup/reset sequence more closely:

          1. run one mj_step from fresh mjData/reset data,
          2. copy keyframe("home").qpos into data.qpos,
          3. copy keyframe("home").ctrl into data.ctrl,
          4. do not zero qvel here.

        The small-looking details matter because the pretrained walking policy
        consumes IMU sensors and contact state; changing the startup/reset sensor
        state can make Soridormi diverge from the official runner even when the
        ONNX model and motor targets are nearly identical.
        """
        self.last_reset_used_home_keyframe = False
        if not self.use_home_keyframe:
            return

        key_id = self._keyframe_id(self.home_keyframe_name)
        if key_id < 0:
            raise ValueError(
                f"SORIDORMI_MUJOCO_USE_HOME_KEYFRAME=1 but MuJoCo keyframe "
                f"{self.home_keyframe_name!r} was not found in {self.model_path}"
            )

        if self.official_reset_sequence:
            # This mirrors MJInferBase.__init__ before it applies home qpos/ctrl.
            # It advances sensors/data.time by one 2 ms step from reset/default
            # data, exactly like the official Open Duck inference base class.
            self.mujoco.mj_step(self.model, self.data)

        self.data.qpos[:] = np.asarray(self.model.key_qpos[key_id], dtype=float)
        if self.model.nu:
            self.data.ctrl[:] = np.asarray(self.model.key_ctrl[key_id], dtype=float)

        if self.model.nv and not self.official_reset_sequence:
            self.data.qvel[:] = 0.0

        self.last_reset_used_home_keyframe = True
        if self.official_reset_sequence:
            print(
                f"Applied official MuJoCo keyframe '{self.home_keyframe_name}' "
                "qpos/ctrl with Open Duck reset sequence."
            )
        else:
            print(f"Applied MuJoCo keyframe '{self.home_keyframe_name}' qpos/ctrl.")

    def _keyframe_id(self, name: str) -> int:
        return int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_KEY,
                name,
            )
        )

    def _apply_configured_reset_pose(self) -> None:
        """Apply optional reset_pose from the robot config.

        This sets the free-base qpos and named joint qpos before the first
        mj_forward() call. It makes simulation startup repeatable and lets us
        tune reset/default poses from YAML instead of changing backend code.
        """
        if self.last_reset_used_home_keyframe and env_flag(
            "SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE",
            default=True,
        ):
            print("Skipping configured reset_pose because MuJoCo home keyframe is active.")
            return

        reset_pose = self.config.reset_pose
        if reset_pose is None:
            return

        base_pose = reset_pose.base
        if base_pose is not None:
            if base_pose.position_xyz is not None:
                start, stop = self.config.base.qpos_xyz_slice
                self.data.qpos[start:stop] = np.array(base_pose.position_xyz, dtype=float)

            if base_pose.quat_wxyz is not None:
                start, stop = self.config.base.qpos_quat_wxyz_slice
                self.data.qpos[start:stop] = np.array(base_pose.quat_wxyz, dtype=float)

        for joint_name, value in reset_pose.joints.items():
            joint_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"reset_pose references unknown joint: {joint_name}")

            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_addr] = float(value)

        self.data.qvel[:] = 0.0
        print("Applied reset_pose from robot config.")

    def _apply_startup_debug_options(self) -> None:
        zero_gravity_env = self.config.debug.zero_gravity.enabled_env
        if env_flag(zero_gravity_env, default=False):
            self.model.opt.gravity[:] = 0.0
            print(f"MuJoCo zero-gravity debug mode enabled via {zero_gravity_env}=1")

    def _capture_fixed_base_qpos_slices(self) -> dict[str, np.ndarray]:
        base = self.config.base
        xyz_start, xyz_stop = base.qpos_xyz_slice
        quat_start, quat_stop = base.qpos_quat_wxyz_slice
        lin_start, lin_stop = base.qvel_linear_slice
        ang_start, ang_stop = base.qvel_angular_slice

        return {
            "qpos_xyz": np.array(self.data.qpos[xyz_start:xyz_stop], dtype=float),
            "qpos_quat_wxyz": np.array(self.data.qpos[quat_start:quat_stop], dtype=float),
            "qvel_linear": np.array(self.data.qvel[lin_start:lin_stop], dtype=float),
            "qvel_angular": np.array(self.data.qvel[ang_start:ang_stop], dtype=float),
        }

    def _enforce_fixed_base_debug(self) -> None:
        """Hold the floating base at its reset/start pose.

        This is intentionally a debug-only mode. It is useful for inspecting
        joint pose, actuator directions, and default-pose tuning while gravity
        remains enabled, but it should not be used as a realistic dynamics mode.
        """
        if not self.fixed_base_enabled:
            return

        base = self.config.base
        xyz_start, xyz_stop = base.qpos_xyz_slice
        quat_start, quat_stop = base.qpos_quat_wxyz_slice
        lin_start, lin_stop = base.qvel_linear_slice
        ang_start, ang_stop = base.qvel_angular_slice

        self.data.qpos[xyz_start:xyz_stop] = self.fixed_base_qpos_slices["qpos_xyz"]
        self.data.qpos[quat_start:quat_stop] = self.fixed_base_qpos_slices["qpos_quat_wxyz"]
        self.data.qvel[lin_start:lin_stop] = 0.0
        self.data.qvel[ang_start:ang_stop] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _base_height(self) -> float:
        start, _ = self.config.base.qpos_xyz_slice
        return float(self.data.qpos[start + 2])

    def _base_tilt_rad(self) -> float:
        """Return max absolute roll/pitch tilt from base quaternion.

        MuJoCo free-joint quaternion is stored as wxyz. Yaw is ignored because
        yawing does not mean the robot has fallen.
        """
        start, stop = self.config.base.qpos_quat_wxyz_slice
        w, x, y, z = [float(v) for v in self.data.qpos[start:stop]]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        return max(abs(roll), abs(pitch))

    def _auto_reset_if_needed(self) -> None:
        if not self.auto_reset_enabled:
            return

        # Fixed-base and zero-gravity debug modes are normally used to inspect
        # poses, so auto-reset would be noisy and unnecessary there.
        if self.fixed_base_enabled:
            return

        now = time.monotonic()
        if now - self.last_reset_time < self.auto_reset_cooldown_seconds:
            return

        height = self._base_height()
        tilt = self._base_tilt_rad()

        if height < self.auto_reset_min_base_height or tilt > self.auto_reset_max_tilt_rad:
            print(
                "Auto reset triggered: "
                f"height={height:.3f}, tilt={tilt:.3f} rad, "
                f"reset_count={self.reset_count + 1}"
            )
            self.reset()
            self.last_reset_time = now

    def _initialize_ctrl_from_current_qpos(self) -> None:
        """Initialize actuator controls to the model's current joint pose.

        Without this, MuJoCo position actuators can start with ctrl=0.0 even if
        the XML's initial qpos is not zero. That can snap the robot toward zero
        before the runtime sends its first command.
        """
        for actuator_index, qpos_addr in enumerate(self.qpos_addrs):
            target = float(self.data.qpos[qpos_addr])

            if self.config.control.clip_to_ctrlrange:
                target = float(
                    np.clip(target, self.ctrl_min[actuator_index], self.ctrl_max[actuator_index])
                )

            self.data.ctrl[actuator_index] = target

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
            self._enforce_fixed_base_debug()

        self._auto_reset_if_needed()

        if self.config.viewer.sync_every_api_step:
            self.viewer.sync()

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
            feet_contacts=self._read_feet_contacts(),
            feet_position_xyz=self._read_feet_positions(),
            base_position_xyz=self._slice(self.data.qpos, self.config.base.qpos_xyz_slice),
            base_quat_wxyz=self._slice(self.data.qpos, self.config.base.qpos_quat_wxyz_slice),
            actuator_ctrl=[float(x) for x in self.data.ctrl[: self.model.nu]],
        )

    def apply_command(self, command: MotorCommand) -> None:
        self.last_command = command

    def apply_visual_expression(self, command: VisualExpressionCommand) -> None:
        expression = command.expression
        intensity = float(command.intensity)
        if expression == "eyes_open":
            self._set_social_eye_visuals(open_alpha=intensity, closed_alpha=0.0)
        elif expression == "eyes_closed":
            self._set_social_eye_visuals(open_alpha=0.0, closed_alpha=intensity)
        else:  # pragma: no cover - pydantic Literal validates this first.
            raise ValueError(f"unsupported visual expression: {expression}")
        self.mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def _set_social_eye_visuals(self, *, open_alpha: float, closed_alpha: float) -> None:
        missing: list[str] = []
        for name, alpha in (
            (LEFT_EYE_NAME, open_alpha),
            (RIGHT_EYE_NAME, open_alpha),
            (LEFT_EYE_CLOSED_NAME, closed_alpha),
            (RIGHT_EYE_CLOSED_NAME, closed_alpha),
        ):
            geom_id = self._geom_id(name)
            if geom_id < 0:
                missing.append(name)
                continue
            rgba = np.array(self.model.geom_rgba[geom_id], dtype=float)
            rgba[3] = max(0.0, min(1.0, float(alpha)))
            self.model.geom_rgba[geom_id] = rgba
        if missing:
            raise ValueError(
                "MuJoCo model does not contain Soridormi social-eye geoms. "
                "Start the sim with --social-eyes or regenerate the social-eye scene. "
                f"missing={missing}"
            )

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
                target = float(
                    np.clip(target, self.ctrl_min[actuator_index], self.ctrl_max[actuator_index])
                )

            self.data.ctrl[actuator_index] = target

    def close(self) -> None:
        self.viewer.close()

    def _slice(self, vector: np.ndarray, bounds: tuple[int, int]) -> list[float]:
        start, stop = bounds
        return [float(x) for x in vector[start:stop]]

    def _read_base_imu(self) -> IMUState:
        base = self.config.base
        gyro_fallback = self._slice(self.data.qvel, base.qvel_angular_slice)
        accel_fallback = list(self.config.imu.accel_xyz_default)

        if self.official_sensor_mode:
            gyro = self._read_required_sensor(self.gyro_sensor_id, 3, "gyro")
            accel = self._read_required_sensor(self.accelerometer_sensor_id, 3, "accelerometer")
        else:
            gyro = self._read_sensor_or_qvel(
                sensor_id=self.gyro_sensor_id,
                expected_size=3,
                fallback=gyro_fallback,
            )
            accel = self._read_sensor_or_qvel(
                sensor_id=self.accelerometer_sensor_id,
                expected_size=3,
                fallback=accel_fallback,
            )

        return IMUState(
            quat_wxyz=self._slice(self.data.qpos, base.qpos_quat_wxyz_slice),
            gyro_xyz=gyro,
            accel_xyz=accel,
        )

    def _sensor_id(self, name: str) -> int:
        sensor_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_SENSOR,
            name,
        )
        return int(sensor_id)

    def _read_sensor_or_qvel(
        self,
        *,
        sensor_id: int,
        expected_size: int,
        fallback: list[float],
    ) -> list[float]:
        if sensor_id < 0:
            return [float(x) for x in fallback]

        addr = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        if dim < expected_size:
            return [float(x) for x in fallback]

        values = self.data.sensordata[addr : addr + expected_size]
        return [float(x) for x in values]

    def _read_required_sensor(
        self,
        sensor_id: int,
        expected_size: int,
        sensor_name: str,
    ) -> list[float]:
        if sensor_id < 0:
            raise RuntimeError(
                f"Official sensor mode requires MuJoCo sensor {sensor_name!r}, "
                f"but it was not found in {self.model_path}"
            )

        addr = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        if dim < expected_size:
            raise RuntimeError(
                f"MuJoCo sensor {sensor_name!r} has dim={dim}, "
                f"expected at least {expected_size}"
            )

        values = self.data.sensordata[addr : addr + expected_size]
        return [float(x) for x in values]

    def _read_feet_contacts(self) -> list[float]:
        contact = self.config.policy_observation.foot_contact
        left = self._check_named_body_contact(contact.left_body, contact.ground_body)
        right = self._check_named_body_contact(contact.right_body, contact.ground_body)

        # The official Open Duck inference uses body contact only:
        #   foot_assembly vs floor, foot_assembly_2 vs floor.
        # Keep geom fallback for non-official/generic configs, but disable it
        # in official mode so Soridormi's contact observation is bit-for-bit
        # closer to upstream.
        if not self.official_contact_mode:
            if not left:
                left = self._check_named_geom_contact(contact.left_geoms, contact.ground_geoms)
            if not right:
                right = self._check_named_geom_contact(contact.right_geoms, contact.ground_geoms)

        return [1.0 if left else 0.0, 1.0 if right else 0.0]

    def _read_feet_positions(self) -> list[list[float]] | None:
        contact = self.config.policy_observation.foot_contact

        # Backward compatibility: older M4.0/M4.3 configs/models did not have
        # left_site/right_site fields. Fall back to geom positions in that case.
        left_site = getattr(contact, "left_site", "")
        right_site = getattr(contact, "right_site", "")

        left = self._read_site_or_geom_position(left_site, contact.left_geoms)
        right = self._read_site_or_geom_position(right_site, contact.right_geoms)
        if left is None or right is None:
            return None
        return [left, right]

    def _read_site_or_geom_position(
        self,
        site_name: str,
        fallback_geoms: list[str],
    ) -> list[float] | None:
        site_id = self._site_id(site_name) if site_name else -1
        if site_id >= 0:
            return [float(x) for x in self.data.site_xpos[site_id]]

        for geom_name in fallback_geoms:
            geom_id = self._geom_id(geom_name)
            if geom_id >= 0:
                return [float(x) for x in self.data.geom_xpos[geom_id]]
        return None

    def _body_id(self, name: str) -> int:
        return int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
        )

    def _geom_id(self, name: str) -> int:
        return int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
        )

    def _site_id(self, name: str) -> int:
        return int(
            self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
        )

    def _check_named_body_contact(self, body_a: str, body_b: str) -> bool:
        body_a_id = self._body_id(body_a)
        body_b_id = self._body_id(body_b)
        if body_a_id < 0 or body_b_id < 0:
            return False

        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            geom1_body = int(self.model.geom_bodyid[contact.geom1])
            geom2_body = int(self.model.geom_bodyid[contact.geom2])
            if (geom1_body == body_a_id and geom2_body == body_b_id) or (
                geom1_body == body_b_id and geom2_body == body_a_id
            ):
                return True
        return False

    def _check_named_geom_contact(self, geoms_a: list[str], geoms_b: list[str]) -> bool:
        geom_ids_a = {self._geom_id(name) for name in geoms_a}
        geom_ids_b = {self._geom_id(name) for name in geoms_b}
        geom_ids_a.discard(-1)
        geom_ids_b.discard(-1)
        if not geom_ids_a or not geom_ids_b:
            return False

        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in geom_ids_a and geom2 in geom_ids_b) or (
                geom1 in geom_ids_b and geom2 in geom_ids_a
            ):
                return True
        return False
