from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.action_mapper import PolicyActionMapper
from soridormi_runtime.onnx_policy import OnnxPolicy, resolve_policy_path
from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand


class PolicyLike(Protocol):
    def compute_action(self, state: RobotState) -> np.ndarray:
        ...


class MapperLike(Protocol):
    last_motor_targets_by_name: dict[str, float]

    def action_to_command(
        self,
        action: np.ndarray | list[float],
        state: RobotState | None = None,
        dt: float | None = None,
    ) -> MotorCommand:
        ...


class OnnxPolicyController:
    """Experimental ONNX policy runtime controller.

    M3.5 adds the dynamic pieces required by the Open Duck policy observation:
      - 7D command vector from environment variables
      - gait/imitation phase oscillator
      - action-to-motor speed limiting via PolicyActionMapper
      - motor target feedback into the next observation

    M3.6 adds policy debug payloads for MCAP/JSONL logging.

    It is intentionally explicit and opt-in. Enable it only with:

        SORIDORMI_RUNTIME_MODE=onnx_policy
    """

    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        robot_config_path: str | os.PathLike[str] | None = None,
        policy: PolicyLike | None = None,
        mapper: MapperLike | None = None,
        command: PolicyCommand | None = None,
        phase_generator: GaitPhaseGenerator | None = None,
        control_hz: float | None = None,
    ) -> None:
        self.robot_config_path = robot_config_path or os.environ.get("SORIDORMI_ROBOT_CONFIG")
        self.policy_path = resolve_policy_path(policy_path)
        self.control_hz = float(control_hz or os.environ.get("CONTROL_HZ", "50"))
        self.dt = 1.0 / self.control_hz

        self.policy: PolicyLike = policy or OnnxPolicy(
            policy_path=self.policy_path,
            robot_config_path=self.robot_config_path,
        )
        self.mapper: MapperLike = mapper or PolicyActionMapper.from_robot_config(
            path=self.robot_config_path,
            use_env_overrides=True,
        )
        self.command = command or PolicyCommand.from_env()
        self.phase_generator = phase_generator or GaitPhaseGenerator.from_env()

        self.step_count = 0
        self.last_action: np.ndarray | None = None
        self.last_command: MotorCommand | None = None
        self.last_phase: list[float] = [0.0, 0.0]
        self.last_policy_debug: dict[str, Any] | None = None
        self.last_policy_observation_stats: dict[str, Any] | None = None

    def compute(self, state: RobotState) -> MotorCommand:
        current_step = self.step_count
        command_vector = self.command.as_list()
        phase_vector = self.phase_generator.as_list()
        self.last_phase = list(phase_vector)

        self._set_policy_command(command_vector)
        self._set_policy_phase(phase_vector)

        action = np.asarray(self.policy.compute_action(state), dtype=np.float32)

        if action.shape == (1, 14):
            action = action.reshape(14)
        if action.shape != (14,):
            raise RuntimeError(f"ONNX policy action must have shape (14,), got {action.shape}")

        try:
            command = self.mapper.action_to_command(action, state=state, dt=self.dt)
        except TypeError as exc:
            # Compatibility for tests or custom mappers written before M3.5,
            # where action_to_command(action, state=state) did not accept dt.
            if "unexpected keyword argument 'dt'" not in str(exc):
                raise
            command = self.mapper.action_to_command(action, state=state)
        self._set_policy_motor_targets(command)

        self.last_action = action.copy()
        self.last_command = command
        self.last_policy_observation_stats = self._read_policy_observation_stats()
        self.last_policy_debug = self._build_policy_debug(
            step_count=current_step,
            state=state,
            command=command,
            command_vector=command_vector,
            phase_vector=phase_vector,
            action=action,
        )
        self.step_count += 1

        return command

    def _set_policy_command(self, command_vector: list[float]) -> None:
        setter = getattr(self.policy, "set_command_vector", None)
        if callable(setter):
            setter(command_vector)

    def _set_policy_phase(self, phase_vector: list[float]) -> None:
        setter = getattr(self.policy, "set_imitation_phase", None)
        if callable(setter):
            setter(phase_vector)

    def _set_policy_motor_targets(self, command: MotorCommand) -> None:
        setter = getattr(self.policy, "set_motor_targets", None)
        if callable(setter):
            setter(command.names, command.positions)

    def _read_policy_observation_stats(self) -> dict[str, Any] | None:
        getter = getattr(self.policy, "get_observation_stats", None)
        if callable(getter):
            stats = getter()
            if stats is not None:
                return dict(stats)

        stats = getattr(self.policy, "last_observation_stats", None)
        if isinstance(stats, dict):
            return dict(stats)
        return None

    def _build_policy_debug(
        self,
        *,
        step_count: int,
        state: RobotState,
        command: MotorCommand,
        command_vector: list[float],
        phase_vector: list[float],
        action: np.ndarray,
    ) -> dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(14)
        motor_targets = np.asarray(command.positions, dtype=np.float32)
        joint_positions = np.asarray(state.joints.positions, dtype=np.float32)
        joint_velocities = np.asarray(state.joints.velocities, dtype=np.float32)

        return {
            "step_count": int(step_count),
            "robot_time": float(state.time),
            "control_hz": float(self.control_hz),
            "dt": float(self.dt),
            "command": [float(x) for x in command_vector],
            "phase": [float(x) for x in phase_vector],
            "action_min": float(action_arr.min()),
            "action_max": float(action_arr.max()),
            "action_mean": float(action_arr.mean()),
            "action_std": float(action_arr.std()),
            "motor_target_min": float(motor_targets.min()) if motor_targets.size else 0.0,
            "motor_target_max": float(motor_targets.max()) if motor_targets.size else 0.0,
            "motor_target_mean": float(motor_targets.mean()) if motor_targets.size else 0.0,
            "joint_position_min": float(joint_positions.min()) if joint_positions.size else 0.0,
            "joint_position_max": float(joint_positions.max()) if joint_positions.size else 0.0,
            "joint_velocity_min": float(joint_velocities.min()) if joint_velocities.size else 0.0,
            "joint_velocity_max": float(joint_velocities.max()) if joint_velocities.size else 0.0,
            "action_scale": self._mapper_config_float("action_scale"),
            "max_motor_velocity": self._mapper_config_float("max_motor_velocity"),
            "speed_limit_enabled": self._mapper_config_bool("speed_limit_enabled"),
        }

    def _mapper_config_float(self, name: str) -> float | None:
        config = getattr(self.mapper, "config", None)
        if config is None or not hasattr(config, name):
            return None
        return float(getattr(config, name))

    def _mapper_config_bool(self, name: str) -> bool | None:
        config = getattr(self.mapper, "config", None)
        if config is None or not hasattr(config, name):
            return None
        return bool(getattr(config, name))

    def get_policy_log_payload(self) -> dict[str, Any]:
        return {
            "policy_action": None
            if self.last_action is None
            else [float(x) for x in self.last_action.tolist()],
            "policy_debug": self.last_policy_debug,
            "policy_observation_stats": self.last_policy_observation_stats,
        }

    def describe(self) -> dict[str, object]:
        return {
            "policy_path": str(self.policy_path),
            "robot_config_path": str(self.robot_config_path) if self.robot_config_path else None,
            "step_count": self.step_count,
            "control_hz": self.control_hz,
            "dt": self.dt,
            "command": self.command.describe(),
            "phase": self.phase_generator.describe(),
            "last_phase": list(self.last_phase),
        }
