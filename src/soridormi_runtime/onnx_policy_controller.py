from __future__ import annotations

import os
from typing import Protocol

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
        )
        self.command = command or PolicyCommand.from_env()
        self.phase_generator = phase_generator or GaitPhaseGenerator.from_env()

        self.step_count = 0
        self.last_action: np.ndarray | None = None
        self.last_command: MotorCommand | None = None
        self.last_phase: list[float] = [0.0, 0.0]

    def compute(self, state: RobotState) -> MotorCommand:
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

        self.step_count += 1
        self.last_action = action.copy()
        self.last_command = command

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
