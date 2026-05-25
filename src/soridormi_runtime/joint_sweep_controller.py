from __future__ import annotations

import math
import os
import time

from soridormi_api import MotorCommand, RobotState


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


class JointSweepController:
    """Sweep one actuator at a time for visual joint-direction validation.

    The controller records the first observed joint pose, holds all joints near
    that pose, and applies a smooth sine offset to one selected joint.

    Useful with:
      SORIDORMI_MUJOCO_ZERO_GRAVITY=1
      SORIDORMI_MUJOCO_VIEWER=1
    """

    def __init__(self) -> None:
        self.amplitude = _env_float("SORIDORMI_JOINT_SWEEP_AMPLITUDE", 0.20)
        self.period_seconds = _env_float("SORIDORMI_JOINT_SWEEP_PERIOD_SECONDS", 4.0)
        self.hold_seconds = _env_float("SORIDORMI_JOINT_SWEEP_HOLD_SECONDS", 1.0)
        self.kp = _env_float("SORIDORMI_JOINT_SWEEP_KP", 10.0)
        self.kd = _env_float("SORIDORMI_JOINT_SWEEP_KD", 0.5)
        self.loop = _env_bool("SORIDORMI_JOINT_SWEEP_LOOP", True)

        self.start_time = time.monotonic()
        self.initial_positions_by_name: dict[str, float] | None = None
        self.joint_names: list[str] | None = None
        self._last_printed_joint_index: int | None = None

    def compute(self, state: RobotState) -> MotorCommand:
        names = list(state.joints.names)
        n = len(names)

        if self.initial_positions_by_name is None:
            self.initial_positions_by_name = {
                name: float(pos)
                for name, pos in zip(state.joints.names, state.joints.positions)
            }
            self.joint_names = names
            print(f"Joint sweep initialized with {n} joints:")
            for i, name in enumerate(names):
                print(f"  {i:02d}: {name}")

        assert self.initial_positions_by_name is not None
        assert self.joint_names is not None

        elapsed = time.monotonic() - self.start_time
        segment_seconds = self.hold_seconds + self.period_seconds
        total_seconds = segment_seconds * len(self.joint_names)

        if self.loop:
            elapsed = elapsed % total_seconds
        else:
            elapsed = min(elapsed, total_seconds - 1e-6)

        joint_index = int(elapsed // segment_seconds)
        joint_index = max(0, min(joint_index, len(self.joint_names) - 1))

        segment_t = elapsed - joint_index * segment_seconds
        active_joint = self.joint_names[joint_index]

        if self._last_printed_joint_index != joint_index:
            self._last_printed_joint_index = joint_index
            print(f"\nSweeping joint {joint_index:02d}: {active_joint}")

        # First hold briefly at neutral, then sweep smoothly.
        if segment_t < self.hold_seconds:
            offset = 0.0
        else:
            phase = (segment_t - self.hold_seconds) / self.period_seconds
            offset = self.amplitude * math.sin(2.0 * math.pi * phase)

        target_positions: list[float] = []

        for i, name in enumerate(names):
            base = self.initial_positions_by_name.get(name, float(state.joints.positions[i]))

            if name == active_joint:
                target_positions.append(base + offset)
            else:
                target_positions.append(base)

        return MotorCommand(
            names=names,
            positions=target_positions,
            velocities=[0.0] * n,
            kp=[self.kp] * n,
            kd=[self.kd] * n,
            torques=[0.0] * n,
        )
