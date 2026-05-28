from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.rl_finetune_env import ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.walking_reward import WalkingRewardConfig, compute_walking_reward


JOINT_NAMES = [f"joint_{i}" for i in range(14)]

from soridormi_api import MotorCommand
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile


class FakePolicy:
    def __init__(self) -> None:
        self.commands: list[list[float]] = []
        self.phases: list[list[float]] = []

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        return {name: 0.0 for name in state.joints.names}

    def set_command_vector(self, command: list[float]) -> None:
        self.commands.append(list(command))

    def set_imitation_phase(self, phase: list[float]) -> None:
        self.phases.append(list(phase))

    def compute_action(self, state: RobotState) -> np.ndarray:
        return np.asarray([0.1] * 14, dtype=np.float32)

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        pass


class FakeMapper:
    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        pass

    def reset_targets(self) -> None:
        pass

    def action_to_command(self, action, state=None, dt=None) -> MotorCommand:
        arr = np.asarray(action, dtype=np.float32).reshape(14)
        return MotorCommand(
            names=list(JOINT_NAMES),
            positions=[float(x) for x in arr],
            velocities=[0.0] * 14,
            kp=[10.0] * 14,
            kd=[0.5] * 14,
            torques=[0.0] * 14,
        )


class FakeRobot:
    def __init__(self) -> None:
        self.index = 0

    def reset(self) -> str:
        self.index = 0
        return "reset"

    def read_state(self) -> RobotState:
        return _state_with_pose(0.0, 0.0, 0.0)

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self.index += 1
        return _state_with_pose(0.02 * self.index, 0.003 * self.index, 0.0)


def _profile(tmp_path: Path) -> PolicyProfile:
    return PolicyProfile(
        name="teacher",
        description="test teacher",
        path=tmp_path / "teacher.yaml",
        payload={"runtime": {"control_hz": 50}, "model": {"path": str(tmp_path / "teacher.onnx")}},
        model=PolicyModelSpec(path=str(tmp_path / "teacher.onnx")),
    )



def _quat_from_roll(roll: float) -> list[float]:
    return [math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0]


def _state_with_pose(time: float, x: float, y: float, z: float = 0.30, quat: list[float] | None = None) -> RobotState:
    return RobotState(
        time=time,
        joints=JointState(names=JOINT_NAMES, positions=[0.0] * 14, velocities=[0.0] * 14, torques=[0.0] * 14),
        imu=IMUState(),
        feet_contacts=[1.0, 1.0],
        base_position_xyz=[x, y, z],
        base_quat_wxyz=quat or [1.0, 0.0, 0.0, 0.0],
    )


def test_walking_reward_prefers_commanded_forward_velocity() -> None:
    command = PolicyCommand(x_velocity=0.15, y_velocity=0.0, yaw_velocity=0.0)
    good = compute_walking_reward(
        _state_with_pose(0.0, 0.0, 0.0),
        _state_with_pose(1.0, 0.15, 0.0),
        command=command,
        final_action=[0.0] * 14,
        residual_action=[0.0] * 14,
    )
    slow = compute_walking_reward(
        _state_with_pose(0.0, 0.0, 0.0),
        _state_with_pose(1.0, 0.0, 0.0),
        command=command,
        final_action=[0.0] * 14,
        residual_action=[0.0] * 14,
    )

    assert good.reward > slow.reward
    assert good.terms["forward_tracking"] > slow.terms["forward_tracking"]
    assert not good.terminated


def test_walking_reward_penalizes_fall_and_terminates() -> None:
    result = compute_walking_reward(
        _state_with_pose(0.0, 0.0, 0.0),
        _state_with_pose(0.02, 0.0, 0.0, z=0.08, quat=_quat_from_roll(1.4)),
        command=PolicyCommand(x_velocity=0.15),
        config=WalkingRewardConfig(fall_height=0.14, min_upright=0.65),
    )

    assert result.terminated
    assert result.terms["fall_penalty"] < 0.0
    assert result.diagnostics["fallen"] is True


def test_walking_reward_penalizes_large_residual_and_action_rate() -> None:
    before = _state_with_pose(0.0, 0.0, 0.0)
    after = _state_with_pose(0.02, 0.003, 0.0)
    small = compute_walking_reward(
        before,
        after,
        command=PolicyCommand(x_velocity=0.15),
        final_action=[0.1] * 14,
        residual_action=[0.01] * 14,
        previous_final_action=[0.1] * 14,
    )
    large = compute_walking_reward(
        before,
        after,
        command=PolicyCommand(x_velocity=0.15),
        final_action=[1.0] * 14,
        residual_action=[1.0] * 14,
        previous_final_action=[0.0] * 14,
    )

    assert large.reward < small.reward
    assert large.terms["residual_l2_penalty"] < small.terms["residual_l2_penalty"]
    assert large.terms["action_rate_penalty"] < small.terms["action_rate_penalty"]


def test_rl_finetune_env_includes_reward_metrics(tmp_path: Path) -> None:
    env = RlFineTuneEnv(
        profile=_profile(tmp_path),
        robot=FakeRobot(),
        policy=FakePolicy(),
        mapper=FakeMapper(),
        command=SimpleNamespace(as_list=lambda: [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], x_velocity=0.15, y_velocity=0.0, yaw_velocity=0.0),
        phase_generator=SimpleNamespace(advance_and_as_list=lambda: [0.0, 1.0]),
        residual_config=ResidualActionConfig(residual_scale=0.1),
    )

    env.reset()
    step = env.step(np.zeros(14, dtype=np.float32))

    assert "reward" in step.metrics
    assert "reward_terms" in step.metrics
    assert "reward_diagnostics" in step.metrics
    assert step.metrics["reward_diagnostics"]["target_vx"] == 0.15
