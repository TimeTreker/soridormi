from __future__ import annotations

from pathlib import Path

import pytest
from types import SimpleNamespace

import numpy as np

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.rl_finetune_env import ResidualActionConfig, RlFineTuneEnv, run_zero_residual_smoke
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile


JOINT_NAMES = [f"joint_{i}" for i in range(14)]


def _state(time: float, x: float = 0.0, y: float = 0.0) -> RobotState:
    return RobotState(
        time=time,
        joints=JointState(
            names=JOINT_NAMES,
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            torques=[0.0] * 14,
        ),
        imu=IMUState(),
        feet_contacts=[1.0, 1.0],
        base_position_xyz=[x, y, 0.3],
    )


class FakePolicy:
    def __init__(self) -> None:
        self.commands: list[list[float]] = []
        self.phases: list[list[float]] = []
        self.targets: list[list[float]] = []

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        return {name: 0.0 for name in state.joints.names}

    def set_command_vector(self, command: list[float]) -> None:
        self.commands.append(list(command))

    def set_imitation_phase(self, phase: list[float]) -> None:
        self.phases.append(list(phase))

    def compute_action(self, state: RobotState) -> np.ndarray:
        return np.asarray([0.1] * 14, dtype=np.float32)

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        self.targets.append([float(x) for x in positions])

    def get_observation(self) -> list[float]:
        return [0.0] * 101


class FakeMapper:
    def __init__(self) -> None:
        self.defaults: dict[str, float] = {}
        self.actions: list[list[float]] = []

    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        self.defaults.update(positions_by_name)

    def reset_targets(self) -> None:
        pass

    def action_to_command(self, action, state=None, dt=None) -> MotorCommand:
        arr = np.asarray(action, dtype=np.float32).reshape(14)
        self.actions.append([float(x) for x in arr])
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
        self.reset_calls = 0

    def reset(self) -> str:
        self.index = 0
        self.reset_calls += 1
        return "reset"

    def read_state(self) -> RobotState:
        return _state(0.0, x=0.0)

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self.index += 1
        return _state(0.02 * self.index, x=0.01 * self.index)


def _profile(tmp_path: Path) -> PolicyProfile:
    return PolicyProfile(
        name="teacher",
        description="test teacher",
        path=tmp_path / "teacher.yaml",
        payload={"runtime": {"control_hz": 50}, "model": {"path": str(tmp_path / "teacher.onnx")}},
        model=PolicyModelSpec(path=str(tmp_path / "teacher.onnx")),
    )


def test_residual_action_config_clips_and_scales() -> None:
    teacher = np.asarray([0.2] * 14, dtype=np.float32)
    residual = np.asarray([2.0] * 14, dtype=np.float32)
    applied, final = ResidualActionConfig(residual_scale=0.1, residual_clip_abs=0.5).apply(teacher, residual)

    assert applied.tolist() == pytest.approx([0.05] * 14)
    assert np.allclose(final, np.asarray([0.25] * 14, dtype=np.float32))


def test_rl_finetune_env_steps_teacher_plus_residual(tmp_path: Path) -> None:
    policy = FakePolicy()
    mapper = FakeMapper()
    robot = FakeRobot()
    env = RlFineTuneEnv(
        profile=_profile(tmp_path),
        robot=robot,
        policy=policy,
        mapper=mapper,
        command=SimpleNamespace(as_list=lambda: [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        phase_generator=SimpleNamespace(advance_and_as_list=lambda: [0.0, 1.0]),
        residual_config=ResidualActionConfig(residual_scale=0.2, residual_clip_abs=1.0),
    )

    env.reset()
    step = env.step([0.5] * 14)

    assert robot.reset_calls == 1
    assert step.step_index == 0
    assert step.residual_action == pytest.approx([0.1] * 14)
    assert np.allclose(step.final_action, [0.2] * 14)
    assert np.allclose(mapper.actions[-1], [0.2] * 14)
    assert step.metrics["forward_delta_x"] == 0.01
    assert policy.commands[-1][0] == 0.15
    assert policy.phases[-1] == [0.0, 1.0]


def test_rl_finetune_env_resolves_observation_conditioned_residual(tmp_path: Path) -> None:
    policy = FakePolicy()
    policy.get_observation = lambda: [0.25] * 101  # type: ignore[method-assign]
    mapper = FakeMapper()
    env = RlFineTuneEnv(
        profile=_profile(tmp_path),
        robot=FakeRobot(),
        policy=policy,
        mapper=mapper,
        command=SimpleNamespace(as_list=lambda: [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        phase_generator=SimpleNamespace(advance_and_as_list=lambda: [0.0, 1.0]),
        residual_config=ResidualActionConfig(residual_scale=0.2),
    )

    env.reset()
    step = env.step(lambda observation: [float(observation[-1])] * 14)

    assert step.residual_action == pytest.approx([0.05] * 14)
    assert step.final_action == pytest.approx([0.15] * 14)


def test_run_zero_residual_smoke_writes_json(tmp_path: Path, monkeypatch) -> None:
    def fake_env(**kwargs):
        return RlFineTuneEnv(
            profile=_profile(tmp_path),
            robot=FakeRobot(),
            policy=FakePolicy(),
            mapper=FakeMapper(),
            command=SimpleNamespace(as_list=lambda: [0.0] * 7),
            phase_generator=SimpleNamespace(advance_and_as_list=lambda: [0.0, 0.0]),
            residual_config=kwargs["residual_config"],
            reset_on_start=kwargs.get("reset_on_start", True),
        )

    monkeypatch.setattr("soridormi_runtime.rl_finetune_env.RlFineTuneEnv", fake_env)
    output = tmp_path / "smoke.json"
    result = run_zero_residual_smoke(
        profile="teacher",
        steps=2,
        residual_config=ResidualActionConfig(),
        output_path=output,
    )

    assert result.ok
    assert result.steps_completed == 2
    assert output.exists()
    assert "transitions" in output.read_text(encoding="utf-8")


def test_rl_finetune_env_accepts_context_policy_input(tmp_path: Path) -> None:
    profile = PolicyProfile(
        name="context_teacher",
        description="test context teacher",
        path=tmp_path / "context_teacher.yaml",
        payload={"runtime": {"control_hz": 50}},
        model=PolicyModelSpec(
            path=str(tmp_path / "context_teacher.onnx"),
            input_shape=[1, 104],
            input_mode="context_command_v1",
        ),
    )
    policy = FakePolicy()
    policy.get_observation = lambda: [0.0] * 104  # type: ignore[method-assign]
    env = RlFineTuneEnv(
        profile=profile,
        robot=FakeRobot(),
        policy=policy,
        mapper=FakeMapper(),
        command=SimpleNamespace(as_list=lambda: [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        phase_generator=SimpleNamespace(advance_and_as_list=lambda: [0.0, 1.0]),
    )

    env.reset()
    step = env.step([0.0] * 14)

    assert step.observation is not None
    assert len(step.observation) == 104
