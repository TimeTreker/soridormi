from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.policy_factory import normalize_policy_backend
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile
from soridormi_runtime.residual_policy import ResidualOnnxPolicy
from soridormi_runtime.train_residual_policy import (
    COMMAND_STATE_MLP_PARAMETER_SIZE,
    COMMAND_STATE_FEATURE_SIZE,
    COMMAND_STATE_PARAMETER_SIZE,
    RESIDUAL_ACTOR_COMMAND_STATE,
    PHASE_CONTACT_PARAMETER_SIZE,
    RESIDUAL_ACTOR_PHASE_CONTACT,
    ResidualOptimizationConfig,
    _parse_training_command,
    _write_residual_profile,
    command_state_residual_action,
    command_state_residual_features,
    command_state_mlp_residual_action,
    episodic_clearance_adjustment,
    optimize_residual_bias,
    optimize_command_state_residual,
    optimize_command_state_mlp_residual,
    optimize_phase_contact_residual,
    phase_contact_residual_action,
)


JOINT_NAMES = [f"joint_{i}" for i in range(14)]


class FakeTeacher:
    def __init__(self) -> None:
        self.last_observation = np.ones((1, 101), dtype=np.float32)
        self.commands: list[list[float]] = []
        self.phases: list[list[float]] = []
        self.targets: list[list[float]] = []

    def compute_action(self, state: RobotState) -> np.ndarray:
        return np.asarray([0.2] * 14, dtype=np.float32)

    def set_command_vector(self, command: list[float]) -> None:
        self.commands.append(list(command))

    def set_imitation_phase(self, phase: list[float]) -> None:
        self.phases.append(list(phase))

    def set_motor_targets(self, joint_names: list[str], positions) -> None:
        self.targets.append([float(x) for x in positions])

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        return {name: 0.0 for name in state.joints.names}


class FakeSession:
    def __init__(self, path: str, providers=None) -> None:
        self.path = path
        self.providers = list(providers or ["CPUExecutionProvider"])

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, output_names, inputs):
        obs = inputs["obs"]
        assert obs.shape == (1, 101)
        return [np.asarray([[0.5] * 14], dtype=np.float32)]


class FakeContextTeacher(FakeTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.last_observation = np.ones((1, 104), dtype=np.float32)


class FakeContextSession(FakeSession):
    def run(self, output_names, inputs):
        assert inputs["obs"].shape == (1, 104)
        return [np.asarray([[0.5] * 14], dtype=np.float32)]


def _state() -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(names=JOINT_NAMES, positions=[0.0] * 14, velocities=[0.0] * 14, torques=[0.0] * 14),
        imu=IMUState(),
        base_position_xyz=[0.0, 0.0, 0.30],
        base_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )


def test_residual_backend_is_registered() -> None:
    assert normalize_policy_backend("residual_onnx") == "residual_onnx"
    assert normalize_policy_backend("teacher-residual") == "residual_onnx"


def test_residual_onnx_policy_combines_teacher_and_residual(tmp_path: Path) -> None:
    residual_path = tmp_path / "residual.onnx"
    residual_path.write_bytes(b"fake")
    policy = ResidualOnnxPolicy(
        policy_path=residual_path,
        teacher_policy=FakeTeacher(),
        residual_scale=0.1,
        residual_clip_abs=1.0,
        session_factory=FakeSession,
        providers=["CPUExecutionProvider"],
    )

    action = policy.compute_action(_state())

    assert action.tolist() == pytest.approx([0.25] * 14)
    assert policy.last_teacher_action is not None
    assert policy.last_residual_applied is not None
    assert policy.last_residual_applied.tolist() == pytest.approx([0.05] * 14)
    assert policy.last_debug is not None
    assert policy.last_debug["policy_kind"] == "residual_onnx"


def test_residual_onnx_policy_accepts_context_teacher_observation(tmp_path: Path) -> None:
    residual_path = tmp_path / "residual_context.onnx"
    residual_path.write_bytes(b"fake")
    policy = ResidualOnnxPolicy(
        policy_path=residual_path,
        teacher_policy=FakeContextTeacher(),
        residual_scale=0.1,
        session_factory=FakeContextSession,
        providers=["CPUExecutionProvider"],
    )

    action = policy.compute_action(_state())

    assert action.tolist() == pytest.approx([0.25] * 14)
    assert policy.last_observation is not None
    assert policy.last_observation.shape == (1, 104)


def test_cem_residual_optimizer_improves_toward_target() -> None:
    target = np.asarray([0.3] * 14, dtype=np.float32)

    def evaluate(candidate: np.ndarray) -> float:
        return -float(np.mean((candidate - target) ** 2))

    result = optimize_residual_bias(
        evaluate,
        config=ResidualOptimizationConfig(
            iterations=6,
            population=32,
            elite_fraction=0.25,
            initial_std=0.4,
            min_std=0.01,
            seed=7,
            include_zero_candidate=False,
        ),
    )

    assert result.best_score > -0.05
    assert np.mean(result.best_residual) == pytest.approx(0.3, abs=0.2)
    assert len(result.iterations) == 6


def test_phase_contact_residual_actor_uses_canonical_observation_fields() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [1.0, 0.0, 0.5, -0.5]
    weights = np.zeros((5, 14), dtype=np.float32)
    weights[1, 0] = 0.4
    weights[3, 1] = 0.6

    action = phase_contact_residual_action(observation, weights.reshape(-1))

    assert action[0] == pytest.approx(np.tanh(0.4))
    assert action[1] == pytest.approx(np.tanh(0.3))
    assert action[2:].tolist() == pytest.approx([0.0] * 12)


def test_phase_contact_optimizer_uses_full_actor_parameter_vector() -> None:
    target = np.asarray([0.1] * PHASE_CONTACT_PARAMETER_SIZE, dtype=np.float32)

    result = optimize_phase_contact_residual(
        lambda candidate: -float(np.mean((candidate - target) ** 2)),
        config=ResidualOptimizationConfig(
            iterations=2,
            population=8,
            initial_std=0.2,
            seed=3,
            include_zero_candidate=False,
        ),
    )

    assert len(result.best_residual) == PHASE_CONTACT_PARAMETER_SIZE
    assert len(result.final_mean) == PHASE_CONTACT_PARAMETER_SIZE


def test_command_state_actor_uses_command_joint_state_and_history() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[6:9] = [0.2, -0.1, 0.3]
    observation[97:101] = [1.0, 0.0, 0.5, -0.5]
    observation[[15, 16, 17, 24, 25, 26]] = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
    observation[[43, 44, 45, 52, 53, 54]] = [0.4, 0.5, 0.6, -0.4, -0.5, -0.6]
    features = command_state_residual_features(observation)
    weights = np.zeros((COMMAND_STATE_FEATURE_SIZE, 6), dtype=np.float32)
    weights[1, 0] = 2.0
    weights[-1, 1] = 1.0

    action = command_state_residual_action(observation, weights.reshape(-1))

    assert features.shape == (COMMAND_STATE_FEATURE_SIZE,)
    assert action[2] == pytest.approx(np.tanh(0.4))
    assert action[3] == pytest.approx(np.tanh(-0.6))
    assert action[[0, 1, 5, 6, 7, 8, 9, 10]].tolist() == pytest.approx([0.0] * 8)


def test_command_state_optimizer_uses_full_parameter_vector() -> None:
    result = optimize_command_state_residual(
        lambda candidate: -float(np.mean(candidate**2)),
        config=ResidualOptimizationConfig(iterations=1, population=3, seed=5),
    )

    assert len(result.best_residual) == COMMAND_STATE_PARAMETER_SIZE


def test_command_state_mlp_actor_preserves_linear_warm_start() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[6] = 0.2
    linear = np.zeros(COMMAND_STATE_PARAMETER_SIZE, dtype=np.float32)
    linear.reshape(COMMAND_STATE_FEATURE_SIZE, 6)[1, 0] = 2.0
    parameters = np.zeros(COMMAND_STATE_MLP_PARAMETER_SIZE, dtype=np.float32)
    parameters[:COMMAND_STATE_PARAMETER_SIZE] = linear

    action = command_state_mlp_residual_action(observation, parameters)

    assert action[2] == pytest.approx(np.tanh(0.4))
    assert action[[0, 1, 5, 6, 7, 8, 9, 10]].tolist() == pytest.approx([0.0] * 8)


def test_command_state_mlp_optimizer_accepts_warm_start() -> None:
    initial = np.full(COMMAND_STATE_MLP_PARAMETER_SIZE, 0.1, dtype=np.float32)
    result = optimize_command_state_mlp_residual(
        lambda candidate: -float(np.mean((candidate - initial) ** 2)),
        config=ResidualOptimizationConfig(iterations=1, population=3, seed=9),
        initial_mean=initial,
    )

    assert result.best_score == pytest.approx(0.0)
    assert result.best_residual == pytest.approx(initial.tolist())


def test_episodic_clearance_adjustment_matches_gate_direction() -> None:
    low = episodic_clearance_adjustment(
        [0.005, 0.010],
        target_clearance=0.015,
        clearance_weight=2.0,
        low_clearance_penalty_weight=1.0,
    )
    passing = episodic_clearance_adjustment(
        [0.015, 0.020],
        target_clearance=0.015,
        clearance_weight=2.0,
        low_clearance_penalty_weight=1.0,
    )

    assert passing > low
    assert episodic_clearance_adjustment(
        [],
        target_clearance=0.015,
        clearance_weight=2.0,
        low_clearance_penalty_weight=1.0,
    ) == 0.0


def test_parse_training_command_builds_velocity_command() -> None:
    command = _parse_training_command("0.09,0.0,0.12")

    assert command.x_velocity == pytest.approx(0.09)
    assert command.y_velocity == pytest.approx(0.0)
    assert command.yaw_velocity == pytest.approx(0.12)


def test_write_residual_profile_sets_runtime_kind(tmp_path: Path, monkeypatch) -> None:
    template_path = tmp_path / "teacher.yaml"
    template_path.write_text(
        """
name: teacher
model:
  path: /tmp/teacher.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: [1, 101]
  output_shape: [1, 14]
runtime:
  control_hz: 50
command:
  x: 0.15
phase: {}
action_mapping: {}
observation: {}
simulator: {}
logging: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("soridormi_runtime.create_policy_profile.PolicyProfile.load", lambda _: PolicyProfile(
        name="teacher",
        description="teacher",
        path=template_path,
        payload={
            "name": "teacher",
            "model": {"path": "/tmp/teacher.onnx", "input_name": "obs", "output_name": "continuous_actions", "input_shape": [1, 101], "output_shape": [1, 14]},
            "runtime": {"control_hz": 50},
            "command": {"x": 0.15},
            "phase": {},
            "action_mapping": {},
            "observation": {},
            "simulator": {},
            "logging": {},
        },
        model=PolicyModelSpec(path="/tmp/teacher.onnx"),
    ))
    # Avoid depending on robot YAML contract for this unit test.
    monkeypatch.setattr("soridormi_runtime.create_policy_profile.build_policy_contract", lambda *a, **k: type("C", (), {
        "ok": True,
        "errors": [],
        "observation": {"size": 101},
        "action": {"size": 14, "joint_order": JOINT_NAMES},
    })())

    path = _write_residual_profile(
        profile_name="residual_test",
        teacher_profile="teacher",
        residual_onnx_path="/data/residual.onnx",
        output_dir=tmp_path,
        description="test residual",
        residual_scale=0.05,
        residual_clip_abs=1.0,
        final_action_clip_abs=None,
        force=False,
        actor_kind=RESIDUAL_ACTOR_COMMAND_STATE,
    )

    text = path.read_text(encoding="utf-8")
    assert "kind: residual_onnx" in text
    assert "teacher_profile: teacher" in text
    assert "actor_kind: command_state" in text
    assert "residual_scale: 0.05" in text


def test_write_residual_profile_inherits_context_input_shape(tmp_path: Path, monkeypatch) -> None:
    teacher = PolicyProfile(
        name="context_teacher",
        description="context teacher",
        path=tmp_path / "context_teacher.yaml",
        payload={
            "name": "context_teacher",
            "model": {
                "path": "/tmp/context_teacher.onnx",
                "input_name": "obs",
                "output_name": "continuous_actions",
                "input_shape": [1, 104],
                "output_shape": [1, 14],
                "input_mode": "context_stage1_command",
            },
        },
        model=PolicyModelSpec(
            path="/tmp/context_teacher.onnx",
            input_shape=[1, 104],
            input_mode="context_stage1_command",
        ),
    )
    monkeypatch.setattr(
        "soridormi_runtime.train_residual_policy.PolicyProfile.load",
        lambda _: teacher,
    )
    monkeypatch.setattr(
        "soridormi_runtime.create_policy_profile.PolicyProfile.load",
        lambda _: teacher,
    )
    monkeypatch.setattr(
        "soridormi_runtime.create_policy_profile.build_policy_contract",
        lambda *a, **k: type(
            "C",
            (),
            {
                "ok": True,
                "errors": [],
                "observation": {"size": 101},
                "action": {"size": 14, "joint_order": JOINT_NAMES},
            },
        )(),
    )

    path = _write_residual_profile(
        profile_name="residual_context",
        teacher_profile="context_teacher",
        residual_onnx_path="/data/residual_context.onnx",
        output_dir=tmp_path,
        description=None,
        residual_scale=0.05,
        residual_clip_abs=1.0,
        final_action_clip_abs=None,
        force=False,
    )

    text = path.read_text(encoding="utf-8")
    assert "input_shape:\n  - 1\n  - 104" in text
    assert "input_mode: context_stage1_command" in text
