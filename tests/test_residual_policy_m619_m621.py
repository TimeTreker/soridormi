from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import soridormi_runtime.train_residual_policy as train_residual_policy_module
from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.policy_factory import normalize_policy_backend
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile
from soridormi_runtime.residual_policy import ResidualOnnxPolicy
from soridormi_runtime.walking_reward import WalkingRewardConfig
from soridormi_runtime.train_residual_policy import (
    COMMAND_STATE_MLP_PARAMETER_SIZE,
    COMMAND_STATE_FEATURE_SIZE,
    COMMAND_STATE_PARAMETER_SIZE,
    COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE,
    COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE,
    CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE,
    CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE,
    CONTACT_PHASE_LIFT_PARAMETER_SIZE,
    RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT,
    RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT,
    RESIDUAL_ACTOR_CONTACT_PHASE_LIFT,
    RESIDUAL_ACTOR_COMMAND_STATE,
    PHASE_CONTACT_PARAMETER_SIZE,
    RESIDUAL_ACTOR_PHASE_CONTACT,
    ResidualOptimizationConfig,
    ResidualTrainingCommand,
    ResidualTrainingSegment,
    ResidualTrainingSequence,
    _build_episode_score_breakdown,
    _build_episode_segment_breakdown,
    _normalize_training_commands,
    _normalize_reference_low_clearance_ratios,
    _normalize_training_sequences,
    _parse_training_command,
    _parse_training_command_spec,
    _parse_training_sequence,
    _residual_source_from_parameters,
    _score_for_aggregation,
    _validate_clearance_quantile,
    _validate_score_normalization,
    _write_residual_profile,
    aggregate_training_scores,
    command_state_residual_action,
    command_state_residual_features,
    command_state_mlp_residual_action,
    command_contact_phase_lift_residual_action,
    command_contact_phase_lift_residual_features,
    contact_phase_harmonic_lift_residual_action,
    contact_phase_harmonic_lift_residual_features,
    contact_phase_lift_residual_action,
    contact_phase_lift_residual_features,
    episodic_clearance_adjustment,
    low_clearance_regression_adjustment,
    optimize_residual_bias,
    optimize_command_state_residual,
    optimize_command_state_mlp_residual,
    optimize_command_contact_phase_lift_residual,
    optimize_contact_phase_harmonic_lift_residual,
    optimize_contact_phase_lift_residual,
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


def test_residual_onnx_policy_can_use_residual_teacher_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    residual_path = tmp_path / "outer_residual.onnx"
    residual_path.write_bytes(b"fake")
    teacher_profile = PolicyProfile(
        name="inner_residual",
        description="inner residual teacher",
        path=tmp_path / "inner_residual.yaml",
        payload={
            "name": "inner_residual",
            "model": {
                "path": str(tmp_path / "inner_residual.onnx"),
                "kind": "residual_onnx",
                "input_shape": [1, 101],
                "output_shape": [1, 14],
            },
            "residual_policy": {
                "teacher_profile": "base_teacher",
                "residual_scale": 0.1,
                "residual_clip_abs": 1.0,
            },
        },
        model=PolicyModelSpec(
            path=str(tmp_path / "inner_residual.onnx"),
            kind="residual_onnx",
            input_shape=[1, 101],
            output_shape=[1, 14],
        ),
    )
    created: list[dict[str, str | None]] = []

    def fake_make_runtime_policy(*, policy_path=None, robot_config_path=None):
        created.append(
            {
                "policy_path": str(policy_path),
                "backend": train_residual_policy_module.os.environ.get("SORIDORMI_POLICY_BACKEND"),
            }
        )
        return FakeTeacher()

    monkeypatch.setattr("soridormi_runtime.residual_policy.PolicyProfile.load", lambda _: teacher_profile)
    monkeypatch.setattr("soridormi_runtime.policy_factory.make_runtime_policy", fake_make_runtime_policy)

    policy = ResidualOnnxPolicy(
        policy_path=residual_path,
        session_factory=FakeSession,
        providers=["CPUExecutionProvider"],
    )
    action = policy.compute_action(_state())

    assert action.tolist() == pytest.approx([0.225] * 14)
    assert created == [{"policy_path": str(tmp_path / "inner_residual.onnx"), "backend": "residual_onnx"}]


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


def test_residual_cli_dry_run_can_exclude_zero_candidate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_residual_policy",
            "teacher_profile",
            "--output-dir",
            str(tmp_path),
            "--actor-kind",
            RESIDUAL_ACTOR_CONTACT_PHASE_LIFT,
            "--dry-run",
            "--no-zero-candidate",
            "--episodic-clearance-quantile",
            "0.20",
            "--episodic-clearance-quantile-gap-weight",
            "3.5",
        ],
    )

    train_residual_policy_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["actor_kind"] == RESIDUAL_ACTOR_CONTACT_PHASE_LIFT
    assert payload["optimization_config"]["include_zero_candidate"] is False
    assert payload["episodic_clearance_quantile"] == pytest.approx(0.20)
    assert payload["episodic_clearance_quantile_gap_weight"] == pytest.approx(3.5)


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


def test_contact_phase_lift_actor_uses_swing_contact_and_phase() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [0.0, 1.0, 0.5, -0.5]
    parameters = np.zeros(CONTACT_PHASE_LIFT_PARAMETER_SIZE, dtype=np.float32).reshape(2, 3, 3)
    parameters[0, 0, 0] = 0.4
    parameters[0, 1, 1] = 0.2
    parameters[0, 2, 2] = -0.6

    left_features, right_features = contact_phase_lift_residual_features(observation)
    action = contact_phase_lift_residual_action(observation, parameters.reshape(-1))

    assert left_features.tolist() == pytest.approx([1.0, 0.5, -0.5])
    assert right_features.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert action[2] == pytest.approx(np.tanh(0.4))
    assert action[3] == pytest.approx(np.tanh(0.1))
    assert action[4] == pytest.approx(np.tanh(0.3))
    assert action[11:14].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert action[[0, 1, 5, 6, 7, 8, 9, 10]].tolist() == pytest.approx([0.0] * 8)


def test_contact_phase_lift_optimizer_uses_compact_parameter_vector() -> None:
    result = optimize_contact_phase_lift_residual(
        lambda candidate: -float(np.mean(candidate**2)),
        config=ResidualOptimizationConfig(iterations=1, population=3, seed=11),
    )

    assert len(result.best_residual) == CONTACT_PHASE_LIFT_PARAMETER_SIZE


def test_command_contact_phase_lift_actor_uses_command_tail() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [0.0, 1.0, 0.5, -0.5]
    observation[101:104] = [0.2, -0.1, 0.3]
    parameters = np.zeros(COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE, dtype=np.float32).reshape(
        2,
        COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE,
        3,
    )
    parameters[0, 3, 0] = 2.0
    parameters[0, 10, 1] = 4.0
    parameters[0, 11, 2] = -2.0

    left_features, right_features = command_contact_phase_lift_residual_features(observation)
    action = command_contact_phase_lift_residual_action(observation, parameters.reshape(-1))

    assert left_features.tolist() == pytest.approx(
        [
            1.0,
            0.5,
            -0.5,
            0.2,
            0.1,
            -0.1,
            -0.1,
            -0.05,
            0.05,
            0.3,
            0.15,
            -0.15,
        ]
    )
    assert right_features.tolist() == pytest.approx([0.0] * COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE)
    assert action[2] == pytest.approx(np.tanh(0.4))
    assert action[3] == pytest.approx(np.tanh(0.6))
    assert action[4] == pytest.approx(np.tanh(0.3))
    assert action[11:14].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert action[[0, 1, 5, 6, 7, 8, 9, 10]].tolist() == pytest.approx([0.0] * 8)


def test_command_contact_phase_lift_actor_requires_policy_command_tail() -> None:
    with pytest.raises(ValueError, match="104D policy input"):
        command_contact_phase_lift_residual_features(np.zeros(101, dtype=np.float32))


def test_command_contact_phase_lift_optimizer_uses_compact_parameter_vector() -> None:
    result = optimize_command_contact_phase_lift_residual(
        lambda candidate: -float(np.mean(candidate**2)),
        config=ResidualOptimizationConfig(iterations=1, population=3, seed=13),
    )

    assert len(result.best_residual) == COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE


def test_residual_source_from_command_contact_phase_lift_parameters() -> None:
    parameters = np.zeros(COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE, dtype=np.float32).reshape(
        2,
        COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE,
        3,
    )
    parameters[0, 3, 0] = 2.0
    source = _residual_source_from_parameters(
        RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT,
        parameters.reshape(-1),
    )
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [0.0, 1.0, 0.5, -0.5]
    observation[101:104] = [0.2, 0.0, 0.0]

    action = source(observation)

    assert action[2] == pytest.approx(np.tanh(0.4))


def test_contact_phase_harmonic_lift_actor_uses_phase_harmonics() -> None:
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [0.0, 1.0, 0.5, -0.5]
    parameters = np.zeros(CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE, dtype=np.float32).reshape(
        2,
        CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE,
        3,
    )
    parameters[0, 3, 0] = 2.0
    parameters[0, 4, 1] = -1.0
    parameters[0, 6, 2] = 0.5

    left_features, right_features = contact_phase_harmonic_lift_residual_features(observation)
    action = contact_phase_harmonic_lift_residual_action(observation, parameters.reshape(-1))

    assert left_features.tolist() == pytest.approx([1.0, 0.5, -0.5, 0.0, -0.5, -0.25, -0.25])
    assert right_features.tolist() == pytest.approx([0.0] * CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE)
    assert action[2] == pytest.approx(np.tanh(0.0))
    assert action[3] == pytest.approx(np.tanh(0.5))
    assert action[4] == pytest.approx(np.tanh(-0.125))
    assert action[11:14].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_contact_phase_harmonic_lift_optimizer_uses_compact_parameter_vector() -> None:
    result = optimize_contact_phase_harmonic_lift_residual(
        lambda candidate: -float(np.mean(candidate**2)),
        config=ResidualOptimizationConfig(iterations=1, population=3, seed=15),
    )

    assert len(result.best_residual) == CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE


def test_residual_source_from_contact_phase_harmonic_lift_parameters() -> None:
    parameters = np.zeros(CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE, dtype=np.float32).reshape(
        2,
        CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE,
        3,
    )
    parameters[0, 4, 0] = -1.0
    source = _residual_source_from_parameters(
        RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT,
        parameters.reshape(-1),
    )
    observation = np.zeros(104, dtype=np.float32)
    observation[97:101] = [0.0, 1.0, 0.5, -0.5]

    action = source(observation)

    assert action[2] == pytest.approx(np.tanh(0.5))


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




def test_episodic_clearance_gap_weight_distinguishes_shortfall_size() -> None:
    shallow = episodic_clearance_adjustment(
        [0.013, 0.014],
        target_clearance=0.015,
        clearance_weight=0.0,
        low_clearance_penalty_weight=0.0,
        clearance_gap_weight=1.0,
    )
    deep = episodic_clearance_adjustment(
        [0.006, 0.009],
        target_clearance=0.015,
        clearance_weight=0.0,
        low_clearance_penalty_weight=0.0,
        clearance_gap_weight=1.0,
    )
    passing = episodic_clearance_adjustment(
        [0.015, 0.020],
        target_clearance=0.015,
        clearance_weight=0.0,
        low_clearance_penalty_weight=0.0,
        clearance_gap_weight=1.0,
    )

    assert shallow > deep
    assert passing == pytest.approx(0.0)


def test_episodic_clearance_quantile_gap_weight_targets_lower_tail() -> None:
    tail_low = episodic_clearance_adjustment(
        [0.006, 0.015, 0.020, 0.020],
        target_clearance=0.015,
        clearance_weight=0.0,
        low_clearance_penalty_weight=0.0,
        clearance_quantile=0.25,
        clearance_quantile_gap_weight=2.0,
    )
    tail_passing = episodic_clearance_adjustment(
        [0.015, 0.015, 0.020, 0.020],
        target_clearance=0.015,
        clearance_weight=0.0,
        low_clearance_penalty_weight=0.0,
        clearance_quantile=0.25,
        clearance_quantile_gap_weight=2.0,
    )

    assert tail_passing > tail_low
    assert tail_passing == pytest.approx(0.0)


def test_low_clearance_regression_adjustment_penalizes_reference_regression() -> None:
    penalty = low_clearance_regression_adjustment(
        [0.010, 0.012, 0.018, 0.020],
        target_clearance=0.015,
        reference_low_clearance_ratio=0.25,
        penalty_weight=10.0,
    )
    no_regression = low_clearance_regression_adjustment(
        [0.010, 0.018, 0.020, 0.022],
        target_clearance=0.015,
        reference_low_clearance_ratio=0.25,
        penalty_weight=10.0,
    )

    assert penalty == pytest.approx(-2.5)
    assert no_regression == pytest.approx(0.0)


def test_validate_clearance_quantile_rejects_out_of_range_values() -> None:
    assert _validate_clearance_quantile(0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _validate_clearance_quantile(1.1)


def test_normalize_reference_low_clearance_ratios_matches_objective_count() -> None:
    assert _normalize_reference_low_clearance_ratios(
        [0.26, 0.31],
        objective_count=2,
    ) == [pytest.approx(0.26), pytest.approx(0.31)]
    assert _normalize_reference_low_clearance_ratios(None, objective_count=2) == [None, None]

    with pytest.raises(ValueError, match="match the number of training objectives"):
        _normalize_reference_low_clearance_ratios([0.26], objective_count=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _normalize_reference_low_clearance_ratios([1.2], objective_count=1)


def test_parse_training_command_builds_velocity_command() -> None:
    command = _parse_training_command("0.09,0.0,0.12")

    assert command.x_velocity == pytest.approx(0.09)
    assert command.y_velocity == pytest.approx(0.0)
    assert command.yaw_velocity == pytest.approx(0.12)


def test_parse_training_command_spec_accepts_optional_weight() -> None:
    spec = _parse_training_command_spec("0.08,0.0,0.20,3.5")

    assert spec.command is not None
    assert spec.command.x_velocity == pytest.approx(0.08)
    assert spec.command.yaw_velocity == pytest.approx(0.20)
    assert spec.weight == pytest.approx(3.5)
    assert spec.describe()["weight"] == pytest.approx(3.5)


def test_normalize_training_commands_preserves_legacy_unweighted_commands() -> None:
    command = PolicyCommand(x_velocity=0.1, yaw_velocity=0.2)
    weighted = ResidualTrainingCommand(command=PolicyCommand(x_velocity=0.05), weight=2.0)

    specs = _normalize_training_commands([command, weighted])

    assert [item.weight for item in specs] == pytest.approx([1.0, 2.0])
    assert specs[0].command is command
    assert specs[1] is weighted


def test_normalize_training_commands_rejects_nonpositive_weight() -> None:
    with pytest.raises(ValueError, match="positive"):
        _normalize_training_commands([ResidualTrainingCommand(command=PolicyCommand(), weight=0.0)])




def test_parse_training_sequence_accepts_weighted_segments() -> None:
    sequence = _parse_training_sequence("2.5|0,0,0,50;0.06,0,0,100;0,0,0,25")

    assert sequence.weight == pytest.approx(2.5)
    assert len(sequence.segments) == 3
    assert sequence.segments[1].command.x_velocity == pytest.approx(0.06)
    assert sequence.segments[1].steps == 100
    assert sequence.describe()["total_steps"] == 175


def test_parse_training_sequence_rejects_fractional_steps() -> None:
    with pytest.raises(Exception, match="positive integer"):
        _parse_training_sequence("0,0,0,1.5")


def test_normalize_training_sequences_rejects_bad_weight_and_steps() -> None:
    segment = ResidualTrainingSegment(command=PolicyCommand(), steps=5)

    with pytest.raises(ValueError, match="positive"):
        _normalize_training_sequences([ResidualTrainingSequence(segments=(segment,), weight=0.0)])
    with pytest.raises(ValueError, match="steps"):
        _normalize_training_sequences([ResidualTrainingSequence(segments=(ResidualTrainingSegment(PolicyCommand(), 0),))])



def test_score_for_aggregation_can_normalize_by_requested_steps() -> None:
    episode = {"score": 150.0, "requested_steps": 300, "completed_steps": 120}

    assert _score_for_aggregation(episode, score_normalization="total") == pytest.approx(150.0)
    assert _score_for_aggregation(episode, score_normalization="per_step") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="score_normalization"):
        _validate_score_normalization("bad")

def test_aggregate_training_scores_blends_weighted_mean_with_worst_case() -> None:
    scores = [10.0, 2.0]
    weights = [1.0, 3.0]

    assert aggregate_training_scores(scores, weights, worst_case_score_weight=0.0) == pytest.approx(4.0)
    assert aggregate_training_scores(scores, weights, worst_case_score_weight=1.0) == pytest.approx(2.0)
    assert aggregate_training_scores(scores, weights, worst_case_score_weight=0.25) == pytest.approx(3.5)


def test_aggregate_training_scores_rejects_invalid_worst_case_weight() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_training_scores([1.0], [1.0], worst_case_score_weight=1.1)


def test_residual_live_breakdown_records_each_training_objective(monkeypatch) -> None:
    calls = []

    def fake_episode(residual_source, **kwargs):
        calls.append(kwargs)
        score = 10.0 if kwargs["segments"] is None else 2.0
        return _build_episode_score_breakdown(
            score=score,
            reward_sum=score - 0.01,
            requested_steps=10,
            completed_steps=10 if kwargs["segments"] is None else 5,
            terminated=kwargs["segments"] is not None,
            swing_clearances=[0.010, 0.020] if kwargs["segments"] is None else [0.006],
            target_clearance=0.015,
            clearance_adjustment_per_step=0.0,
            clearance_adjustment_total=0.0,
            survival_bonus=0.01,
        )

    monkeypatch.setattr(train_residual_policy_module, "_evaluate_residual_episode_breakdown", fake_episode)
    command = ResidualTrainingCommand(command=PolicyCommand(x_velocity=0.125), weight=1.0)
    sequence = ResidualTrainingSequence(
        segments=(ResidualTrainingSegment(command=PolicyCommand(x_velocity=0.06), steps=5),),
        weight=3.0,
    )

    breakdown = train_residual_policy_module._evaluate_residual_live_breakdown(
        np.zeros(14, dtype=np.float32),
        teacher_profile="teacher",
        steps=10,
        residual_scale=0.05,
        residual_clip_abs=1.0,
        final_action_clip_abs=None,
        reward_config=WalkingRewardConfig(),
        host="127.0.0.1",
        port=5555,
        training_commands=[command],
        training_sequences=[sequence],
        worst_case_score_weight=0.25,
    )

    assert len(calls) == 2
    assert breakdown["weighted_mean_score"] == pytest.approx(4.0)
    assert breakdown["worst_score"] == pytest.approx(2.0)
    assert breakdown["aggregate_score"] == pytest.approx(3.5)
    assert [item["kind"] for item in breakdown["items"]] == ["command", "sequence"]
    assert breakdown["items"][0]["episode"]["median_swing_clearance_m"] == pytest.approx(0.015)
    assert breakdown["items"][1]["episode"]["completed_steps"] == 5
    assert breakdown["items"][1]["episode"]["terminated"] is True
    assert breakdown["items"][1]["sequence"]["total_steps"] == 5


def test_residual_live_breakdown_per_step_normalization_uses_requested_steps(monkeypatch) -> None:
    def fake_episode(residual_source, **kwargs):
        if kwargs["segments"] is None:
            return _build_episode_score_breakdown(
                score=300.0,
                reward_sum=300.0,
                requested_steps=300,
                completed_steps=300,
                terminated=False,
                swing_clearances=[],
                target_clearance=0.015,
                clearance_adjustment_per_step=0.0,
                clearance_adjustment_total=0.0,
                survival_bonus=0.0,
            )
        return _build_episode_score_breakdown(
            score=150.0,
            reward_sum=150.0,
            requested_steps=100,
            completed_steps=100,
            terminated=False,
            swing_clearances=[],
            target_clearance=0.015,
            clearance_adjustment_per_step=0.0,
            clearance_adjustment_total=0.0,
            survival_bonus=0.0,
        )

    monkeypatch.setattr(train_residual_policy_module, "_evaluate_residual_episode_breakdown", fake_episode)
    command = ResidualTrainingCommand(command=PolicyCommand(x_velocity=0.125), weight=1.0)
    sequence = ResidualTrainingSequence(
        segments=(ResidualTrainingSegment(command=PolicyCommand(x_velocity=0.09), steps=100),),
        weight=1.0,
    )

    breakdown = train_residual_policy_module._evaluate_residual_live_breakdown(
        np.zeros(14, dtype=np.float32),
        teacher_profile="teacher",
        steps=300,
        residual_scale=0.05,
        residual_clip_abs=1.0,
        final_action_clip_abs=None,
        reward_config=WalkingRewardConfig(),
        host="127.0.0.1",
        port=5555,
        training_commands=[command],
        training_sequences=[sequence],
        worst_case_score_weight=1.0,
        score_normalization="per_step",
    )

    assert breakdown["score_normalization"] == "per_step"
    assert breakdown["items"][0]["score"] == pytest.approx(300.0)
    assert breakdown["items"][0]["objective_score"] == pytest.approx(1.0)
    assert breakdown["items"][1]["score"] == pytest.approx(150.0)
    assert breakdown["items"][1]["objective_score"] == pytest.approx(1.5)
    assert breakdown["worst_item_index"] == 0
    assert breakdown["aggregate_score"] == pytest.approx(1.0)


def test_build_episode_score_breakdown_records_clearance_gate_metrics() -> None:
    breakdown = _build_episode_score_breakdown(
        score=12.0,
        reward_sum=11.0,
        requested_steps=10,
        completed_steps=8,
        terminated=True,
        swing_clearances=[0.006, 0.012, 0.018],
        target_clearance=0.015,
        clearance_adjustment_per_step=-0.1,
        clearance_adjustment_total=-0.8,
        survival_bonus=0.008,
    )

    assert breakdown["completed_steps"] == 8
    assert breakdown["completion_ratio"] == pytest.approx(0.8)
    assert breakdown["terminated"] is True
    assert breakdown["median_swing_clearance_m"] == pytest.approx(0.012)
    assert breakdown["min_swing_clearance_m"] == pytest.approx(0.006)
    assert breakdown["low_clearance_ratio"] == pytest.approx(2.0 / 3.0)
    assert breakdown["p25_swing_clearance_m"] == pytest.approx(0.009)
    assert breakdown["mean_clearance_gap_ratio"] == pytest.approx(((0.015 - 0.006) / 0.015 + (0.015 - 0.012) / 0.015) / 3.0)
    assert breakdown["median_clearance_gap_ratio"] == pytest.approx((0.015 - 0.012) / 0.015)
    assert breakdown["max_clearance_gap_ratio"] == pytest.approx((0.015 - 0.006) / 0.015)
    assert breakdown["episodic_clearance_adjustment_total"] == pytest.approx(-0.8)
    assert breakdown["episodic_clearance_quantile"] == pytest.approx(0.25)


def test_build_episode_score_breakdown_records_segment_clearance_diagnostics() -> None:
    segment = _build_episode_segment_breakdown(
        index=1,
        command=PolicyCommand(x_velocity=0.09, yaw_velocity=0.12),
        reward_sum=3.0,
        requested_steps=5,
        completed_steps=4,
        terminated=True,
        swing_clearances=[0.004, 0.008],
        target_clearance=0.015,
    )
    breakdown = _build_episode_score_breakdown(
        score=4.0,
        reward_sum=3.0,
        requested_steps=5,
        completed_steps=4,
        terminated=True,
        swing_clearances=[0.004, 0.008],
        target_clearance=0.015,
        clearance_adjustment_per_step=-0.4,
        clearance_adjustment_total=-1.6,
        survival_bonus=0.004,
        segments=[segment],
    )

    assert breakdown["segments"][0]["index"] == 1
    assert breakdown["segments"][0]["completed_steps"] == 4
    assert breakdown["segments"][0]["terminated"] is True
    assert breakdown["segments"][0]["command"]["x_velocity"] == pytest.approx(0.09)
    assert breakdown["segments"][0]["swing_clearance_sample_count"] == 2
    assert breakdown["segments"][0]["median_swing_clearance_m"] == pytest.approx(0.006)
    assert breakdown["segments"][0]["low_clearance_ratio"] == pytest.approx(1.0)
    assert breakdown["segments"][0]["mean_clearance_gap_ratio"] == pytest.approx(((0.015 - 0.004) / 0.015 + (0.015 - 0.008) / 0.015) / 2.0)

def test_residual_source_from_parameters_matches_actor_kind() -> None:
    constant = _residual_source_from_parameters("constant", [0.1] * 14)
    assert isinstance(constant, np.ndarray)
    assert constant.shape == (14,)

    command_state = _residual_source_from_parameters(RESIDUAL_ACTOR_COMMAND_STATE, [0.0] * COMMAND_STATE_PARAMETER_SIZE)
    assert callable(command_state)
    assert command_state(np.zeros(104, dtype=np.float32)).shape == (14,)


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
