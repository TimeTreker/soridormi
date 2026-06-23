from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from soridormi_runtime.create_policy_profile import build_replacement_profile_payload
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.rl_finetune_env import ACTION_SIZE, ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.walking_reward import WalkingRewardConfig


DEFAULT_OUTPUT_ROOT = Path("/data/rl_finetune/residual_policy")
DEFAULT_RESIDUAL_ONNX_NAME = "residual_policy.onnx"
DEFAULT_RESIDUAL_PT_NAME = "residual_policy.pt"
SCORE_NORMALIZATION_TOTAL = "total"
SCORE_NORMALIZATION_PER_STEP = "per_step"
SCORE_NORMALIZATION_CHOICES = (SCORE_NORMALIZATION_TOTAL, SCORE_NORMALIZATION_PER_STEP)
RESIDUAL_ACTOR_CONSTANT = "constant"
RESIDUAL_ACTOR_PHASE_CONTACT = "phase_contact"
RESIDUAL_ACTOR_COMMAND_STATE = "command_state"
RESIDUAL_ACTOR_COMMAND_STATE_MLP = "command_state_mlp"
RESIDUAL_ACTOR_CONTACT_PHASE_LIFT = "contact_phase_lift"
RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT = "contact_phase_harmonic_lift"
RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT = "command_contact_phase_lift"
PHASE_CONTACT_OBSERVATION_START = 97
PHASE_CONTACT_OBSERVATION_STOP = 101
PHASE_CONTACT_FEATURE_SIZE = 5
PHASE_CONTACT_PARAMETER_SIZE = PHASE_CONTACT_FEATURE_SIZE * ACTION_SIZE
COMMAND_STATE_COMMAND_SLICE = slice(6, 9)
COMMAND_STATE_CONTACT_PHASE_SLICE = slice(97, 101)
COMMAND_STATE_LEG_JOINT_OFFSET_INDICES = (15, 16, 17, 24, 25, 26)
COMMAND_STATE_LAST_ACTION_INDICES = (43, 44, 45, 52, 53, 54)
COMMAND_STATE_ACTION_INDICES = (2, 3, 4, 11, 12, 13)
COMMAND_STATE_FEATURE_SIZE = 20
COMMAND_STATE_OUTPUT_SIZE = len(COMMAND_STATE_ACTION_INDICES)
COMMAND_STATE_PARAMETER_SIZE = COMMAND_STATE_FEATURE_SIZE * COMMAND_STATE_OUTPUT_SIZE
COMMAND_STATE_MLP_HIDDEN_SIZE = 4
COMMAND_STATE_MLP_PARAMETER_SIZE = (
    COMMAND_STATE_PARAMETER_SIZE
    + COMMAND_STATE_FEATURE_SIZE * COMMAND_STATE_MLP_HIDDEN_SIZE
    + COMMAND_STATE_MLP_HIDDEN_SIZE
    + COMMAND_STATE_MLP_HIDDEN_SIZE * COMMAND_STATE_OUTPUT_SIZE
)
CONTACT_PHASE_LIFT_FEATURE_SIZE = 3
CONTACT_PHASE_LIFT_ACTION_INDICES = COMMAND_STATE_ACTION_INDICES
CONTACT_PHASE_LIFT_PARAMETER_SIZE = 2 * CONTACT_PHASE_LIFT_FEATURE_SIZE * 3
CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE = 7
CONTACT_PHASE_HARMONIC_LIFT_ACTION_INDICES = COMMAND_STATE_ACTION_INDICES
CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE = 2 * CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE * 3
COMMAND_CONTACT_PHASE_LIFT_COMMAND_SLICE = slice(101, 104)
COMMAND_CONTACT_PHASE_LIFT_INPUT_SIZE = 104
COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE = 12
COMMAND_CONTACT_PHASE_LIFT_ACTION_INDICES = COMMAND_STATE_ACTION_INDICES
COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE = 2 * COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE * 3


@dataclass(frozen=True)
class ResidualOptimizationConfig:
    iterations: int = 5
    population: int = 16
    elite_fraction: float = 0.25
    initial_std: float = 0.25
    min_std: float = 0.01
    std_decay: float = 0.85
    seed: int = 0
    residual_clip_abs: float = 1.0
    include_zero_candidate: bool = True


@dataclass(frozen=True)
class ResidualOptimizationResult:
    best_residual: list[float]
    best_score: float
    final_mean: list[float]
    final_std: list[float]
    iterations: list[dict[str, Any]]


@dataclass(frozen=True)
class ResidualTrainingCommand:
    command: PolicyCommand | None
    weight: float = 1.0

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"weight": float(self.weight)}
        if self.command is not None:
            payload.update(self.command.describe())
        return payload


@dataclass(frozen=True)
class ResidualTrainingSegment:
    command: PolicyCommand
    steps: int

    def describe(self) -> dict[str, Any]:
        payload = self.command.describe()
        payload["steps"] = int(self.steps)
        return payload


@dataclass(frozen=True)
class ResidualTrainingSequence:
    segments: tuple[ResidualTrainingSegment, ...]
    weight: float = 1.0

    def describe(self) -> dict[str, Any]:
        return {
            "weight": float(self.weight),
            "segments": [segment.describe() for segment in self.segments],
            "total_steps": int(sum(segment.steps for segment in self.segments)),
        }


@dataclass(frozen=True)
class ResidualPolicyTrainResult:
    ok: bool
    teacher_profile: str
    output_dir: str
    residual_onnx_path: str | None
    residual_checkpoint_path: str | None
    metrics_path: str
    report_path: str
    profile_name: str | None
    profile_path: str | None
    optimization: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "PyTorch is required for residual policy ONNX export. Build the runtime image with training deps: "
            "./scripts/build_runtime_training.sh"
        ) from exc
    return torch


def optimize_residual_bias(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Cross-entropy optimizer for a safe 14D residual bias policy.

    This is intentionally simple and robust: M6.19 starts residual RL with a
    bounded constant residual, then later work can replace the optimizer/model
    with PPO/SAC or a recurrent residual actor without changing the deployment
    contract.
    """
    return _optimize_parameter_vector(evaluate, parameter_size=ACTION_SIZE, config=config)


def optimize_phase_contact_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Optimize a bounded linear actor over bias, foot contacts, and gait phase."""
    return _optimize_parameter_vector(evaluate, parameter_size=PHASE_CONTACT_PARAMETER_SIZE, config=config)


def optimize_command_state_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Optimize a compact actor over command, gait state, and action history."""
    return _optimize_parameter_vector(evaluate, parameter_size=COMMAND_STATE_PARAMETER_SIZE, config=config)


def optimize_command_state_mlp_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
    initial_mean: np.ndarray | list[float] | None = None,
) -> ResidualOptimizationResult:
    return _optimize_parameter_vector(
        evaluate,
        parameter_size=COMMAND_STATE_MLP_PARAMETER_SIZE,
        config=config,
        initial_mean=initial_mean,
    )


def optimize_contact_phase_lift_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Optimize a compact swing-leg lift actor from contacts and gait phase."""
    return _optimize_parameter_vector(evaluate, parameter_size=CONTACT_PHASE_LIFT_PARAMETER_SIZE, config=config)


def optimize_command_contact_phase_lift_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Optimize swing-leg lift conditioned on desired command, contacts, and gait phase."""
    return _optimize_parameter_vector(
        evaluate,
        parameter_size=COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE,
        config=config,
    )


def optimize_contact_phase_harmonic_lift_residual(
    evaluate: Callable[[np.ndarray], float],
    *,
    config: ResidualOptimizationConfig | None = None,
) -> ResidualOptimizationResult:
    """Optimize a swing-leg lift actor with phase harmonics."""
    return _optimize_parameter_vector(
        evaluate,
        parameter_size=CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE,
        config=config,
    )


def _optimize_parameter_vector(
    evaluate: Callable[[np.ndarray], float],
    *,
    parameter_size: int,
    config: ResidualOptimizationConfig | None,
    initial_mean: np.ndarray | list[float] | None = None,
) -> ResidualOptimizationResult:
    cfg = config or ResidualOptimizationConfig()
    rng = np.random.default_rng(int(cfg.seed))
    if initial_mean is None:
        mean = np.zeros(parameter_size, dtype=np.float32)
    else:
        mean = np.asarray(initial_mean, dtype=np.float32).reshape(parameter_size).copy()
    std = np.full(parameter_size, float(cfg.initial_std), dtype=np.float32)
    elite_count = max(1, int(math.ceil(float(cfg.population) * float(cfg.elite_fraction))))
    best_residual = mean.copy()
    best_score = float("-inf")
    history: list[dict[str, Any]] = []

    for iteration in range(max(1, int(cfg.iterations))):
        candidates = rng.normal(mean, std, size=(max(1, int(cfg.population)), parameter_size)).astype(np.float32)
        candidates = np.clip(candidates, -float(cfg.residual_clip_abs), float(cfg.residual_clip_abs))
        if cfg.include_zero_candidate:
            candidates[0, :] = mean
        scores = np.asarray([float(evaluate(candidate)) for candidate in candidates], dtype=np.float64)
        order = np.argsort(scores)[::-1]
        elites = candidates[order[:elite_count]]
        elite_scores = scores[order[:elite_count]]
        if float(scores[order[0]]) > best_score:
            best_score = float(scores[order[0]])
            best_residual = candidates[order[0]].copy()
        mean = elites.mean(axis=0).astype(np.float32)
        std = np.maximum(elites.std(axis=0).astype(np.float32), float(cfg.min_std))
        std = np.maximum(std * float(cfg.std_decay), float(cfg.min_std)).astype(np.float32)
        history.append(
            {
                "iteration": iteration,
                "best_score": float(scores[order[0]]),
                "mean_score": float(scores.mean()),
                "elite_mean_score": float(elite_scores.mean()),
                "best_residual_abs_max": float(np.max(np.abs(candidates[order[0]]))),
                "distribution_std_mean": float(std.mean()),
            }
        )

    return ResidualOptimizationResult(
        best_residual=[float(x) for x in best_residual.tolist()],
        best_score=float(best_score),
        final_mean=[float(x) for x in mean.tolist()],
        final_std=[float(x) for x in std.tolist()],
        iterations=history,
    )


def phase_contact_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    """Compute a 14D residual from canonical contact and imitation-phase fields."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            "phase_contact actor requires an observation with at least "
            f"{PHASE_CONTACT_OBSERVATION_STOP} values, got {obs.size}"
        )
    weights = np.asarray(parameters, dtype=np.float32).reshape(PHASE_CONTACT_FEATURE_SIZE, ACTION_SIZE)
    features = np.concatenate(
        (
            np.ones(1, dtype=np.float32),
            obs[PHASE_CONTACT_OBSERVATION_START:PHASE_CONTACT_OBSERVATION_STOP],
        )
    )
    return np.tanh(features @ weights).astype(np.float32)


def command_state_residual_features(observation: np.ndarray | list[float]) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            "command_state actor requires an observation with at least "
            f"{PHASE_CONTACT_OBSERVATION_STOP} values, got {obs.size}"
        )
    return np.concatenate(
        (
            np.ones(1, dtype=np.float32),
            obs[COMMAND_STATE_COMMAND_SLICE],
            obs[COMMAND_STATE_CONTACT_PHASE_SLICE],
            obs[list(COMMAND_STATE_LEG_JOINT_OFFSET_INDICES)],
            obs[list(COMMAND_STATE_LAST_ACTION_INDICES)],
        )
    ).astype(np.float32)


def command_state_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    weights = np.asarray(parameters, dtype=np.float32).reshape(
        COMMAND_STATE_FEATURE_SIZE,
        COMMAND_STATE_OUTPUT_SIZE,
    )
    leg_action = np.tanh(command_state_residual_features(observation) @ weights).astype(np.float32)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[list(COMMAND_STATE_ACTION_INDICES)] = leg_action
    return action


def command_state_mlp_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    features = command_state_residual_features(observation)
    values = np.asarray(parameters, dtype=np.float32).reshape(COMMAND_STATE_MLP_PARAMETER_SIZE)
    cursor = 0
    linear_size = COMMAND_STATE_PARAMETER_SIZE
    linear = values[cursor : cursor + linear_size].reshape(
        COMMAND_STATE_FEATURE_SIZE,
        COMMAND_STATE_OUTPUT_SIZE,
    )
    cursor += linear_size
    input_hidden_size = COMMAND_STATE_FEATURE_SIZE * COMMAND_STATE_MLP_HIDDEN_SIZE
    input_hidden = values[cursor : cursor + input_hidden_size].reshape(
        COMMAND_STATE_FEATURE_SIZE,
        COMMAND_STATE_MLP_HIDDEN_SIZE,
    )
    cursor += input_hidden_size
    hidden_bias = values[cursor : cursor + COMMAND_STATE_MLP_HIDDEN_SIZE]
    cursor += COMMAND_STATE_MLP_HIDDEN_SIZE
    hidden_output = values[cursor:].reshape(
        COMMAND_STATE_MLP_HIDDEN_SIZE,
        COMMAND_STATE_OUTPUT_SIZE,
    )
    hidden = np.tanh(features @ input_hidden + hidden_bias)
    leg_action = np.tanh(features @ linear + hidden @ hidden_output).astype(np.float32)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[list(COMMAND_STATE_ACTION_INDICES)] = leg_action
    return action


def contact_phase_lift_residual_features(observation: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            "contact_phase_lift actor requires an observation with at least "
            f"{PHASE_CONTACT_OBSERVATION_STOP} values, got {obs.size}"
        )
    left_contact, right_contact, phase_cos, phase_sin = obs[COMMAND_STATE_CONTACT_PHASE_SLICE]
    left_swing = float(np.clip(1.0 - float(left_contact), 0.0, 1.0))
    right_swing = float(np.clip(1.0 - float(right_contact), 0.0, 1.0))
    left_features = np.asarray(
        [left_swing, left_swing * float(phase_cos), left_swing * float(phase_sin)],
        dtype=np.float32,
    )
    right_features = np.asarray(
        [right_swing, right_swing * float(phase_cos), right_swing * float(phase_sin)],
        dtype=np.float32,
    )
    return left_features, right_features


def contact_phase_lift_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float32).reshape(2, CONTACT_PHASE_LIFT_FEATURE_SIZE, 3)
    left_features, right_features = contact_phase_lift_residual_features(observation)
    left_action = np.tanh(left_features @ values[0]).astype(np.float32)
    right_action = np.tanh(right_features @ values[1]).astype(np.float32)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[2:5] = left_action
    action[11:14] = right_action
    return action


def _command_contact_phase_lift_leg_features(
    swing: float,
    phase_cos: float,
    phase_sin: float,
    desired_vx: float,
    desired_vy: float,
    desired_yaw: float,
) -> np.ndarray:
    return np.asarray(
        [
            swing,
            swing * phase_cos,
            swing * phase_sin,
            swing * desired_vx,
            swing * desired_vx * phase_cos,
            swing * desired_vx * phase_sin,
            swing * desired_vy,
            swing * desired_vy * phase_cos,
            swing * desired_vy * phase_sin,
            swing * desired_yaw,
            swing * desired_yaw * phase_cos,
            swing * desired_yaw * phase_sin,
        ],
        dtype=np.float32,
    )


def command_contact_phase_lift_residual_features(
    observation: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < COMMAND_CONTACT_PHASE_LIFT_INPUT_SIZE:
        raise ValueError(
            "command_contact_phase_lift actor requires a 104D policy input with appended command, "
            f"got {obs.size} values"
        )
    left_contact, right_contact, phase_cos, phase_sin = obs[COMMAND_STATE_CONTACT_PHASE_SLICE]
    desired_vx, desired_vy, desired_yaw = obs[COMMAND_CONTACT_PHASE_LIFT_COMMAND_SLICE]
    left_swing = float(np.clip(1.0 - float(left_contact), 0.0, 1.0))
    right_swing = float(np.clip(1.0 - float(right_contact), 0.0, 1.0))
    left_features = _command_contact_phase_lift_leg_features(
        left_swing,
        float(phase_cos),
        float(phase_sin),
        float(desired_vx),
        float(desired_vy),
        float(desired_yaw),
    )
    right_features = _command_contact_phase_lift_leg_features(
        right_swing,
        float(phase_cos),
        float(phase_sin),
        float(desired_vx),
        float(desired_vy),
        float(desired_yaw),
    )
    return left_features, right_features


def command_contact_phase_lift_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float32).reshape(2, COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE, 3)
    left_features, right_features = command_contact_phase_lift_residual_features(observation)
    left_action = np.tanh(left_features @ values[0]).astype(np.float32)
    right_action = np.tanh(right_features @ values[1]).astype(np.float32)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[2:5] = left_action
    action[11:14] = right_action
    return action


def contact_phase_harmonic_lift_residual_features(
    observation: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            "contact_phase_harmonic_lift actor requires an observation with at least "
            f"{PHASE_CONTACT_OBSERVATION_STOP} values, got {obs.size}"
        )
    left_contact, right_contact, phase_cos, phase_sin = obs[COMMAND_STATE_CONTACT_PHASE_SLICE]
    cos1 = float(phase_cos)
    sin1 = float(phase_sin)
    cos2 = cos1 * cos1 - sin1 * sin1
    sin2 = 2.0 * cos1 * sin1
    cos3 = cos1 * cos2 - sin1 * sin2
    sin3 = sin1 * cos2 + cos1 * sin2
    left_swing = float(np.clip(1.0 - float(left_contact), 0.0, 1.0))
    right_swing = float(np.clip(1.0 - float(right_contact), 0.0, 1.0))

    def leg_features(swing: float) -> np.ndarray:
        return np.asarray(
            [
                swing,
                swing * cos1,
                swing * sin1,
                swing * cos2,
                swing * sin2,
                swing * cos3,
                swing * sin3,
            ],
            dtype=np.float32,
        )

    return leg_features(left_swing), leg_features(right_swing)


def contact_phase_harmonic_lift_residual_action(
    observation: np.ndarray | list[float],
    parameters: np.ndarray | list[float],
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float32).reshape(2, CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE, 3)
    left_features, right_features = contact_phase_harmonic_lift_residual_features(observation)
    left_action = np.tanh(left_features @ values[0]).astype(np.float32)
    right_action = np.tanh(right_features @ values[1]).astype(np.float32)
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[2:5] = left_action
    action[11:14] = right_action
    return action


def evaluate_residual_bias_live(
    residual: np.ndarray,
    *,
    teacher_profile: str,
    steps: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    host: str,
    port: int,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None = None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
) -> float:
    return _evaluate_residual_live(
        residual,
        teacher_profile=teacher_profile,
        steps=steps,
        residual_scale=residual_scale,
        residual_clip_abs=residual_clip_abs,
        final_action_clip_abs=final_action_clip_abs,
        reward_config=reward_config,
        host=host,
        port=port,
        training_commands=training_commands,
        training_sequences=training_sequences,
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
        episodic_clearance_gap_weight=episodic_clearance_gap_weight,
        episodic_clearance_quantile=episodic_clearance_quantile,
        episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
        reference_low_clearance_ratios=reference_low_clearance_ratios,
        low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        worst_case_score_weight=worst_case_score_weight,
        score_normalization=score_normalization,
    )


def evaluate_phase_contact_residual_live(
    parameters: np.ndarray,
    *,
    teacher_profile: str,
    steps: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    host: str,
    port: int,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None = None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
) -> float:
    actor = lambda observation: phase_contact_residual_action(observation, parameters)
    return _evaluate_residual_live(
        actor,
        teacher_profile=teacher_profile,
        steps=steps,
        residual_scale=residual_scale,
        residual_clip_abs=residual_clip_abs,
        final_action_clip_abs=final_action_clip_abs,
        reward_config=reward_config,
        host=host,
        port=port,
        training_commands=training_commands,
        training_sequences=training_sequences,
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
        episodic_clearance_gap_weight=episodic_clearance_gap_weight,
        episodic_clearance_quantile=episodic_clearance_quantile,
        episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
        reference_low_clearance_ratios=reference_low_clearance_ratios,
        low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        worst_case_score_weight=worst_case_score_weight,
        score_normalization=score_normalization,
    )


def evaluate_command_state_residual_live(
    parameters: np.ndarray,
    *,
    teacher_profile: str,
    steps: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    host: str,
    port: int,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None = None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
) -> float:
    actor = lambda observation: command_state_residual_action(observation, parameters)
    return _evaluate_residual_live(
        actor,
        teacher_profile=teacher_profile,
        steps=steps,
        residual_scale=residual_scale,
        residual_clip_abs=residual_clip_abs,
        final_action_clip_abs=final_action_clip_abs,
        reward_config=reward_config,
        host=host,
        port=port,
        training_commands=training_commands,
        training_sequences=training_sequences,
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
        episodic_clearance_gap_weight=episodic_clearance_gap_weight,
        episodic_clearance_quantile=episodic_clearance_quantile,
        episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
        reference_low_clearance_ratios=reference_low_clearance_ratios,
        low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        worst_case_score_weight=worst_case_score_weight,
        score_normalization=score_normalization,
    )


def evaluate_command_state_mlp_residual_live(
    parameters: np.ndarray,
    **kwargs: Any,
) -> float:
    actor = lambda observation: command_state_mlp_residual_action(observation, parameters)
    return _evaluate_residual_live(actor, **kwargs)


def evaluate_contact_phase_lift_residual_live(
    parameters: np.ndarray,
    **kwargs: Any,
) -> float:
    actor = lambda observation: contact_phase_lift_residual_action(observation, parameters)
    return _evaluate_residual_live(actor, **kwargs)


def evaluate_command_contact_phase_lift_residual_live(
    parameters: np.ndarray,
    **kwargs: Any,
) -> float:
    actor = lambda observation: command_contact_phase_lift_residual_action(observation, parameters)
    return _evaluate_residual_live(actor, **kwargs)


def evaluate_contact_phase_harmonic_lift_residual_live(
    parameters: np.ndarray,
    **kwargs: Any,
) -> float:
    actor = lambda observation: contact_phase_harmonic_lift_residual_action(observation, parameters)
    return _evaluate_residual_live(actor, **kwargs)


def _evaluate_residual_live(
    residual_source: np.ndarray | Callable[[np.ndarray], np.ndarray],
    *,
    teacher_profile: str,
    steps: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    host: str,
    port: int,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
) -> float:
    return float(
        _evaluate_residual_live_breakdown(
            residual_source,
            teacher_profile=teacher_profile,
            steps=steps,
            residual_scale=residual_scale,
            residual_clip_abs=residual_clip_abs,
            final_action_clip_abs=final_action_clip_abs,
            reward_config=reward_config,
            host=host,
            port=port,
            training_commands=training_commands,
            training_sequences=training_sequences,
            episodic_clearance_weight=episodic_clearance_weight,
            episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
            episodic_clearance_gap_weight=episodic_clearance_gap_weight,
            episodic_clearance_quantile=episodic_clearance_quantile,
            episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
            reference_low_clearance_ratios=reference_low_clearance_ratios,
            low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
            worst_case_score_weight=worst_case_score_weight,
            score_normalization=score_normalization,
        )["aggregate_score"]
    )


def _evaluate_residual_live_breakdown(
    residual_source: np.ndarray | Callable[[np.ndarray], np.ndarray],
    *,
    teacher_profile: str,
    steps: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    host: str,
    port: int,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
) -> dict[str, Any]:
    score_normalization = _validate_score_normalization(score_normalization)
    episodic_clearance_quantile = _validate_clearance_quantile(episodic_clearance_quantile)
    if (
        not math.isfinite(float(low_clearance_regression_penalty_weight))
        or float(low_clearance_regression_penalty_weight) < 0.0
    ):
        raise ValueError(
            "low_clearance_regression_penalty_weight must be non-negative and finite, "
            f"got {low_clearance_regression_penalty_weight!r}"
        )
    if not math.isfinite(float(worst_case_score_weight)) or not 0.0 <= float(worst_case_score_weight) <= 1.0:
        raise ValueError(
            "worst_case_score_weight must be finite and in [0, 1], "
            f"got {worst_case_score_weight!r}"
        )

    score_items: list[dict[str, Any]] = []
    scores: list[float] = []
    weights: list[float] = []
    commands = _normalize_training_commands(training_commands)
    sequences = _normalize_training_sequences(training_sequences)

    if not commands and not sequences:
        commands = [ResidualTrainingCommand(command=None, weight=1.0)]
    reference_ratios = _normalize_reference_low_clearance_ratios(
        reference_low_clearance_ratios,
        objective_count=len(commands) + len(sequences),
    )

    for command_index, command_spec in enumerate(commands):
        objective_index = len(scores)
        episode = _evaluate_residual_episode_breakdown(
            residual_source,
            teacher_profile=teacher_profile,
            host=host,
            port=port,
            residual_scale=residual_scale,
            residual_clip_abs=residual_clip_abs,
            final_action_clip_abs=final_action_clip_abs,
            reward_config=reward_config,
            initial_command=command_spec.command,
            segments=None,
            steps=steps,
            episodic_clearance_weight=episodic_clearance_weight,
            episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
            episodic_clearance_gap_weight=episodic_clearance_gap_weight,
            episodic_clearance_quantile=episodic_clearance_quantile,
            episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
            reference_low_clearance_ratio=reference_ratios[objective_index],
            low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        )
        score = float(episode["score"])
        objective_score = _score_for_aggregation(episode, score_normalization=score_normalization)
        scores.append(objective_score)
        weights.append(float(command_spec.weight))
        score_items.append(
            {
                "kind": "command",
                "index": command_index,
                "weight": float(command_spec.weight),
                "score": float(score),
                "objective_score": float(objective_score),
                "score_normalization": score_normalization,
                "command": None if command_spec.command is None else command_spec.command.describe(),
                "episode": episode,
            }
        )

    for sequence_index, sequence in enumerate(sequences):
        objective_index = len(scores)
        episode = _evaluate_residual_episode_breakdown(
            residual_source,
            teacher_profile=teacher_profile,
            host=host,
            port=port,
            residual_scale=residual_scale,
            residual_clip_abs=residual_clip_abs,
            final_action_clip_abs=final_action_clip_abs,
            reward_config=reward_config,
            initial_command=sequence.segments[0].command,
            segments=sequence.segments,
            steps=steps,
            episodic_clearance_weight=episodic_clearance_weight,
            episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
            episodic_clearance_gap_weight=episodic_clearance_gap_weight,
            episodic_clearance_quantile=episodic_clearance_quantile,
            episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
            reference_low_clearance_ratio=reference_ratios[objective_index],
            low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        )
        score = float(episode["score"])
        objective_score = _score_for_aggregation(episode, score_normalization=score_normalization)
        scores.append(objective_score)
        weights.append(float(sequence.weight))
        score_items.append(
            {
                "kind": "sequence",
                "index": sequence_index,
                "weight": float(sequence.weight),
                "score": float(score),
                "objective_score": float(objective_score),
                "score_normalization": score_normalization,
                "sequence": sequence.describe(),
                "episode": episode,
            }
        )

    objective_values = np.asarray(scores, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    raw_values = np.asarray([float(item["score"]) for item in score_items], dtype=np.float64)
    weighted_mean = float(np.average(objective_values, weights=weight_values))
    raw_weighted_mean = float(np.average(raw_values, weights=weight_values))
    worst_index = int(np.argmin(objective_values))
    worst_score = float(scores[worst_index])
    aggregate = aggregate_training_scores(scores, weights, worst_case_score_weight=worst_case_score_weight)
    return {
        "aggregate_score": float(aggregate),
        "weighted_mean_score": weighted_mean,
        "worst_score": worst_score,
        "raw_weighted_mean_score": raw_weighted_mean,
        "score_normalization": score_normalization,
        "worst_case_score_weight": float(worst_case_score_weight),
        "worst_item_index": worst_index,
        "items": score_items,
    }


def _evaluate_residual_episode(
    residual_source: np.ndarray | Callable[[np.ndarray], np.ndarray],
    *,
    teacher_profile: str,
    host: str,
    port: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    initial_command: PolicyCommand | None,
    segments: Sequence[ResidualTrainingSegment] | None,
    steps: int,
    episodic_clearance_weight: float,
    episodic_low_clearance_penalty_weight: float,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratio: float | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
) -> float:
    return float(
        _evaluate_residual_episode_breakdown(
            residual_source,
            teacher_profile=teacher_profile,
            host=host,
            port=port,
            residual_scale=residual_scale,
            residual_clip_abs=residual_clip_abs,
            final_action_clip_abs=final_action_clip_abs,
            reward_config=reward_config,
            initial_command=initial_command,
            segments=segments,
            steps=steps,
            episodic_clearance_weight=episodic_clearance_weight,
            episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
            episodic_clearance_gap_weight=episodic_clearance_gap_weight,
            episodic_clearance_quantile=episodic_clearance_quantile,
            episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
            reference_low_clearance_ratio=reference_low_clearance_ratio,
            low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        )["score"]
    )


def _evaluate_residual_episode_breakdown(
    residual_source: np.ndarray | Callable[[np.ndarray], np.ndarray],
    *,
    teacher_profile: str,
    host: str,
    port: int,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    initial_command: PolicyCommand | None,
    segments: Sequence[ResidualTrainingSegment] | None,
    steps: int,
    episodic_clearance_weight: float,
    episodic_low_clearance_penalty_weight: float,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratio: float | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
) -> dict[str, Any]:
    episodic_clearance_quantile = _validate_clearance_quantile(episodic_clearance_quantile)
    if reference_low_clearance_ratio is not None:
        reference_low_clearance_ratio = _validate_low_clearance_ratio(
            reference_low_clearance_ratio,
            name="reference_low_clearance_ratio",
        )
    if (
        not math.isfinite(float(low_clearance_regression_penalty_weight))
        or float(low_clearance_regression_penalty_weight) < 0.0
    ):
        raise ValueError(
            "low_clearance_regression_penalty_weight must be non-negative and finite, "
            f"got {low_clearance_regression_penalty_weight!r}"
        )
    env = RlFineTuneEnv(
        profile=teacher_profile,
        host=host,
        port=port,
        command=initial_command,
        residual_config=ResidualActionConfig(
            residual_scale=residual_scale,
            residual_clip_abs=residual_clip_abs,
            final_action_clip_abs=final_action_clip_abs,
        ),
        reward_config=reward_config,
        reset_on_start=True,
    )
    reward_sum = 0.0
    completed = 0
    swing_clearances: list[float] = []
    segment_breakdowns: list[dict[str, Any]] = []
    env.reset()

    if segments is None:
        step_commands: list[tuple[PolicyCommand | None, int]] = [(initial_command, max(1, int(steps)))]
    else:
        step_commands = [(segment.command, max(1, int(segment.steps))) for segment in segments]
    requested_steps = int(sum(segment_steps for _, segment_steps in step_commands))

    terminated = False
    for segment_index, (command, segment_steps) in enumerate(step_commands):
        if command is not None:
            env.command = command
        segment_reward_sum = 0.0
        segment_completed = 0
        segment_clearances: list[float] = []
        segment_terminated = False
        for _ in range(segment_steps):
            step = env.step(residual_source)
            step_reward = float(step.metrics.get("reward", 0.0))
            reward_sum += step_reward
            segment_reward_sum += step_reward
            completed += 1
            segment_completed += 1
            diagnostics = step.metrics.get("reward_diagnostics", {})
            clearance = diagnostics.get("swing_clearance_m") if isinstance(diagnostics, dict) else None
            if clearance is not None and math.isfinite(float(clearance)):
                clearance_value = float(clearance)
                swing_clearances.append(clearance_value)
                segment_clearances.append(clearance_value)
            if bool(step.metrics.get("terminated", False)):
                terminated = True
                segment_terminated = True
                break
        segment_breakdowns.append(
            _build_episode_segment_breakdown(
                index=segment_index,
                command=command,
                reward_sum=segment_reward_sum,
                requested_steps=segment_steps,
                completed_steps=segment_completed,
                terminated=segment_terminated,
                swing_clearances=segment_clearances,
                target_clearance=reward_config.target_swing_clearance,
            )
        )
        if terminated:
            break

    clearance_adjustment = episodic_clearance_adjustment(
        swing_clearances,
        target_clearance=reward_config.target_swing_clearance,
        clearance_weight=episodic_clearance_weight,
        low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
        clearance_gap_weight=episodic_clearance_gap_weight,
        clearance_quantile=episodic_clearance_quantile,
        clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
    )
    low_clearance_regression_penalty = low_clearance_regression_adjustment(
        swing_clearances,
        target_clearance=reward_config.target_swing_clearance,
        reference_low_clearance_ratio=reference_low_clearance_ratio,
        penalty_weight=low_clearance_regression_penalty_weight,
    )
    clearance_total = completed * clearance_adjustment
    low_clearance_regression_total = completed * low_clearance_regression_penalty
    survival_bonus = 0.001 * completed
    score = float(reward_sum + clearance_total + low_clearance_regression_total + survival_bonus)
    return _build_episode_score_breakdown(
        score=score,
        reward_sum=reward_sum,
        requested_steps=requested_steps,
        completed_steps=completed,
        terminated=terminated,
        swing_clearances=swing_clearances,
        target_clearance=reward_config.target_swing_clearance,
        clearance_adjustment_per_step=clearance_adjustment,
        clearance_adjustment_total=clearance_total,
        low_clearance_regression_penalty_per_step=low_clearance_regression_penalty,
        low_clearance_regression_penalty_total=low_clearance_regression_total,
        reference_low_clearance_ratio=reference_low_clearance_ratio,
        low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
        clearance_quantile=episodic_clearance_quantile,
        clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
        survival_bonus=survival_bonus,
        segments=segment_breakdowns,
    )


def _build_episode_score_breakdown(
    *,
    score: float,
    reward_sum: float,
    requested_steps: int,
    completed_steps: int,
    terminated: bool,
    swing_clearances: Sequence[float],
    target_clearance: float,
    clearance_adjustment_per_step: float,
    clearance_adjustment_total: float,
    survival_bonus: float,
    low_clearance_regression_penalty_per_step: float = 0.0,
    low_clearance_regression_penalty_total: float = 0.0,
    reference_low_clearance_ratio: float | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    clearance_quantile: float = 0.25,
    clearance_quantile_gap_weight: float = 0.0,
    segments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": float(score),
        "reward_sum": float(reward_sum),
        "requested_steps": int(requested_steps),
        "completed_steps": int(completed_steps),
        "completion_ratio": float(completed_steps / requested_steps) if requested_steps > 0 else 0.0,
        "terminated": bool(terminated),
        "survival_bonus": float(survival_bonus),
        "episodic_clearance_adjustment_per_step": float(clearance_adjustment_per_step),
        "episodic_clearance_adjustment_total": float(clearance_adjustment_total),
        "reference_low_clearance_ratio": None
        if reference_low_clearance_ratio is None
        else float(reference_low_clearance_ratio),
        "low_clearance_regression_penalty_weight": float(low_clearance_regression_penalty_weight),
        "low_clearance_regression_penalty_per_step": float(low_clearance_regression_penalty_per_step),
        "low_clearance_regression_penalty_total": float(low_clearance_regression_penalty_total),
        "episodic_clearance_quantile": float(_validate_clearance_quantile(clearance_quantile)),
        "episodic_clearance_quantile_gap_weight": float(clearance_quantile_gap_weight),
        "target_swing_clearance_m": float(target_clearance),
    }
    payload.update(_build_clearance_diagnostics(swing_clearances, target_clearance=target_clearance))
    payload["segments"] = [dict(segment) for segment in segments] if segments is not None else []
    return payload


def _build_episode_segment_breakdown(
    *,
    index: int,
    command: PolicyCommand | None,
    reward_sum: float,
    requested_steps: int,
    completed_steps: int,
    terminated: bool,
    swing_clearances: Sequence[float],
    target_clearance: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": int(index),
        "command": None if command is None else command.describe(),
        "reward_sum": float(reward_sum),
        "requested_steps": int(requested_steps),
        "completed_steps": int(completed_steps),
        "completion_ratio": float(completed_steps / requested_steps) if requested_steps > 0 else 0.0,
        "terminated": bool(terminated),
        "target_swing_clearance_m": float(target_clearance),
    }
    payload.update(_build_clearance_diagnostics(swing_clearances, target_clearance=target_clearance))
    return payload


def _build_clearance_diagnostics(
    swing_clearances: Sequence[float],
    *,
    target_clearance: float,
) -> dict[str, Any]:
    clearances = np.asarray(list(swing_clearances), dtype=np.float64)
    payload: dict[str, Any] = {
        "swing_clearance_sample_count": int(clearances.size),
        "median_swing_clearance_m": None,
        "mean_swing_clearance_m": None,
        "min_swing_clearance_m": None,
        "max_swing_clearance_m": None,
        "low_clearance_ratio": None,
        "p05_swing_clearance_m": None,
        "p10_swing_clearance_m": None,
        "p25_swing_clearance_m": None,
        "mean_clearance_gap_ratio": None,
        "median_clearance_gap_ratio": None,
        "max_clearance_gap_ratio": None,
    }
    if clearances.size:
        target = max(float(target_clearance), 1e-9)
        gap_ratios = np.maximum(target - clearances, 0.0) / target
        payload.update(
            {
                "median_swing_clearance_m": float(np.median(clearances)),
                "mean_swing_clearance_m": float(np.mean(clearances)),
                "min_swing_clearance_m": float(np.min(clearances)),
                "max_swing_clearance_m": float(np.max(clearances)),
                "low_clearance_ratio": float(np.mean(clearances < target)),
                "p05_swing_clearance_m": float(np.quantile(clearances, 0.05)),
                "p10_swing_clearance_m": float(np.quantile(clearances, 0.10)),
                "p25_swing_clearance_m": float(np.quantile(clearances, 0.25)),
                "mean_clearance_gap_ratio": float(np.mean(gap_ratios)),
                "median_clearance_gap_ratio": float(np.median(gap_ratios)),
                "max_clearance_gap_ratio": float(np.max(gap_ratios)),
            }
        )
    return payload


def episodic_clearance_adjustment(
    swing_clearances: Sequence[float],
    *,
    target_clearance: float,
    clearance_weight: float,
    low_clearance_penalty_weight: float,
    clearance_gap_weight: float = 0.0,
    clearance_quantile: float = 0.25,
    clearance_quantile_gap_weight: float = 0.0,
) -> float:
    if not swing_clearances:
        return 0.0
    target = max(float(target_clearance), 1e-9)
    quantile = _validate_clearance_quantile(clearance_quantile)
    values = np.asarray(swing_clearances, dtype=np.float64)
    median_ratio = float(np.median(values)) / target
    low_ratio = float(np.mean(values < target))
    gap_ratio = float(np.mean(np.maximum(target - values, 0.0) / target))
    quantile_clearance = float(np.quantile(values, quantile))
    quantile_gap_ratio = max(target - quantile_clearance, 0.0) / target
    return (
        float(clearance_weight) * median_ratio
        - float(low_clearance_penalty_weight) * low_ratio
        - float(clearance_gap_weight) * gap_ratio
        - float(clearance_quantile_gap_weight) * quantile_gap_ratio
    )


def low_clearance_regression_adjustment(
    swing_clearances: Sequence[float],
    *,
    target_clearance: float,
    reference_low_clearance_ratio: float | None,
    penalty_weight: float,
) -> float:
    penalty = float(penalty_weight)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError(f"penalty_weight must be non-negative and finite, got {penalty_weight!r}")
    if reference_low_clearance_ratio is None or penalty <= 0.0:
        return 0.0
    reference = _validate_low_clearance_ratio(
        reference_low_clearance_ratio,
        name="reference_low_clearance_ratio",
    )
    if not swing_clearances:
        return 0.0
    target = max(float(target_clearance), 1e-9)
    values = np.asarray(swing_clearances, dtype=np.float64)
    low_ratio = float(np.mean(values < target))
    return -penalty * max(low_ratio - reference, 0.0)


def _normalize_training_commands(
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None,
) -> list[ResidualTrainingCommand]:
    if not training_commands:
        return []
    normalized: list[ResidualTrainingCommand] = []
    for item in training_commands:
        if isinstance(item, ResidualTrainingCommand):
            command = item.command
            weight = item.weight
        else:
            command = item
            weight = 1.0
        if command is not None and not isinstance(command, PolicyCommand):
            raise TypeError(f"training command must be PolicyCommand, got {type(command).__name__}")
        if not math.isfinite(float(weight)) or float(weight) <= 0.0:
            raise ValueError(f"training command weight must be positive and finite, got {weight!r}")
        if isinstance(item, ResidualTrainingCommand) and float(item.weight) == float(weight):
            normalized.append(item)
        else:
            normalized.append(ResidualTrainingCommand(command=command, weight=float(weight)))
    return normalized


def _normalize_training_sequences(
    training_sequences: Sequence[ResidualTrainingSequence] | None,
) -> list[ResidualTrainingSequence]:
    if not training_sequences:
        return []
    normalized: list[ResidualTrainingSequence] = []
    for sequence in training_sequences:
        if not isinstance(sequence, ResidualTrainingSequence):
            raise TypeError(f"training sequence must be ResidualTrainingSequence, got {type(sequence).__name__}")
        if not math.isfinite(float(sequence.weight)) or float(sequence.weight) <= 0.0:
            raise ValueError(f"training sequence weight must be positive and finite, got {sequence.weight!r}")
        if not sequence.segments:
            raise ValueError("training sequence must contain at least one segment")
        segments: list[ResidualTrainingSegment] = []
        for segment in sequence.segments:
            if not isinstance(segment, ResidualTrainingSegment):
                raise TypeError(f"training sequence segment must be ResidualTrainingSegment, got {type(segment).__name__}")
            if not isinstance(segment.command, PolicyCommand):
                raise TypeError(f"training sequence command must be PolicyCommand, got {type(segment.command).__name__}")
            if int(segment.steps) <= 0:
                raise ValueError(f"training sequence segment steps must be positive, got {segment.steps!r}")
            segments.append(ResidualTrainingSegment(command=segment.command, steps=int(segment.steps)))
        normalized.append(ResidualTrainingSequence(segments=tuple(segments), weight=float(sequence.weight)))
    return normalized


def _validate_score_normalization(score_normalization: str) -> str:
    value = str(score_normalization).strip().lower()
    if value not in SCORE_NORMALIZATION_CHOICES:
        raise ValueError(
            "score_normalization must be one of "
            f"{', '.join(SCORE_NORMALIZATION_CHOICES)}, got {score_normalization!r}"
        )
    return value


def _validate_clearance_quantile(clearance_quantile: float) -> float:
    value = float(clearance_quantile)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"clearance quantile must be finite and in [0, 1], got {clearance_quantile!r}")
    return value


def _validate_low_clearance_ratio(value: float, *, name: str = "low_clearance_ratio") -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return parsed


def _normalize_reference_low_clearance_ratios(
    values: Sequence[float] | None,
    *,
    objective_count: int,
) -> list[float | None]:
    if objective_count < 0:
        raise ValueError(f"objective_count must be non-negative, got {objective_count!r}")
    if values is None or len(values) == 0:
        return [None] * objective_count
    if len(values) != objective_count:
        raise ValueError(
            "reference_low_clearance_ratios must match the number of training objectives "
            f"({objective_count}), got {len(values)}"
        )
    return [
        _validate_low_clearance_ratio(value, name="reference_low_clearance_ratio")
        for value in values
    ]


def _score_for_aggregation(episode: Mapping[str, Any], *, score_normalization: str) -> float:
    score = float(episode.get("score", 0.0))
    mode = _validate_score_normalization(score_normalization)
    if mode == SCORE_NORMALIZATION_TOTAL:
        return score
    requested_steps = max(1, int(episode.get("requested_steps", 0)))
    return float(score / requested_steps)


def aggregate_training_scores(scores: Sequence[float], weights: Sequence[float], worst_case_score_weight: float = 0.0) -> float:
    if len(scores) == 0:
        raise ValueError("at least one score is required")
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")
    if not math.isfinite(float(worst_case_score_weight)) or not 0.0 <= float(worst_case_score_weight) <= 1.0:
        raise ValueError(
            "worst_case_score_weight must be finite and in [0, 1], "
            f"got {worst_case_score_weight!r}"
        )
    score_values = np.asarray(scores, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if np.any(~np.isfinite(score_values)):
        raise ValueError("scores must be finite")
    if np.any(~np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
        raise ValueError("weights must be positive and finite")
    weighted_mean = float(np.average(score_values, weights=weight_values))
    worst_score = float(np.min(score_values))
    blend = float(worst_case_score_weight)
    return float((1.0 - blend) * weighted_mean + blend * worst_score)


class _ConstantResidualModule:  # created dynamically after torch import
    pass


def export_constant_residual_policy(
    residual: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()

    class ConstantResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, residual_values: np.ndarray) -> None:
            super().__init__()
            tensor = torch.as_tensor(residual_values.reshape(1, ACTION_SIZE), dtype=torch.float32)
            self.residual = torch.nn.Parameter(tensor, requires_grad=False)

        def forward(self, obs: Any) -> Any:  # noqa: ANN401 - torch module signature
            batch = obs.shape[0]
            return self.residual.expand(batch, ACTION_SIZE)

    arr = np.asarray(residual, dtype=np.float32).reshape(ACTION_SIZE)
    module = ConstantResidualPolicy(arr)
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    resolved_input_size = int(input_size)
    if resolved_input_size <= 0:
        raise ValueError("input_size must be positive")
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # Keep exporter behavior aligned with the neural BC exporter and avoid
        # depending on PyTorch's version-dependent default exporter selection.
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "constant_residual_policy",
                "residual": arr.tolist(),
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_phase_contact_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()

    class PhaseContactResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, parameter_values: np.ndarray) -> None:
            super().__init__()
            tensor = torch.as_tensor(
                parameter_values.reshape(PHASE_CONTACT_FEATURE_SIZE, ACTION_SIZE),
                dtype=torch.float32,
            )
            self.weights = torch.nn.Parameter(tensor, requires_grad=False)

        def forward(self, obs: Any) -> Any:  # noqa: ANN401 - torch module signature
            batch = obs.shape[0]
            ones = torch.ones((batch, 1), dtype=obs.dtype, device=obs.device)
            features = torch.cat(
                (ones, obs[:, PHASE_CONTACT_OBSERVATION_START:PHASE_CONTACT_OBSERVATION_STOP]),
                dim=1,
            )
            return torch.tanh(features @ self.weights)

    arr = np.asarray(parameters, dtype=np.float32).reshape(PHASE_CONTACT_PARAMETER_SIZE)
    resolved_input_size = int(input_size)
    if resolved_input_size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            f"phase_contact residual policy requires input_size >= {PHASE_CONTACT_OBSERVATION_STOP}"
        )
    module = PhaseContactResidualPolicy(arr)
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "phase_contact_residual_policy",
                "parameters": arr.tolist(),
                "feature_names": ["bias", "left_contact", "right_contact", "phase_cos", "phase_sin"],
                "observation_slice": [
                    PHASE_CONTACT_OBSERVATION_START,
                    PHASE_CONTACT_OBSERVATION_STOP,
                ],
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_command_state_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()

    class CommandStateResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, parameter_values: np.ndarray) -> None:
            super().__init__()
            tensor = torch.as_tensor(
                parameter_values.reshape(COMMAND_STATE_FEATURE_SIZE, COMMAND_STATE_OUTPUT_SIZE),
                dtype=torch.float32,
            )
            self.weights = torch.nn.Parameter(tensor, requires_grad=False)
            projection = torch.zeros((COMMAND_STATE_OUTPUT_SIZE, ACTION_SIZE), dtype=torch.float32)
            for source_index, action_index in enumerate(COMMAND_STATE_ACTION_INDICES):
                projection[source_index, action_index] = 1.0
            self.register_buffer("projection", projection)

        def forward(self, obs: Any) -> Any:  # noqa: ANN401 - torch module signature
            batch = obs.shape[0]
            ones = torch.ones((batch, 1), dtype=obs.dtype, device=obs.device)
            leg_offsets = obs[:, list(COMMAND_STATE_LEG_JOINT_OFFSET_INDICES)]
            last_actions = obs[:, list(COMMAND_STATE_LAST_ACTION_INDICES)]
            features = torch.cat(
                (
                    ones,
                    obs[:, COMMAND_STATE_COMMAND_SLICE],
                    obs[:, COMMAND_STATE_CONTACT_PHASE_SLICE],
                    leg_offsets,
                    last_actions,
                ),
                dim=1,
            )
            return torch.tanh(features @ self.weights) @ self.projection

    arr = np.asarray(parameters, dtype=np.float32).reshape(COMMAND_STATE_PARAMETER_SIZE)
    resolved_input_size = int(input_size)
    if resolved_input_size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            f"command_state residual policy requires input_size >= {PHASE_CONTACT_OBSERVATION_STOP}"
        )
    module = CommandStateResidualPolicy(arr)
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "command_state_residual_policy",
                "parameters": arr.tolist(),
                "feature_names": [
                    "bias",
                    "command_vx",
                    "command_vy",
                    "command_yaw",
                    "left_contact",
                    "right_contact",
                    "phase_cos",
                    "phase_sin",
                    "left_hip_pitch_offset",
                    "left_knee_offset",
                    "left_ankle_offset",
                    "right_hip_pitch_offset",
                    "right_knee_offset",
                    "right_ankle_offset",
                    "left_hip_pitch_last_action",
                    "left_knee_last_action",
                    "left_ankle_last_action",
                    "right_hip_pitch_last_action",
                    "right_knee_last_action",
                    "right_ankle_last_action",
                ],
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "controlled_action_indices": list(COMMAND_STATE_ACTION_INDICES),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_command_state_mlp_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()
    values = np.asarray(parameters, dtype=np.float32).reshape(COMMAND_STATE_MLP_PARAMETER_SIZE)
    cursor = 0
    linear_size = COMMAND_STATE_PARAMETER_SIZE
    linear = values[cursor : cursor + linear_size].reshape(
        COMMAND_STATE_FEATURE_SIZE,
        COMMAND_STATE_OUTPUT_SIZE,
    )
    cursor += linear_size
    input_hidden_size = COMMAND_STATE_FEATURE_SIZE * COMMAND_STATE_MLP_HIDDEN_SIZE
    input_hidden = values[cursor : cursor + input_hidden_size].reshape(
        COMMAND_STATE_FEATURE_SIZE,
        COMMAND_STATE_MLP_HIDDEN_SIZE,
    )
    cursor += input_hidden_size
    hidden_bias = values[cursor : cursor + COMMAND_STATE_MLP_HIDDEN_SIZE]
    cursor += COMMAND_STATE_MLP_HIDDEN_SIZE
    hidden_output = values[cursor:].reshape(
        COMMAND_STATE_MLP_HIDDEN_SIZE,
        COMMAND_STATE_OUTPUT_SIZE,
    )

    class CommandStateMlpResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("linear", torch.as_tensor(linear, dtype=torch.float32))
            self.register_buffer("input_hidden", torch.as_tensor(input_hidden, dtype=torch.float32))
            self.register_buffer("hidden_bias", torch.as_tensor(hidden_bias, dtype=torch.float32))
            self.register_buffer("hidden_output", torch.as_tensor(hidden_output, dtype=torch.float32))
            projection = torch.zeros((COMMAND_STATE_OUTPUT_SIZE, ACTION_SIZE), dtype=torch.float32)
            for source_index, action_index in enumerate(COMMAND_STATE_ACTION_INDICES):
                projection[source_index, action_index] = 1.0
            self.register_buffer("projection", projection)

        def forward(self, obs: Any) -> Any:  # noqa: ANN401
            batch = obs.shape[0]
            ones = torch.ones((batch, 1), dtype=obs.dtype, device=obs.device)
            features = torch.cat(
                (
                    ones,
                    obs[:, COMMAND_STATE_COMMAND_SLICE],
                    obs[:, COMMAND_STATE_CONTACT_PHASE_SLICE],
                    obs[:, list(COMMAND_STATE_LEG_JOINT_OFFSET_INDICES)],
                    obs[:, list(COMMAND_STATE_LAST_ACTION_INDICES)],
                ),
                dim=1,
            )
            hidden = torch.tanh(features @ self.input_hidden + self.hidden_bias)
            leg_action = torch.tanh(features @ self.linear + hidden @ self.hidden_output)
            return leg_action @ self.projection

    resolved_input_size = int(input_size)
    if resolved_input_size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            f"command_state_mlp residual policy requires input_size >= {PHASE_CONTACT_OBSERVATION_STOP}"
        )
    module = CommandStateMlpResidualPolicy()
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "command_state_mlp_residual_policy",
                "parameters": values.tolist(),
                "hidden_size": COMMAND_STATE_MLP_HIDDEN_SIZE,
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "controlled_action_indices": list(COMMAND_STATE_ACTION_INDICES),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_contact_phase_lift_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()
    values = np.asarray(parameters, dtype=np.float32).reshape(2, CONTACT_PHASE_LIFT_FEATURE_SIZE, 3)

    class ContactPhaseLiftResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weights", torch.as_tensor(values, dtype=torch.float32))
            projection = torch.zeros((CONTACT_PHASE_LIFT_PARAMETER_SIZE // CONTACT_PHASE_LIFT_FEATURE_SIZE, ACTION_SIZE), dtype=torch.float32)
            for source_index, action_index in enumerate(CONTACT_PHASE_LIFT_ACTION_INDICES):
                projection[source_index, action_index] = 1.0
            self.register_buffer("projection", projection)

        def forward(self, obs: Any) -> Any:  # noqa: ANN401
            left_contact = obs[:, 97:98]
            right_contact = obs[:, 98:99]
            phase_cos = obs[:, 99:100]
            phase_sin = obs[:, 100:101]
            left_swing = torch.clamp(1.0 - left_contact, min=0.0, max=1.0)
            right_swing = torch.clamp(1.0 - right_contact, min=0.0, max=1.0)
            left_features = torch.cat(
                (left_swing, left_swing * phase_cos, left_swing * phase_sin),
                dim=1,
            )
            right_features = torch.cat(
                (right_swing, right_swing * phase_cos, right_swing * phase_sin),
                dim=1,
            )
            left_action = torch.tanh(left_features @ self.weights[0])
            right_action = torch.tanh(right_features @ self.weights[1])
            return torch.cat((left_action, right_action), dim=1) @ self.projection

    resolved_input_size = int(input_size)
    if resolved_input_size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            f"contact_phase_lift residual policy requires input_size >= {PHASE_CONTACT_OBSERVATION_STOP}"
        )
    module = ContactPhaseLiftResidualPolicy()
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "contact_phase_lift_residual_policy",
                "parameters": values.reshape(CONTACT_PHASE_LIFT_PARAMETER_SIZE).tolist(),
                "feature_names": ["swing", "swing_phase_cos", "swing_phase_sin"],
                "observation_slice": [
                    PHASE_CONTACT_OBSERVATION_START,
                    PHASE_CONTACT_OBSERVATION_STOP,
                ],
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "controlled_action_indices": list(CONTACT_PHASE_LIFT_ACTION_INDICES),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_command_contact_phase_lift_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()
    values = np.asarray(parameters, dtype=np.float32).reshape(2, COMMAND_CONTACT_PHASE_LIFT_FEATURE_SIZE, 3)

    class CommandContactPhaseLiftResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weights", torch.as_tensor(values, dtype=torch.float32))
            projection = torch.zeros(
                (len(COMMAND_CONTACT_PHASE_LIFT_ACTION_INDICES), ACTION_SIZE),
                dtype=torch.float32,
            )
            for source_index, action_index in enumerate(COMMAND_CONTACT_PHASE_LIFT_ACTION_INDICES):
                projection[source_index, action_index] = 1.0
            self.register_buffer("projection", projection)

        def _leg_features(self, swing: Any, phase_cos: Any, phase_sin: Any, command: Any) -> Any:  # noqa: ANN401
            desired_vx = command[:, 0:1]
            desired_vy = command[:, 1:2]
            desired_yaw = command[:, 2:3]
            return torch.cat(
                (
                    swing,
                    swing * phase_cos,
                    swing * phase_sin,
                    swing * desired_vx,
                    swing * desired_vx * phase_cos,
                    swing * desired_vx * phase_sin,
                    swing * desired_vy,
                    swing * desired_vy * phase_cos,
                    swing * desired_vy * phase_sin,
                    swing * desired_yaw,
                    swing * desired_yaw * phase_cos,
                    swing * desired_yaw * phase_sin,
                ),
                dim=1,
            )

        def forward(self, obs: Any) -> Any:  # noqa: ANN401
            left_contact = obs[:, 97:98]
            right_contact = obs[:, 98:99]
            phase_cos = obs[:, 99:100]
            phase_sin = obs[:, 100:101]
            command = obs[:, 101:104]
            left_swing = torch.clamp(1.0 - left_contact, min=0.0, max=1.0)
            right_swing = torch.clamp(1.0 - right_contact, min=0.0, max=1.0)
            left_features = self._leg_features(left_swing, phase_cos, phase_sin, command)
            right_features = self._leg_features(right_swing, phase_cos, phase_sin, command)
            left_action = torch.tanh(left_features @ self.weights[0])
            right_action = torch.tanh(right_features @ self.weights[1])
            return torch.cat((left_action, right_action), dim=1) @ self.projection

    resolved_input_size = int(input_size)
    if resolved_input_size < COMMAND_CONTACT_PHASE_LIFT_INPUT_SIZE:
        raise ValueError(
            "command_contact_phase_lift residual policy requires input_size >= "
            f"{COMMAND_CONTACT_PHASE_LIFT_INPUT_SIZE}"
        )
    module = CommandContactPhaseLiftResidualPolicy()
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "command_contact_phase_lift_residual_policy",
                "parameters": values.reshape(COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE).tolist(),
                "feature_names": [
                    "swing",
                    "swing_phase_cos",
                    "swing_phase_sin",
                    "swing_vx",
                    "swing_vx_phase_cos",
                    "swing_vx_phase_sin",
                    "swing_vy",
                    "swing_vy_phase_cos",
                    "swing_vy_phase_sin",
                    "swing_yaw",
                    "swing_yaw_phase_cos",
                    "swing_yaw_phase_sin",
                ],
                "contact_phase_observation_slice": [
                    PHASE_CONTACT_OBSERVATION_START,
                    PHASE_CONTACT_OBSERVATION_STOP,
                ],
                "command_observation_slice": [
                    COMMAND_CONTACT_PHASE_LIFT_COMMAND_SLICE.start,
                    COMMAND_CONTACT_PHASE_LIFT_COMMAND_SLICE.stop,
                ],
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "controlled_action_indices": list(COMMAND_CONTACT_PHASE_LIFT_ACTION_INDICES),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def export_contact_phase_harmonic_lift_residual_policy(
    parameters: np.ndarray | list[float],
    *,
    output_onnx: Path,
    output_checkpoint: Path | None = None,
    input_size: int = 101,
) -> None:
    torch = _import_torch()
    values = np.asarray(parameters, dtype=np.float32).reshape(2, CONTACT_PHASE_HARMONIC_LIFT_FEATURE_SIZE, 3)

    class ContactPhaseHarmonicLiftResidualPolicy(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weights", torch.as_tensor(values, dtype=torch.float32))
            projection = torch.zeros(
                (len(CONTACT_PHASE_HARMONIC_LIFT_ACTION_INDICES), ACTION_SIZE),
                dtype=torch.float32,
            )
            for source_index, action_index in enumerate(CONTACT_PHASE_HARMONIC_LIFT_ACTION_INDICES):
                projection[source_index, action_index] = 1.0
            self.register_buffer("projection", projection)

        def _leg_features(self, swing: Any, phase_cos: Any, phase_sin: Any) -> Any:  # noqa: ANN401
            phase_cos_2 = phase_cos * phase_cos - phase_sin * phase_sin
            phase_sin_2 = 2.0 * phase_cos * phase_sin
            phase_cos_3 = phase_cos * phase_cos_2 - phase_sin * phase_sin_2
            phase_sin_3 = phase_sin * phase_cos_2 + phase_cos * phase_sin_2
            return torch.cat(
                (
                    swing,
                    swing * phase_cos,
                    swing * phase_sin,
                    swing * phase_cos_2,
                    swing * phase_sin_2,
                    swing * phase_cos_3,
                    swing * phase_sin_3,
                ),
                dim=1,
            )

        def forward(self, obs: Any) -> Any:  # noqa: ANN401
            left_contact = obs[:, 97:98]
            right_contact = obs[:, 98:99]
            phase_cos = obs[:, 99:100]
            phase_sin = obs[:, 100:101]
            left_swing = torch.clamp(1.0 - left_contact, min=0.0, max=1.0)
            right_swing = torch.clamp(1.0 - right_contact, min=0.0, max=1.0)
            left_features = self._leg_features(left_swing, phase_cos, phase_sin)
            right_features = self._leg_features(right_swing, phase_cos, phase_sin)
            left_action = torch.tanh(left_features @ self.weights[0])
            right_action = torch.tanh(right_features @ self.weights[1])
            return torch.cat((left_action, right_action), dim=1) @ self.projection

    resolved_input_size = int(input_size)
    if resolved_input_size < PHASE_CONTACT_OBSERVATION_STOP:
        raise ValueError(
            f"contact_phase_harmonic_lift residual policy requires input_size >= {PHASE_CONTACT_OBSERVATION_STOP}"
        )
    module = ContactPhaseHarmonicLiftResidualPolicy()
    module.eval()
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(output_onnx),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_kind": "contact_phase_harmonic_lift_residual_policy",
                "parameters": values.reshape(CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE).tolist(),
                "feature_names": [
                    "swing",
                    "swing_phase_cos",
                    "swing_phase_sin",
                    "swing_phase_cos_2",
                    "swing_phase_sin_2",
                    "swing_phase_cos_3",
                    "swing_phase_sin_3",
                ],
                "observation_slice": [
                    PHASE_CONTACT_OBSERVATION_START,
                    PHASE_CONTACT_OBSERVATION_STOP,
                ],
                "observation_size": resolved_input_size,
                "action_size": ACTION_SIZE,
                "controlled_action_indices": list(CONTACT_PHASE_HARMONIC_LIFT_ACTION_INDICES),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_checkpoint,
        )


def load_residual_initial_parameters(
    checkpoint_path: Path,
    *,
    actor_kind: str,
) -> np.ndarray:
    torch = _import_torch()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = np.asarray(payload.get("parameters", payload.get("residual", [])), dtype=np.float32).reshape(-1)
    if actor_kind == RESIDUAL_ACTOR_COMMAND_STATE_MLP:
        if source.size == COMMAND_STATE_PARAMETER_SIZE:
            initialized = np.zeros(COMMAND_STATE_MLP_PARAMETER_SIZE, dtype=np.float32)
            initialized[:COMMAND_STATE_PARAMETER_SIZE] = source
            return initialized
        return source.reshape(COMMAND_STATE_MLP_PARAMETER_SIZE)
    expected = {
        RESIDUAL_ACTOR_CONSTANT: ACTION_SIZE,
        RESIDUAL_ACTOR_PHASE_CONTACT: PHASE_CONTACT_PARAMETER_SIZE,
        RESIDUAL_ACTOR_COMMAND_STATE: COMMAND_STATE_PARAMETER_SIZE,
        RESIDUAL_ACTOR_CONTACT_PHASE_LIFT: CONTACT_PHASE_LIFT_PARAMETER_SIZE,
        RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT: CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE,
        RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT: COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE,
    }[actor_kind]
    return source.reshape(expected)


def _residual_source_from_parameters(
    actor_kind: str,
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray | Callable[[np.ndarray], np.ndarray]:
    values = np.asarray(parameters, dtype=np.float32).reshape(-1)
    if actor_kind == RESIDUAL_ACTOR_CONSTANT:
        return values.reshape(ACTION_SIZE)
    if actor_kind == RESIDUAL_ACTOR_PHASE_CONTACT:
        shaped = values.reshape(PHASE_CONTACT_PARAMETER_SIZE)
        return lambda observation: phase_contact_residual_action(observation, shaped)
    if actor_kind == RESIDUAL_ACTOR_COMMAND_STATE:
        shaped = values.reshape(COMMAND_STATE_PARAMETER_SIZE)
        return lambda observation: command_state_residual_action(observation, shaped)
    if actor_kind == RESIDUAL_ACTOR_COMMAND_STATE_MLP:
        shaped = values.reshape(COMMAND_STATE_MLP_PARAMETER_SIZE)
        return lambda observation: command_state_mlp_residual_action(observation, shaped)
    if actor_kind == RESIDUAL_ACTOR_CONTACT_PHASE_LIFT:
        shaped = values.reshape(CONTACT_PHASE_LIFT_PARAMETER_SIZE)
        return lambda observation: contact_phase_lift_residual_action(observation, shaped)
    if actor_kind == RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT:
        shaped = values.reshape(CONTACT_PHASE_HARMONIC_LIFT_PARAMETER_SIZE)
        return lambda observation: contact_phase_harmonic_lift_residual_action(observation, shaped)
    if actor_kind == RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT:
        shaped = values.reshape(COMMAND_CONTACT_PHASE_LIFT_PARAMETER_SIZE)
        return lambda observation: command_contact_phase_lift_residual_action(observation, shaped)
    raise ValueError(f"unsupported residual actor kind: {actor_kind}")


def _write_residual_profile(
    *,
    profile_name: str,
    teacher_profile: str,
    residual_onnx_path: str,
    output_dir: Path,
    description: str | None,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    force: bool,
    actor_kind: str = RESIDUAL_ACTOR_CONSTANT,
) -> Path:
    teacher = PolicyProfile.load(teacher_profile)
    input_shape = list(teacher.model.input_shape)
    payload = build_replacement_profile_payload(
        name=profile_name,
        model_path=residual_onnx_path,
        template=teacher_profile,
        description=description or f"Residual policy fine-tuned on top of {teacher_profile}.",
        input_name="obs",
        output_name="continuous_actions",
        input_shape=input_shape,
        output_shape=[1, 14],
    )
    metadata = payload.setdefault("metadata", {})
    metadata["generated_by"] = "soridormi_m619_residual_rl"
    metadata["training_output"] = str(Path(residual_onnx_path).parent)
    metadata["promotion_status"] = "blocked_clearance_gate"
    for stale_key in ("initial_checkpoint", "scenario_suite", "clearance_readiness"):
        metadata.pop(stale_key, None)
    payload["model"]["kind"] = "residual_onnx"
    payload["residual_policy"] = {
        "teacher_profile": teacher_profile,
        "actor_kind": actor_kind,
        "residual_scale": float(residual_scale),
        "residual_clip_abs": float(residual_clip_abs),
        "final_action_clip_abs": 0.0 if final_action_clip_abs is None else float(final_action_clip_abs),
        "combination": "final_action = teacher_action + residual_scale * clip(residual_model(obs))",
    }
    path = output_dir / f"{profile_name}.yaml"
    if path.exists() and not force:
        raise FileExistsError(f"Residual profile already exists: {path}. Pass --force-profile to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train_residual_policy(
    *,
    teacher_profile: str,
    output_dir: Path,
    steps_per_episode: int,
    optimization_config: ResidualOptimizationConfig,
    residual_scale: float,
    residual_clip_abs: float,
    final_action_clip_abs: float | None,
    reward_config: WalkingRewardConfig,
    profile_name: str | None = None,
    profile_output_dir: Path = Path("configs/policies"),
    force_profile: bool = False,
    host: str = "127.0.0.1",
    port: int = 5555,
    actor_kind: str = RESIDUAL_ACTOR_CONSTANT,
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None = None,
    training_sequences: Sequence[ResidualTrainingSequence] | None = None,
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    episodic_clearance_gap_weight: float = 0.0,
    episodic_clearance_quantile: float = 0.25,
    episodic_clearance_quantile_gap_weight: float = 0.0,
    reference_low_clearance_ratios: Sequence[float] | None = None,
    low_clearance_regression_penalty_weight: float = 0.0,
    worst_case_score_weight: float = 0.0,
    score_normalization: str = SCORE_NORMALIZATION_TOTAL,
    initial_checkpoint: Path | None = None,
    final_score_breakdown: bool = False,
) -> ResidualPolicyTrainResult:
    errors: list[str] = []
    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / DEFAULT_RESIDUAL_ONNX_NAME
    checkpoint_path = output_dir / DEFAULT_RESIDUAL_PT_NAME
    metrics_path = output_dir / "residual_train_metrics.json"
    report_path = output_dir / "residual_train_report.md"
    profile_path: Path | None = None
    optimization: ResidualOptimizationResult | None = None
    input_size: int | None = None
    score_breakdown: dict[str, Any] | None = None

    try:
        score_normalization = _validate_score_normalization(score_normalization)
        if not math.isfinite(float(episodic_clearance_gap_weight)) or float(episodic_clearance_gap_weight) < 0.0:
            raise ValueError(
                "episodic_clearance_gap_weight must be non-negative and finite, "
                f"got {episodic_clearance_gap_weight!r}"
            )
        if (
            not math.isfinite(float(episodic_clearance_quantile_gap_weight))
            or float(episodic_clearance_quantile_gap_weight) < 0.0
        ):
            raise ValueError(
                "episodic_clearance_quantile_gap_weight must be non-negative and finite, "
                f"got {episodic_clearance_quantile_gap_weight!r}"
            )
        episodic_clearance_quantile = _validate_clearance_quantile(episodic_clearance_quantile)
        objective_count = len(_normalize_training_commands(training_commands)) + len(
            _normalize_training_sequences(training_sequences)
        )
        if objective_count == 0:
            objective_count = 1
        normalized_reference_low_clearance_ratios = _normalize_reference_low_clearance_ratios(
            reference_low_clearance_ratios,
            objective_count=objective_count,
        )
        if (
            not math.isfinite(float(low_clearance_regression_penalty_weight))
            or float(low_clearance_regression_penalty_weight) < 0.0
        ):
            raise ValueError(
                "low_clearance_regression_penalty_weight must be non-negative and finite, "
                f"got {low_clearance_regression_penalty_weight!r}"
            )
        teacher = PolicyProfile.load(teacher_profile)
        input_size = _profile_input_size(teacher)
        initial_parameters = (
            None
            if initial_checkpoint is None
            else load_residual_initial_parameters(initial_checkpoint, actor_kind=actor_kind)
        )
        if actor_kind == RESIDUAL_ACTOR_CONSTANT:
            optimization = optimize_residual_bias(
                lambda residual: evaluate_residual_bias_live(
                    residual,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_constant_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_PHASE_CONTACT:
            optimization = optimize_phase_contact_residual(
                lambda parameters: evaluate_phase_contact_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_phase_contact_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_COMMAND_STATE:
            optimization = optimize_command_state_residual(
                lambda parameters: evaluate_command_state_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_command_state_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_COMMAND_STATE_MLP:
            optimization = optimize_command_state_mlp_residual(
                lambda parameters: evaluate_command_state_mlp_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
                initial_mean=initial_parameters,
            )
            export_command_state_mlp_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_CONTACT_PHASE_LIFT:
            optimization = optimize_contact_phase_lift_residual(
                lambda parameters: evaluate_contact_phase_lift_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_contact_phase_lift_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT:
            optimization = optimize_contact_phase_harmonic_lift_residual(
                lambda parameters: evaluate_contact_phase_harmonic_lift_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_contact_phase_harmonic_lift_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        elif actor_kind == RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT:
            optimization = optimize_command_contact_phase_lift_residual(
                lambda parameters: evaluate_command_contact_phase_lift_residual_live(
                    parameters,
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                ),
                config=optimization_config,
            )
            export_command_contact_phase_lift_residual_policy(
                optimization.best_residual,
                output_onnx=onnx_path,
                output_checkpoint=checkpoint_path,
                input_size=input_size,
            )
        else:
            raise ValueError(f"unsupported residual actor kind: {actor_kind}")
        if profile_name:
            profile_path = _write_residual_profile(
                profile_name=profile_name,
                teacher_profile=teacher_profile,
                residual_onnx_path=str(onnx_path),
                output_dir=profile_output_dir,
                description=None,
                residual_scale=residual_scale,
                residual_clip_abs=residual_clip_abs,
                final_action_clip_abs=final_action_clip_abs,
                force=force_profile,
                actor_kind=actor_kind,
            )
        if final_score_breakdown and optimization is not None:
            try:
                score_breakdown = _evaluate_residual_live_breakdown(
                    _residual_source_from_parameters(actor_kind, optimization.best_residual),
                    teacher_profile=teacher_profile,
                    steps=steps_per_episode,
                    residual_scale=residual_scale,
                    residual_clip_abs=residual_clip_abs,
                    final_action_clip_abs=final_action_clip_abs,
                    reward_config=reward_config,
                    host=host,
                    port=port,
                    training_commands=training_commands,
                    training_sequences=training_sequences,
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
                    episodic_clearance_gap_weight=episodic_clearance_gap_weight,
                    episodic_clearance_quantile=episodic_clearance_quantile,
                    episodic_clearance_quantile_gap_weight=episodic_clearance_quantile_gap_weight,
                    reference_low_clearance_ratios=normalized_reference_low_clearance_ratios,
                    low_clearance_regression_penalty_weight=low_clearance_regression_penalty_weight,
                    worst_case_score_weight=worst_case_score_weight,
                    score_normalization=score_normalization,
                )
            except Exception as exc:  # pragma: no cover - diagnostic live simulator path
                warnings.append(f"final score breakdown failed: {exc!r}")
    except Exception as exc:  # pragma: no cover - live simulator/training environment
        errors.append(repr(exc))

    payload = {
        "schema_version": 1,
        "teacher_profile": teacher_profile,
        "actor_kind": actor_kind,
        "training_commands": []
        if not training_commands
        else [command.describe() for command in _normalize_training_commands(training_commands)],
        "training_sequences": []
        if not training_sequences
        else [sequence.describe() for sequence in _normalize_training_sequences(training_sequences)],
        "policy_input_size": input_size,
        "output_dir": str(output_dir),
        "steps_per_episode": int(steps_per_episode),
        "residual_scale": float(residual_scale),
        "residual_clip_abs": float(residual_clip_abs),
        "final_action_clip_abs": final_action_clip_abs,
        "reward_config": asdict(reward_config),
        "episodic_clearance_weight": float(episodic_clearance_weight),
        "episodic_low_clearance_penalty_weight": float(episodic_low_clearance_penalty_weight),
        "episodic_clearance_gap_weight": float(episodic_clearance_gap_weight),
        "episodic_clearance_quantile": float(episodic_clearance_quantile),
        "episodic_clearance_quantile_gap_weight": float(episodic_clearance_quantile_gap_weight),
        "reference_low_clearance_ratios": [
            value for value in (reference_low_clearance_ratios or []) if value is not None
        ],
        "low_clearance_regression_penalty_weight": float(low_clearance_regression_penalty_weight),
        "worst_case_score_weight": float(worst_case_score_weight),
        "score_normalization": score_normalization,
        "final_score_breakdown_requested": bool(final_score_breakdown),
        "score_breakdown": score_breakdown,
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        "optimization_config": asdict(optimization_config),
        "optimization": None if optimization is None else asdict(optimization),
        "residual_onnx_path": str(onnx_path) if onnx_path.exists() else None,
        "residual_checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "profile_name": profile_name,
        "profile_path": None if profile_path is None else str(profile_path),
        "errors": errors,
        "warnings": warnings,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    return ResidualPolicyTrainResult(
        ok=not errors,
        teacher_profile=teacher_profile,
        output_dir=str(output_dir),
        residual_onnx_path=str(onnx_path) if onnx_path.exists() else None,
        residual_checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        profile_name=profile_name,
        profile_path=None if profile_path is None else str(profile_path),
        optimization=None if optimization is None else asdict(optimization),
        errors=errors,
        warnings=warnings,
    )


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Residual Policy Fine-Tuning Report",
        "",
        f"Teacher profile: `{payload['teacher_profile']}`",
        f"Actor kind: `{payload['actor_kind']}`",
        f"Output directory: `{payload['output_dir']}`",
        f"Residual scale: `{payload['residual_scale']}`",
        f"Worst-case score weight: `{payload.get('worst_case_score_weight', 0.0):.6g}`",
        f"Episodic clearance gap weight: `{payload.get('episodic_clearance_gap_weight', 0.0):.6g}`",
        f"Episodic clearance quantile: `{payload.get('episodic_clearance_quantile', 0.25):.6g}`",
        f"Episodic clearance quantile gap weight: `{payload.get('episodic_clearance_quantile_gap_weight', 0.0):.6g}`",
        f"Low-clearance regression penalty weight: `{payload.get('low_clearance_regression_penalty_weight', 0.0):.6g}`",
        f"Score normalization: `{payload.get('score_normalization', SCORE_NORMALIZATION_TOTAL)}`",
        f"Policy input size: `{payload.get('policy_input_size')}`",
        "",
    ]
    training_commands = payload.get("training_commands") or []
    if training_commands:
        lines.extend(
            [
                "## Training commands",
                "",
                "| vx | vy | yaw | weight |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for command in training_commands:
            lines.append(
                "| {vx:.6g} | {vy:.6g} | {yaw:.6g} | {weight:.6g} |".format(
                    vx=float(command.get("x_velocity", 0.0)),
                    vy=float(command.get("y_velocity", 0.0)),
                    yaw=float(command.get("yaw_velocity", 0.0)),
                    weight=float(command.get("weight", 1.0)),
                )
            )
        lines.append("")
    training_sequences = payload.get("training_sequences") or []
    if training_sequences:
        lines.extend(
            [
                "## Training sequences",
                "",
                "| sequence | segment | vx | vy | yaw | steps | weight |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for sequence_index, sequence in enumerate(training_sequences):
            segments = sequence.get("segments") or []
            for segment_index, segment in enumerate(segments):
                lines.append(
                    "| {sequence_index} | {segment_index} | {vx:.6g} | {vy:.6g} | {yaw:.6g} | {steps:d} | {weight:.6g} |".format(
                        sequence_index=sequence_index,
                        segment_index=segment_index,
                        vx=float(segment.get("x_velocity", 0.0)),
                        vy=float(segment.get("y_velocity", 0.0)),
                        yaw=float(segment.get("yaw_velocity", 0.0)),
                        steps=int(segment.get("steps", 0)),
                        weight=float(sequence.get("weight", 1.0)),
                    )
                )
        lines.append("")
    optimization = payload.get("optimization")
    if optimization:
        lines.extend(
            [
                f"Best score: `{optimization['best_score']:.6g}`",
                f"Best parameter abs max: `{max(abs(float(x)) for x in optimization['best_residual']):.6g}`",
                "",
                "## Iterations",
                "",
                "| iteration | best score | mean score | elite mean | std mean |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in optimization.get("iterations", []):
            lines.append(
                f"| {item['iteration']} | {item['best_score']:.6g} | {item['mean_score']:.6g} | "
                f"{item['elite_mean_score']:.6g} | {item['distribution_std_mean']:.6g} |"
            )
        lines.append("")
    score_breakdown = payload.get("score_breakdown")
    if isinstance(score_breakdown, dict):
        lines.extend(
            [
                "## Final score breakdown",
                "",
                f"Aggregate score: `{float(score_breakdown.get('aggregate_score', 0.0)):.6g}`",
                f"Weighted mean score: `{float(score_breakdown.get('weighted_mean_score', 0.0)):.6g}`",
                f"Worst score: `{float(score_breakdown.get('worst_score', 0.0)):.6g}`",
                f"Score normalization: `{score_breakdown.get('score_normalization', SCORE_NORMALIZATION_TOTAL)}`",
                "",
                "| kind | index | weight | score | objective score | completed | terminated | median clearance m | low clearance | reference low clearance | regression penalty | mean gap | objective |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        has_segment_diagnostics = False
        for item in score_breakdown.get("items", []):
            if not isinstance(item, dict):
                continue
            objective = _describe_score_breakdown_item(item)
            episode = item.get("episode") if isinstance(item.get("episode"), Mapping) else {}
            completed = int(episode.get("completed_steps", 0)) if isinstance(episode, Mapping) else 0
            terminated = bool(episode.get("terminated", False)) if isinstance(episode, Mapping) else False
            median_clearance = _format_optional_float(episode.get("median_swing_clearance_m") if isinstance(episode, Mapping) else None)
            low_clearance = _format_optional_float(episode.get("low_clearance_ratio") if isinstance(episode, Mapping) else None)
            reference_low_clearance = _format_optional_float(episode.get("reference_low_clearance_ratio") if isinstance(episode, Mapping) else None)
            regression_penalty = _format_optional_float(episode.get("low_clearance_regression_penalty_total") if isinstance(episode, Mapping) else None)
            mean_gap = _format_optional_float(episode.get("mean_clearance_gap_ratio") if isinstance(episode, Mapping) else None)
            lines.append(
                "| {kind} | {index} | {weight:.6g} | {score:.6g} | {objective_score:.6g} | {completed:d} | {terminated} | {median_clearance} | {low_clearance} | {reference_low_clearance} | {regression_penalty} | {mean_gap} | {objective} |".format(
                    kind=str(item.get("kind", "unknown")),
                    index=int(item.get("index", 0)),
                    weight=float(item.get("weight", 1.0)),
                    score=float(item.get("score", 0.0)),
                    objective_score=float(item.get("objective_score", item.get("score", 0.0))),
                    completed=completed,
                    terminated="yes" if terminated else "no",
                    median_clearance=median_clearance,
                    low_clearance=low_clearance,
                    reference_low_clearance=reference_low_clearance,
                    regression_penalty=regression_penalty,
                    mean_gap=mean_gap,
                    objective=objective,
                )
            )
            if isinstance(episode, Mapping) and episode.get("segments"):
                has_segment_diagnostics = True
        lines.append("")
        if has_segment_diagnostics:
            lines.extend(
                [
                    "### Segment diagnostics",
                    "",
                    "| objective | segment | completed | terminated | samples | median clearance m | low clearance | mean gap | command |",
                    "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for item in score_breakdown.get("items", []):
                if not isinstance(item, dict):
                    continue
                episode = item.get("episode") if isinstance(item.get("episode"), Mapping) else {}
                if not isinstance(episode, Mapping):
                    continue
                objective_label = f"{item.get('kind', 'unknown')} {int(item.get('index', 0))}"
                for segment in episode.get("segments", []):
                    if not isinstance(segment, Mapping):
                        continue
                    lines.append(
                        "| {objective} | {segment_index} | {completed:d} | {terminated} | {samples:d} | {median_clearance} | {low_clearance} | {mean_gap} | {command} |".format(
                            objective=objective_label,
                            segment_index=int(segment.get("index", 0)),
                            completed=int(segment.get("completed_steps", 0)),
                            terminated="yes" if bool(segment.get("terminated", False)) else "no",
                            samples=int(segment.get("swing_clearance_sample_count", 0)),
                            median_clearance=_format_optional_float(segment.get("median_swing_clearance_m")),
                            low_clearance=_format_optional_float(segment.get("low_clearance_ratio")),
                            mean_gap=_format_optional_float(segment.get("mean_clearance_gap_ratio")),
                            command=_describe_command_payload(segment["command"]) if isinstance(segment.get("command"), Mapping) else "default",
                        )
                    )
            lines.append("")
    if payload.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in payload["warnings"]:
            lines.append(f"- `{warning}`")
        lines.append("")
    if payload.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for error in payload["errors"]:
            lines.append(f"- `{error}`")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")



def _format_optional_float(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{precision}g}"

def _describe_score_breakdown_item(item: Mapping[str, Any]) -> str:
    command = item.get("command")
    if isinstance(command, Mapping):
        return _describe_command_payload(command)
    sequence = item.get("sequence")
    if isinstance(sequence, Mapping):
        segments = sequence.get("segments") or []
        total_steps = int(sequence.get("total_steps", 0))
        return f"{len(segments)} segments / {total_steps} steps"
    return "default command"


def _describe_command_payload(command: Mapping[str, Any]) -> str:
    return "vx={vx:.6g}, vy={vy:.6g}, yaw={yaw:.6g}".format(
        vx=float(command.get("x_velocity", 0.0)),
        vy=float(command.get("y_velocity", 0.0)),
        yaw=float(command.get("yaw_velocity", 0.0)),
    )


def _profile_input_size(profile: PolicyProfile) -> int:
    shape = list(profile.model.input_shape)
    if len(shape) != 2 or not isinstance(shape[-1], int) or int(shape[-1]) <= 0:
        raise ValueError(f"teacher profile input_shape must be [batch, positive_size], got {shape}")
    return int(shape[-1])


def _parse_training_command(value: str) -> PolicyCommand:
    return _parse_training_command_spec(value).command or PolicyCommand()


def _parse_training_command_spec(value: str) -> ResidualTrainingCommand:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            f"training command must be VX,VY,YAW or VX,VY,YAW,WEIGHT, got {value!r}"
        )
    try:
        vx, vy, yaw = (float(part) for part in parts[:3])
        weight = float(parts[3]) if len(parts) == 4 else 1.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"training command must contain numeric VX,VY,YAW[,WEIGHT] values, got {value!r}"
        ) from exc
    if not np.all(np.isfinite([vx, vy, yaw, weight])):
        raise argparse.ArgumentTypeError(
            f"training command must contain finite VX,VY,YAW[,WEIGHT] values, got {value!r}"
        )
    if weight <= 0.0:
        raise argparse.ArgumentTypeError(
            f"training command weight must be positive when provided, got {value!r}"
        )
    return ResidualTrainingCommand(
        command=PolicyCommand(x_velocity=vx, y_velocity=vy, yaw_velocity=yaw),
        weight=weight,
    )


def _parse_training_sequence(value: str) -> ResidualTrainingSequence:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("training sequence must not be empty")
    weight = 1.0
    body = text
    if "|" in text:
        prefix, body = text.split("|", 1)
        try:
            weight = float(prefix.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"training sequence weight must be numeric in [WEIGHT|]VX,VY,YAW,STEPS;..., got {value!r}"
            ) from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise argparse.ArgumentTypeError(f"training sequence weight must be positive and finite, got {value!r}")

    segments: list[ResidualTrainingSegment] = []
    for raw_segment in body.split(";"):
        segment_text = raw_segment.strip()
        if not segment_text:
            continue
        parts = [part.strip() for part in segment_text.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"training sequence segments must be VX,VY,YAW,STEPS, got {segment_text!r} in {value!r}"
            )
        try:
            vx, vy, yaw = (float(part) for part in parts[:3])
            raw_steps = float(parts[3])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"training sequence must contain numeric VX,VY,YAW,STEPS values, got {value!r}"
            ) from exc
        if not np.all(np.isfinite([vx, vy, yaw, raw_steps])):
            raise argparse.ArgumentTypeError(
                f"training sequence must contain finite VX,VY,YAW,STEPS values, got {value!r}"
            )
        steps = int(raw_steps)
        if steps <= 0 or not math.isclose(raw_steps, float(steps)):
            raise argparse.ArgumentTypeError(
                f"training sequence steps must be a positive integer, got {parts[3]!r} in {value!r}"
            )
        segments.append(
            ResidualTrainingSegment(
                command=PolicyCommand(x_velocity=vx, y_velocity=vy, yaw_velocity=yaw),
                steps=steps,
            )
        )
    if not segments:
        raise argparse.ArgumentTypeError(f"training sequence must contain at least one segment, got {value!r}")
    return ResidualTrainingSequence(segments=tuple(segments), weight=weight)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a bounded residual policy on top of a teacher profile.")
    parser.add_argument("teacher_profile", nargs="?", default="open_duck_forward", help="Teacher policy profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for residual policy artifacts")
    parser.add_argument("--profile-name", default=None, help="Optional runtime profile name to write")
    parser.add_argument("--profile-output-dir", type=Path, default=Path("configs/policies"), help="Profile YAML output directory")
    parser.add_argument("--force-profile", action="store_true", help="Overwrite existing generated profile")
    parser.add_argument(
        "--actor-kind",
        choices=[
            RESIDUAL_ACTOR_CONSTANT,
            RESIDUAL_ACTOR_PHASE_CONTACT,
            RESIDUAL_ACTOR_COMMAND_STATE,
            RESIDUAL_ACTOR_COMMAND_STATE_MLP,
            RESIDUAL_ACTOR_CONTACT_PHASE_LIFT,
            RESIDUAL_ACTOR_CONTACT_PHASE_HARMONIC_LIFT,
            RESIDUAL_ACTOR_COMMAND_CONTACT_PHASE_LIFT,
        ],
        default=RESIDUAL_ACTOR_CONSTANT,
        help="Residual actor architecture",
    )
    parser.add_argument(
        "--training-command",
        action="append",
        default=[],
        metavar="VX,VY,YAW[,WEIGHT]",
        help=(
            "Repeat to score every residual candidate across multiple velocity commands. "
            "Optional WEIGHT emphasizes harder commands such as start/stop or turning."
        ),
    )
    parser.add_argument(
        "--training-sequence",
        action="append",
        default=[],
        metavar="[WEIGHT|]VX,VY,YAW,STEPS;...",
        help=(
            "Repeat to score a residual candidate on a command sequence in one reset, for example "
            "'2.5|0,0,0,50;0.06,0,0,100;0,0,0,50'."
        ),
    )
    parser.add_argument(
        "--worst-case-score-weight",
        type=float,
        default=0.0,
        help="Blend weighted mean candidate score with the worst scenario score; 0 disables, 1 uses only worst-case.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default=SCORE_NORMALIZATION_TOTAL,
        help=(
            "Score used for weighted/worst-case aggregation. 'total' preserves historical total reward; "
            "'per_step' divides each objective score by requested steps so shorter sequences do not dominate worst-case."
        ),
    )
    parser.add_argument("--steps-per-episode", type=int, default=300, help="Simulator steps per candidate residual episode")
    parser.add_argument("--iterations", type=int, default=5, help="CEM iterations")
    parser.add_argument("--population", type=int, default=16, help="Candidates per CEM iteration")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="Fraction of candidates used to update CEM mean")
    parser.add_argument("--initial-std", type=float, default=0.25, help="Initial residual search std")
    parser.add_argument("--min-std", type=float, default=0.01, help="Minimum residual search std")
    parser.add_argument("--std-decay", type=float, default=0.85, help="Std decay after elite update")
    parser.add_argument(
        "--no-zero-candidate",
        action="store_true",
        help="Do not force the current CEM mean into each generation; useful when probing beyond a warm-start checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--residual-scale", type=float, default=0.05, help="Runtime scale for residual output")
    parser.add_argument("--residual-clip-abs", type=float, default=1.0, help="Clip residual model output to ±this value")
    parser.add_argument("--final-action-clip-abs", type=float, default=0.0, help="Optional final action clip; 0 disables")
    parser.add_argument("--target-height", type=float, default=0.30, help="Nominal base height for reward shaping")
    parser.add_argument("--fall-height", type=float, default=0.14, help="Fall termination height")
    parser.add_argument("--min-upright", type=float, default=0.65, help="Fall termination upright score")
    parser.add_argument("--forward-velocity-sigma", type=float, default=0.20, help="Forward velocity tracking sigma")
    parser.add_argument("--swing-clearance-weight", type=float, default=0.0, help="Reward weight for reaching target swing-foot clearance")
    parser.add_argument("--low-clearance-penalty-weight", type=float, default=0.0, help="Penalty weight for swing-foot clearance below target")
    parser.add_argument("--target-swing-clearance", type=float, default=0.015, help="Target swing-foot world height in meters")
    parser.add_argument("--foot-contact-threshold", type=float, default=0.5, help="Contact value at or above which a foot is in stance")
    parser.add_argument(
        "--episodic-clearance-weight",
        type=float,
        default=0.0,
        help="Episode-level weight for median swing clearance divided by target clearance",
    )
    parser.add_argument(
        "--episodic-low-clearance-penalty-weight",
        type=float,
        default=0.0,
        help="Episode-level penalty weight for the fraction of swing samples below target clearance",
    )
    parser.add_argument(
        "--episodic-clearance-gap-weight",
        type=float,
        default=0.0,
        help="Episode-level penalty weight for the mean normalized swing-clearance shortfall below target",
    )
    parser.add_argument(
        "--episodic-clearance-quantile",
        type=float,
        default=0.25,
        help=(
            "Episode-level lower-tail swing-clearance quantile used by "
            "--episodic-clearance-quantile-gap-weight"
        ),
    )
    parser.add_argument(
        "--episodic-clearance-quantile-gap-weight",
        type=float,
        default=0.0,
        help="Episode-level penalty weight for the normalized target shortfall at the configured clearance quantile",
    )
    parser.add_argument(
        "--reference-low-clearance-ratio",
        type=float,
        action="append",
        default=[],
        help=(
            "Reference low-clearance ratio for one training objective; repeat in command/sequence order "
            "to penalize scenario regressions versus a retained profile."
        ),
    )
    parser.add_argument(
        "--low-clearance-regression-penalty-weight",
        type=float,
        default=0.0,
        help="Per-step penalty weight for exceeding the matching --reference-low-clearance-ratio.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Warm-start actor parameters from an existing residual_policy.pt",
    )
    parser.add_argument(
        "--final-score-breakdown",
        action="store_true",
        help="After training, re-score the best residual per command/sequence and write a diagnostic breakdown.",
    )
    parser.add_argument("--host", default=os.environ.get("SIM_HOST", "127.0.0.1"), help="Simulator API host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "5555")), help="Simulator API port")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without connecting to simulator")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    final_clip = args.final_action_clip_abs if args.final_action_clip_abs > 0 else None
    training_commands = [_parse_training_command_spec(value) for value in args.training_command]
    training_sequences = [_parse_training_sequence(value) for value in args.training_sequence]
    if not math.isfinite(float(args.worst_case_score_weight)) or not 0.0 <= float(args.worst_case_score_weight) <= 1.0:
        raise SystemExit("--worst-case-score-weight must be finite and in [0, 1]")
    if not math.isfinite(float(args.episodic_clearance_gap_weight)) or float(args.episodic_clearance_gap_weight) < 0.0:
        raise SystemExit("--episodic-clearance-gap-weight must be non-negative and finite")
    try:
        episodic_clearance_quantile = _validate_clearance_quantile(args.episodic_clearance_quantile)
    except ValueError as exc:
        raise SystemExit(f"--episodic-clearance-quantile invalid: {exc}") from exc
    if (
        not math.isfinite(float(args.episodic_clearance_quantile_gap_weight))
        or float(args.episodic_clearance_quantile_gap_weight) < 0.0
    ):
        raise SystemExit("--episodic-clearance-quantile-gap-weight must be non-negative and finite")
    objective_count = len(training_commands) + len(training_sequences)
    if objective_count == 0:
        objective_count = 1
    try:
        reference_low_clearance_ratios = _normalize_reference_low_clearance_ratios(
            args.reference_low_clearance_ratio,
            objective_count=objective_count,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        not math.isfinite(float(args.low_clearance_regression_penalty_weight))
        or float(args.low_clearance_regression_penalty_weight) < 0.0
    ):
        raise SystemExit("--low-clearance-regression-penalty-weight must be non-negative and finite")
    opt_cfg = ResidualOptimizationConfig(
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        initial_std=args.initial_std,
        min_std=args.min_std,
        std_decay=args.std_decay,
        seed=args.seed,
        residual_clip_abs=args.residual_clip_abs,
        include_zero_candidate=not bool(args.no_zero_candidate),
    )
    reward_cfg = WalkingRewardConfig(
        target_height=args.target_height,
        fall_height=args.fall_height,
        min_upright=args.min_upright,
        forward_velocity_sigma=args.forward_velocity_sigma,
        swing_clearance_weight=args.swing_clearance_weight,
        low_clearance_penalty_weight=args.low_clearance_penalty_weight,
        target_swing_clearance=args.target_swing_clearance,
        foot_contact_threshold=args.foot_contact_threshold,
    )
    if args.dry_run:
        print(json.dumps({
            "teacher_profile": args.teacher_profile,
            "output_dir": str(args.output_dir),
            "profile_name": args.profile_name,
            "actor_kind": args.actor_kind,
            "training_commands": [command.describe() for command in training_commands],
            "training_sequences": [sequence.describe() for sequence in training_sequences],
            "worst_case_score_weight": args.worst_case_score_weight,
            "score_normalization": args.score_normalization,
            "final_score_breakdown_requested": bool(args.final_score_breakdown),
            "steps_per_episode": args.steps_per_episode,
            "optimization_config": asdict(opt_cfg),
            "reward_config": asdict(reward_cfg),
            "episodic_clearance_weight": args.episodic_clearance_weight,
            "episodic_low_clearance_penalty_weight": args.episodic_low_clearance_penalty_weight,
            "episodic_clearance_gap_weight": args.episodic_clearance_gap_weight,
            "episodic_clearance_quantile": episodic_clearance_quantile,
            "episodic_clearance_quantile_gap_weight": args.episodic_clearance_quantile_gap_weight,
            "reference_low_clearance_ratios": [
                value for value in reference_low_clearance_ratios if value is not None
            ],
            "low_clearance_regression_penalty_weight": args.low_clearance_regression_penalty_weight,
            "initial_checkpoint": None if args.initial_checkpoint is None else str(args.initial_checkpoint),
            "residual_scale": args.residual_scale,
            "residual_clip_abs": args.residual_clip_abs,
            "final_action_clip_abs": final_clip,
        }, indent=2, sort_keys=True))
        return
    result = train_residual_policy(
        teacher_profile=args.teacher_profile,
        output_dir=args.output_dir,
        steps_per_episode=args.steps_per_episode,
        optimization_config=opt_cfg,
        residual_scale=args.residual_scale,
        residual_clip_abs=args.residual_clip_abs,
        final_action_clip_abs=final_clip,
        reward_config=reward_cfg,
        profile_name=args.profile_name,
        profile_output_dir=args.profile_output_dir,
        force_profile=args.force_profile,
        host=args.host,
        port=args.port,
        actor_kind=args.actor_kind,
        training_commands=training_commands,
        training_sequences=training_sequences,
        episodic_clearance_weight=args.episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=args.episodic_low_clearance_penalty_weight,
        episodic_clearance_gap_weight=args.episodic_clearance_gap_weight,
        episodic_clearance_quantile=episodic_clearance_quantile,
        episodic_clearance_quantile_gap_weight=args.episodic_clearance_quantile_gap_weight,
        reference_low_clearance_ratios=reference_low_clearance_ratios
        if args.reference_low_clearance_ratio
        else None,
        low_clearance_regression_penalty_weight=args.low_clearance_regression_penalty_weight,
        worst_case_score_weight=args.worst_case_score_weight,
        score_normalization=args.score_normalization,
        initial_checkpoint=args.initial_checkpoint,
        final_score_breakdown=bool(args.final_score_breakdown),
    )
    print("Soridormi residual policy fine-tuning")
    print("======================================")
    print(f"Teacher profile: {result.teacher_profile}")
    print(f"Output: {result.output_dir}")
    print(f"ONNX: {result.residual_onnx_path}")
    print(f"Profile: {result.profile_path or 'n/a'}")
    if result.optimization:
        print(f"Best score: {result.optimization['best_score']:.6g}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
