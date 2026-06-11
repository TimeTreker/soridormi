from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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
RESIDUAL_ACTOR_CONSTANT = "constant"
RESIDUAL_ACTOR_PHASE_CONTACT = "phase_contact"
RESIDUAL_ACTOR_COMMAND_STATE = "command_state"
RESIDUAL_ACTOR_COMMAND_STATE_MLP = "command_state_mlp"
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
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
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
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
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
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
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
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
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
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
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
        episodic_clearance_weight=episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
    )


def evaluate_command_state_mlp_residual_live(
    parameters: np.ndarray,
    **kwargs: Any,
) -> float:
    actor = lambda observation: command_state_mlp_residual_action(observation, parameters)
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
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
) -> float:
    scores: list[float] = []
    weights: list[float] = []
    commands = _normalize_training_commands(training_commands)
    for command_spec in commands:
        env = RlFineTuneEnv(
            profile=teacher_profile,
            host=host,
            port=port,
            command=command_spec.command,
            residual_config=ResidualActionConfig(
                residual_scale=residual_scale,
                residual_clip_abs=residual_clip_abs,
                final_action_clip_abs=final_action_clip_abs,
            ),
            reward_config=reward_config,
            reset_on_start=True,
        )
        total = 0.0
        completed = 0
        swing_clearances: list[float] = []
        env.reset()
        for _ in range(max(1, int(steps))):
            step = env.step(residual_source)
            total += float(step.metrics.get("reward", 0.0))
            completed += 1
            diagnostics = step.metrics.get("reward_diagnostics", {})
            clearance = diagnostics.get("swing_clearance_m") if isinstance(diagnostics, dict) else None
            if clearance is not None and math.isfinite(float(clearance)):
                swing_clearances.append(float(clearance))
            if bool(step.metrics.get("terminated", False)):
                break
        total += completed * episodic_clearance_adjustment(
            swing_clearances,
            target_clearance=reward_config.target_swing_clearance,
            clearance_weight=episodic_clearance_weight,
            low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
        )
        # Prefer policies that survive longer when total reward ties.
        scores.append(float(total + 0.001 * completed))
        weights.append(float(command_spec.weight))
    return float(
        np.average(
            np.asarray(scores, dtype=np.float64),
            weights=np.asarray(weights, dtype=np.float64),
        )
    )


def episodic_clearance_adjustment(
    swing_clearances: Sequence[float],
    *,
    target_clearance: float,
    clearance_weight: float,
    low_clearance_penalty_weight: float,
) -> float:
    if not swing_clearances:
        return 0.0
    target = max(float(target_clearance), 1e-9)
    values = np.asarray(swing_clearances, dtype=np.float64)
    median_ratio = float(np.median(values)) / target
    low_ratio = float(np.mean(values < target))
    return float(clearance_weight) * median_ratio - float(low_clearance_penalty_weight) * low_ratio


def _normalize_training_commands(
    training_commands: Sequence[PolicyCommand | ResidualTrainingCommand] | None,
) -> list[ResidualTrainingCommand]:
    if not training_commands:
        return [ResidualTrainingCommand(command=None, weight=1.0)]
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
    }[actor_kind]
    return source.reshape(expected)


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
    payload.setdefault("metadata", {})["generated_by"] = "soridormi_m619_residual_rl"
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
    episodic_clearance_weight: float = 0.0,
    episodic_low_clearance_penalty_weight: float = 0.0,
    initial_checkpoint: Path | None = None,
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

    try:
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
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
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
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
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
                    episodic_clearance_weight=episodic_clearance_weight,
                    episodic_low_clearance_penalty_weight=episodic_low_clearance_penalty_weight,
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
    except Exception as exc:  # pragma: no cover - live simulator/training environment
        errors.append(repr(exc))

    payload = {
        "schema_version": 1,
        "teacher_profile": teacher_profile,
        "actor_kind": actor_kind,
        "training_commands": []
        if not training_commands
        else [command.describe() for command in _normalize_training_commands(training_commands)],
        "policy_input_size": input_size,
        "output_dir": str(output_dir),
        "steps_per_episode": int(steps_per_episode),
        "residual_scale": float(residual_scale),
        "residual_clip_abs": float(residual_clip_abs),
        "final_action_clip_abs": final_action_clip_abs,
        "reward_config": asdict(reward_config),
        "episodic_clearance_weight": float(episodic_clearance_weight),
        "episodic_low_clearance_penalty_weight": float(episodic_low_clearance_penalty_weight),
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
    if payload.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for error in payload["errors"]:
            lines.append(f"- `{error}`")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    parser.add_argument("--steps-per-episode", type=int, default=300, help="Simulator steps per candidate residual episode")
    parser.add_argument("--iterations", type=int, default=5, help="CEM iterations")
    parser.add_argument("--population", type=int, default=16, help="Candidates per CEM iteration")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="Fraction of candidates used to update CEM mean")
    parser.add_argument("--initial-std", type=float, default=0.25, help="Initial residual search std")
    parser.add_argument("--min-std", type=float, default=0.01, help="Minimum residual search std")
    parser.add_argument("--std-decay", type=float, default=0.85, help="Std decay after elite update")
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
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Warm-start actor parameters from an existing residual_policy.pt",
    )
    parser.add_argument("--host", default=os.environ.get("SIM_HOST", "127.0.0.1"), help="Simulator API host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "5555")), help="Simulator API port")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without connecting to simulator")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    final_clip = args.final_action_clip_abs if args.final_action_clip_abs > 0 else None
    training_commands = [_parse_training_command_spec(value) for value in args.training_command]
    opt_cfg = ResidualOptimizationConfig(
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        initial_std=args.initial_std,
        min_std=args.min_std,
        std_decay=args.std_decay,
        seed=args.seed,
        residual_clip_abs=args.residual_clip_abs,
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
            "steps_per_episode": args.steps_per_episode,
            "optimization_config": asdict(opt_cfg),
            "reward_config": asdict(reward_cfg),
            "episodic_clearance_weight": args.episodic_clearance_weight,
            "episodic_low_clearance_penalty_weight": args.episodic_low_clearance_penalty_weight,
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
        episodic_clearance_weight=args.episodic_clearance_weight,
        episodic_low_clearance_penalty_weight=args.episodic_low_clearance_penalty_weight,
        initial_checkpoint=args.initial_checkpoint,
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
