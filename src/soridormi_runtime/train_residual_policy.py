from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from soridormi_runtime.create_policy_profile import build_replacement_profile_payload
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.rl_finetune_env import ACTION_SIZE, ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.walking_reward import WalkingRewardConfig


DEFAULT_OUTPUT_ROOT = Path("/data/rl_finetune/residual_policy")
DEFAULT_RESIDUAL_ONNX_NAME = "residual_policy.onnx"
DEFAULT_RESIDUAL_PT_NAME = "residual_policy.pt"


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
    cfg = config or ResidualOptimizationConfig()
    rng = np.random.default_rng(int(cfg.seed))
    mean = np.zeros(ACTION_SIZE, dtype=np.float32)
    std = np.full(ACTION_SIZE, float(cfg.initial_std), dtype=np.float32)
    elite_count = max(1, int(math.ceil(float(cfg.population) * float(cfg.elite_fraction))))
    best_residual = mean.copy()
    best_score = float("-inf")
    history: list[dict[str, Any]] = []

    for iteration in range(max(1, int(cfg.iterations))):
        candidates = rng.normal(mean, std, size=(max(1, int(cfg.population)), ACTION_SIZE)).astype(np.float32)
        candidates = np.clip(candidates, -float(cfg.residual_clip_abs), float(cfg.residual_clip_abs))
        if cfg.include_zero_candidate:
            candidates[0, :] = 0.0
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
) -> float:
    env = RlFineTuneEnv(
        profile=teacher_profile,
        host=host,
        port=port,
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
    env.reset()
    for _ in range(max(1, int(steps))):
        step = env.step(residual)
        total += float(step.metrics.get("reward", 0.0))
        completed += 1
        if bool(step.metrics.get("terminated", False)):
            break
    # Prefer policies that survive longer when total reward ties.
    return float(total + 0.001 * completed)


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
            ),
            config=optimization_config,
        )
        export_constant_residual_policy(
            optimization.best_residual,
            output_onnx=onnx_path,
            output_checkpoint=checkpoint_path,
            input_size=input_size,
        )
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
            )
    except Exception as exc:  # pragma: no cover - live simulator/training environment
        errors.append(repr(exc))

    payload = {
        "schema_version": 1,
        "teacher_profile": teacher_profile,
        "policy_input_size": input_size,
        "output_dir": str(output_dir),
        "steps_per_episode": int(steps_per_episode),
        "residual_scale": float(residual_scale),
        "residual_clip_abs": float(residual_clip_abs),
        "final_action_clip_abs": final_action_clip_abs,
        "reward_config": asdict(reward_config),
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
        f"Output directory: `{payload['output_dir']}`",
        f"Residual scale: `{payload['residual_scale']}`",
        f"Policy input size: `{payload.get('policy_input_size')}`",
        "",
    ]
    optimization = payload.get("optimization")
    if optimization:
        lines.extend(
            [
                f"Best score: `{optimization['best_score']:.6g}`",
                f"Best residual abs max: `{max(abs(float(x)) for x in optimization['best_residual']):.6g}`",
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a bounded residual policy on top of a teacher profile.")
    parser.add_argument("teacher_profile", nargs="?", default="open_duck_forward", help="Teacher policy profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for residual policy artifacts")
    parser.add_argument("--profile-name", default=None, help="Optional runtime profile name to write")
    parser.add_argument("--profile-output-dir", type=Path, default=Path("configs/policies"), help="Profile YAML output directory")
    parser.add_argument("--force-profile", action="store_true", help="Overwrite existing generated profile")
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
    parser.add_argument("--host", default=os.environ.get("SIM_HOST", "127.0.0.1"), help="Simulator API host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "5555")), help="Simulator API port")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without connecting to simulator")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    final_clip = args.final_action_clip_abs if args.final_action_clip_abs > 0 else None
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
            "steps_per_episode": args.steps_per_episode,
            "optimization_config": asdict(opt_cfg),
            "reward_config": asdict(reward_cfg),
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
