from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from soridormi_runtime.evaluate_policy_profile import evaluate_policy_profile
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.policy_relabel import merge_supervised_datasets, relabel_policy_rollouts_with_teacher
from soridormi_runtime.train_neural_behavior_clone import train_neural_behavior_clone
from soridormi_runtime.training_dataset_prepare import split_training_dataset
from soridormi_runtime.training_dataset_stats import analyze_prepared_training_dataset

ITERATION_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("/data/policy_iterations")


@dataclass
class PolicyIterationResult:
    ok: bool
    iteration_name: str
    output_dir: str
    teacher_profile: str
    relabel_dataset_path: str
    merged_dataset_path: str
    prepared_manifest_path: str
    stats_path: str
    training_output_dir: str
    trained_profile_path: str | None
    evaluation_path: str
    promoted_profile_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, result: PolicyIterationResult) -> None:
    lines = [
        "# Soridormi Policy Iteration Report",
        "",
        f"- Iteration: `{result.iteration_name}`",
        f"- Teacher profile: `{result.teacher_profile}`",
        f"- Relabel dataset: `{result.relabel_dataset_path}`",
        f"- Merged dataset: `{result.merged_dataset_path}`",
        f"- Prepared manifest: `{result.prepared_manifest_path}`",
        f"- Training output: `{result.training_output_dir}`",
        f"- Trained profile: `{result.trained_profile_path or 'n/a'}`",
        f"- Evaluation: `{result.evaluation_path}`",
        f"- Promoted profile: `{result.promoted_profile_path or 'n/a'}`",
        f"- Result: **{'OK' if result.ok else 'FAILED'}**",
    ]
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Profile YAML must be a mapping: {path}")
    return payload


def promote_trained_profile(
    source_profile_path: str | Path,
    *,
    target_profile_name: str,
    output_dir: str | Path = "configs/policies",
    force: bool = False,
) -> Path:
    """Copy a successful iteration profile into a stable promoted profile name."""
    source = Path(source_profile_path)
    if not source.exists():
        raise FileNotFoundError(source)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{target_profile_name}.yaml"
    if target.exists() and not force:
        raise FileExistsError(f"Promoted profile already exists: {target}; pass --force-promote to overwrite")
    payload = _load_yaml(source)
    payload["name"] = target_profile_name
    description = str(payload.get("description") or "")
    suffix = f"Promoted from iteration profile {source.stem}."
    payload["description"] = f"{description}\n{suffix}".strip()
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def run_policy_iteration_from_rollouts(
    *,
    iteration_name: str,
    candidate_logs: Iterable[str | Path],
    base_datasets: Iterable[str | Path],
    teacher_profile: str | Path | PolicyProfile = "open_duck_forward",
    output_root: str | Path | None = None,
    profile_template: str = "open_duck_forward",
    seed: int = 123,
    epochs: int = 50,
    hidden_sizes: str = "256,256",
    device: str = "auto",
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_samples: int | None = None,
    max_test_mae: float | None = None,
    max_test_rmse: float | None = None,
    require_provider: list[str] | str | None = None,
    promote_to: str | None = None,
    profile_force: bool = False,
    force_promote: bool = False,
    promote_output_dir: str | Path = "configs/policies",
) -> PolicyIterationResult:
    output = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT / f"{iteration_name}_{utc_stamp()}"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "policy_iteration_report.md"
    result_path = output / "policy_iteration.json"

    teacher = teacher_profile if isinstance(teacher_profile, PolicyProfile) else PolicyProfile.load(teacher_profile)
    errors: list[str] = []
    warnings: list[str] = []

    relabel_path = output / "relabel" / "teacher_relabel.jsonl"
    relabel = relabel_policy_rollouts_with_teacher(
        candidate_logs,
        teacher_profile=teacher,
        output_path=relabel_path,
        max_samples=max_samples,
        require_providers=require_provider,
    )
    errors.extend(f"relabel: {error}" for error in relabel.errors)
    warnings.extend(f"relabel: {warning}" for warning in relabel.warnings)

    merged_path = output / "dataset" / "combined_supervised.jsonl"
    merge = merge_supervised_datasets(
        [*base_datasets, relabel.output_path],
        output_path=merged_path,
    )
    errors.extend(f"merge: {error}" for error in merge.errors)
    warnings.extend(f"merge: {warning}" for warning in merge.warnings)

    prepared_dir = output / "prepared"
    prepared = split_training_dataset(merged_path, output_dir=prepared_dir, seed=seed)
    errors.extend(f"prepare: {error}" for error in prepared.errors)
    warnings.extend(f"prepare: {warning}" for warning in prepared.warnings)

    stats = analyze_prepared_training_dataset(prepared.manifest_path, output_dir=prepared_dir)
    errors.extend(f"stats: {error}" for error in stats.errors)
    warnings.extend(f"stats: {warning}" for warning in stats.warnings)

    train_dir = output / "train"
    train = train_neural_behavior_clone(
        prepared.manifest_path,
        output_dir=train_dir,
        normalization_path=stats.normalization_path,
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        export_onnx=True,
        create_profile=True,
        profile_name=iteration_name,
        profile_template=profile_template,
        profile_force=profile_force,
    )
    errors.extend(f"train: {error}" for error in train.errors)
    warnings.extend(f"train: {warning}" for warning in train.warnings)

    evaluation = evaluate_policy_profile(
        iteration_name,
        prepared.manifest_path,
        output_dir=output / "evaluation",
        require_providers=require_provider,
        max_test_mae=max_test_mae,
        max_test_rmse=max_test_rmse,
    )
    errors.extend(f"evaluate: {error}" for error in evaluation.errors)
    warnings.extend(f"evaluate: {warning}" for warning in evaluation.warnings)

    promoted_path: Path | None = None
    if promote_to:
        if errors:
            warnings.append(f"promotion skipped because iteration failed: {promote_to}")
        elif train.profile_path is None:
            errors.append("promotion requested but training did not create a profile")
        else:
            try:
                promoted_path = promote_trained_profile(
                    train.profile_path,
                    target_profile_name=promote_to,
                    force=force_promote,
                    output_dir=promote_output_dir,
                )
            except Exception as exc:
                errors.append(f"promote: {exc!r}")

    result = PolicyIterationResult(
        ok=not errors,
        iteration_name=iteration_name,
        output_dir=str(output),
        teacher_profile=teacher.name,
        relabel_dataset_path=relabel.output_path,
        merged_dataset_path=merge.output_path,
        prepared_manifest_path=prepared.manifest_path,
        stats_path=stats.stats_path,
        training_output_dir=train.output_dir,
        trained_profile_path=train.profile_path,
        evaluation_path=evaluation.evaluation_path,
        promoted_profile_path=str(promoted_path) if promoted_path is not None else None,
        errors=errors,
        warnings=warnings,
    )
    _write_json(result_path, {"schema_version": ITERATION_SCHEMA_VERSION, "iteration_type": "soridormi.dagger_neural_bc_iteration.v1", **asdict(result)})
    _write_report(report_path, result)
    return result


def print_iteration_summary(result: PolicyIterationResult) -> None:
    print("Soridormi policy iteration")
    print("==========================")
    print(f"Iteration: {result.iteration_name}")
    print(f"Output: {result.output_dir}")
    print(f"Relabel dataset: {result.relabel_dataset_path}")
    print(f"Merged dataset: {result.merged_dataset_path}")
    print(f"Prepared manifest: {result.prepared_manifest_path}")
    print(f"Training output: {result.training_output_dir}")
    print(f"Evaluation: {result.evaluation_path}")
    if result.promoted_profile_path:
        print(f"Promoted profile: {result.promoted_profile_path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:40]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="iterative retraining and promotion DAgger-style retrain/evaluate/promote iteration from candidate rollouts.")
    parser.add_argument("iteration_name", help="Name for the trained iteration profile")
    parser.add_argument("--candidate-log", action="append", required=True, help="Candidate rollout log to relabel; repeatable")
    parser.add_argument("--base-dataset", action="append", required=True, help="Existing supervised JSONL dataset; repeatable")
    parser.add_argument("--teacher-profile", default="open_duck_forward")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profile-template", default="open_duck_forward")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-test-mae", type=float, default=None)
    parser.add_argument("--max-test-rmse", type=float, default=None)
    parser.add_argument("--require-provider", action="append", default=None)
    parser.add_argument("--promote-to", default=None, help="Stable profile name to write if the iteration passes")
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--force-promote", action="store_true")
    parser.add_argument("--promote-output-dir", type=Path, default=Path("configs/policies"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_policy_iteration_from_rollouts(
        iteration_name=args.iteration_name,
        candidate_logs=args.candidate_log,
        base_datasets=args.base_dataset,
        teacher_profile=args.teacher_profile,
        output_root=args.output_root,
        profile_template=args.profile_template,
        seed=args.seed,
        epochs=args.epochs,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_samples=args.max_samples,
        max_test_mae=args.max_test_mae,
        max_test_rmse=args.max_test_rmse,
        require_provider=args.require_provider,
        promote_to=args.promote_to,
        profile_force=args.force_profile,
        force_promote=args.force_promote,
        promote_output_dir=args.promote_output_dir,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_iteration_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
