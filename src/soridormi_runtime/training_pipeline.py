"""End-to-end supervised training pipeline planner/runner for Soridormi policies.

M6.7 intentionally stays at the workflow layer.  It orchestrates the already
validated M6 data/export/train/evaluate tools without changing policy runtime
semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

DEFAULT_TEMPLATE_PROFILE = "open_duck_forward"
DEFAULT_OUTPUT_ROOT = Path("data/training_pipelines")


@dataclass(frozen=True)
class PipelineStep:
    """One executable shell step in the training pipeline."""

    name: str
    description: str
    command: list[str]
    outputs: list[str] = field(default_factory=list)

    @property
    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shell"] = self.shell
        return payload


@dataclass(frozen=True)
class PipelinePlan:
    """A deterministic end-to-end behavior-cloning pipeline plan."""

    candidate_profile: str
    template_profile: str
    output_root: str
    dataset_jsonl: str
    prepared_dir: str
    training_run_dir: str
    evaluation_dir: str
    logs: list[str]
    steps: list[PipelineStep]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pipeline_kind": "linear_behavior_clone",
            "candidate_profile": self.candidate_profile,
            "template_profile": self.template_profile,
            "output_root": self.output_root,
            "artifacts": {
                "dataset_jsonl": self.dataset_jsonl,
                "prepared_dir": self.prepared_dir,
                "training_run_dir": self.training_run_dir,
                "evaluation_dir": self.evaluation_dir,
            },
            "logs": list(self.logs),
            "steps": [step.to_dict() for step in self.steps],
        }

    def markdown(self) -> str:
        lines = [
            f"# Soridormi training pipeline: `{self.candidate_profile}`",
            "",
            f"Template profile: `{self.template_profile}`",
            f"Output root: `{self.output_root}`",
            "",
            "## Artifacts",
            "",
            f"- Dataset JSONL: `{self.dataset_jsonl}`",
            f"- Prepared dataset: `{self.prepared_dir}`",
            f"- Training run: `{self.training_run_dir}`",
            f"- Evaluation: `{self.evaluation_dir}`",
            "",
            "## Steps",
            "",
        ]
        for index, step in enumerate(self.steps, start=1):
            lines.extend(
                [
                    f"### {index}. {step.name}",
                    "",
                    step.description,
                    "",
                    "```bash",
                    step.shell,
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class PipelineRunResult:
    ok: bool
    plan: PipelinePlan
    plan_path: str | None
    report_path: str | None
    completed_steps: list[str]
    failed_step: str | None = None
    returncode: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "plan_path": self.plan_path,
            "report_path": self.report_path,
            "completed_steps": list(self.completed_steps),
            "failed_step": self.failed_step,
            "returncode": self.returncode,
            "plan": self.plan.to_dict(),
        }


def _clean_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name.strip())
    if not cleaned:
        raise ValueError("candidate profile name must not be empty")
    return cleaned


def build_training_pipeline_plan(
    *,
    candidate_profile: str,
    logs: Sequence[str],
    template_profile: str = DEFAULT_TEMPLATE_PROFILE,
    output_root: str | Path | None = None,
    seed: int = 123,
    ridge_lambda: float = 1e-4,
    max_test_mae: float | None = None,
    max_test_rmse: float | None = None,
    max_test_max_abs_error: float | None = None,
    require_provider: Sequence[str] | None = None,
    include_model: bool = False,
    check_model: bool = False,
    require_model: bool = False,
    write_predictions: bool = False,
    force_profile: bool = False,
) -> PipelinePlan:
    """Build an executable host-side training pipeline plan.

    The generated commands call Soridormi host wrapper scripts. Those scripts are
    responsible for Docker/container path translation, so the plan should keep
    host-style paths such as ``data/...``.
    """

    candidate = _clean_name(candidate_profile)
    log_list = [str(item) for item in logs]
    if not log_list:
        raise ValueError("at least one source log is required")

    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT / candidate
    dataset_dir = root / "dataset"
    prepared_dir = root / "prepared"
    training_run_dir = root / "linear_bc"
    evaluation_dir = root / "evaluation"
    dataset_jsonl = dataset_dir / "supervised.jsonl"
    model_npz = training_run_dir / "linear_behavior_clone.npz"

    steps: list[PipelineStep] = []
    steps.append(
        PipelineStep(
            name="export_dataset",
            description="Export policy observations/actions from runtime logs into supervised JSONL.",
            command=[
                "./scripts/export_training_dataset.sh",
                *log_list,
                "--output",
                str(dataset_jsonl),
            ],
            outputs=[str(dataset_jsonl), str(dataset_dir / "manifest.json")],
        )
    )
    steps.append(
        PipelineStep(
            name="prepare_dataset",
            description="Validate and split the supervised dataset deterministically.",
            command=[
                "./scripts/prepare_training_dataset.sh",
                str(dataset_jsonl),
                "--output-dir",
                str(prepared_dir),
                "--seed",
                str(seed),
            ],
            outputs=[str(prepared_dir / "prepared_manifest.json")],
        )
    )
    steps.append(
        PipelineStep(
            name="summarize_dataset",
            description="Write dataset statistics and normalization artifacts from the train split.",
            command=[
                "./scripts/summarize_training_dataset.sh",
                str(prepared_dir),
            ],
            outputs=[str(prepared_dir / "dataset_stats.json"), str(prepared_dir / "normalization.json")],
        )
    )
    steps.append(
        PipelineStep(
            name="train_linear_bc",
            description="Train the deterministic NumPy linear behavior-cloning baseline.",
            command=[
                "./scripts/train_behavior_clone.sh",
                str(prepared_dir),
                "--output-dir",
                str(training_run_dir),
                "--ridge-lambda",
                f"{ridge_lambda:g}",
            ],
            outputs=[str(model_npz), str(training_run_dir / "train_metrics.json")],
        )
    )

    create_profile_cmd = [
        "./scripts/create_linear_bc_profile.sh",
        candidate,
        "--model",
        str(model_npz),
        "--template",
        template_profile,
        "--description",
        f"Linear BC candidate trained by M6.7 pipeline from {template_profile}",
    ]
    if force_profile:
        create_profile_cmd.append("--force")
    steps.append(
        PipelineStep(
            name="create_profile",
            description="Create a runtime policy profile for the trained linear BC artifact.",
            command=create_profile_cmd,
            outputs=[f"configs/policies/{candidate}.yaml"],
        )
    )

    eval_cmd = [
        "./scripts/evaluate_policy_profile.sh",
        candidate,
        str(prepared_dir),
        "--output-dir",
        str(evaluation_dir),
    ]
    if write_predictions:
        eval_cmd.append("--write-predictions")
    if max_test_mae is not None:
        eval_cmd.extend(["--max-test-mae", f"{max_test_mae:g}"])
    if max_test_rmse is not None:
        eval_cmd.extend(["--max-test-rmse", f"{max_test_rmse:g}"])
    if max_test_max_abs_error is not None:
        eval_cmd.extend(["--max-test-max-abs-error", f"{max_test_max_abs_error:g}"])
    for provider in require_provider or []:
        eval_cmd.extend(["--require-provider", provider])
    steps.append(
        PipelineStep(
            name="evaluate_profile",
            description="Evaluate the generated runtime policy profile against the prepared dataset.",
            command=eval_cmd,
            outputs=[str(evaluation_dir / "evaluation.json"), str(evaluation_dir / "evaluation_report.md")],
        )
    )

    accept_cmd = ["./scripts/accept_policy_profile.sh", candidate]
    if check_model:
        accept_cmd.append("--check-model")
    if require_model:
        accept_cmd.append("--require-model")
    for provider in require_provider or []:
        accept_cmd.extend(["--require-provider", provider])
    steps.append(
        PipelineStep(
            name="accept_profile",
            description="Build Soridormi acceptance artifacts for the generated profile.",
            command=accept_cmd,
            outputs=[f"data/policy_acceptance/{candidate}/acceptance.json"],
        )
    )

    package_cmd = ["./scripts/package_policy_profile.sh", candidate]
    if include_model:
        package_cmd.append("--include-model")
    if check_model:
        package_cmd.append("--check-model")
    if require_model:
        package_cmd.append("--require-model")
    for provider in require_provider or []:
        package_cmd.extend(["--require-provider", provider])
    steps.append(
        PipelineStep(
            name="package_profile",
            description="Create a handoff package for the generated candidate profile.",
            command=package_cmd,
            outputs=[f"data/policy_packages/{candidate}_*.policy.tar.gz"],
        )
    )

    return PipelinePlan(
        candidate_profile=candidate,
        template_profile=template_profile,
        output_root=str(root),
        dataset_jsonl=str(dataset_jsonl),
        prepared_dir=str(prepared_dir),
        training_run_dir=str(training_run_dir),
        evaluation_dir=str(evaluation_dir),
        logs=log_list,
        steps=steps,
    )


def write_pipeline_plan(plan: PipelinePlan, output_root: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(output_root or plan.output_root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "pipeline_plan.json"
    report_path = root / "pipeline_plan.md"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(plan.markdown() + "\n", encoding="utf-8")
    return plan_path, report_path


def run_training_pipeline(plan: PipelinePlan, *, dry_run: bool = False, env: dict[str, str] | None = None) -> PipelineRunResult:
    plan_path, report_path = write_pipeline_plan(plan)
    completed: list[str] = []
    if dry_run:
        return PipelineRunResult(
            ok=True,
            plan=plan,
            plan_path=str(plan_path),
            report_path=str(report_path),
            completed_steps=completed,
        )

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    for step in plan.steps:
        print(f"\n==> {step.name}: {step.description}", flush=True)
        print(step.shell, flush=True)
        proc = subprocess.run(step.command, env=run_env, text=True)
        if proc.returncode != 0:
            return PipelineRunResult(
                ok=False,
                plan=plan,
                plan_path=str(plan_path),
                report_path=str(report_path),
                completed_steps=completed,
                failed_step=step.name,
                returncode=proc.returncode,
            )
        completed.append(step.name)
    return PipelineRunResult(
        ok=True,
        plan=plan,
        plan_path=str(plan_path),
        report_path=str(report_path),
        completed_steps=completed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or print the M6.7 end-to-end linear BC training pipeline")
    parser.add_argument("candidate_profile", help="Name for the generated candidate policy profile")
    parser.add_argument("logs", nargs="+", help="Runtime logs to export, typically data/logs/policy_*.mcap")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE_PROFILE, help="Template profile to clone")
    parser.add_argument("--output-root", default=None, help="Pipeline output root directory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--max-test-mae", type=float, default=None)
    parser.add_argument("--max-test-rmse", type=float, default=None)
    parser.add_argument("--max-test-max-abs-error", type=float, default=None)
    parser.add_argument("--require-provider", action="append", default=None)
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--force-profile", action="store_true", help="Overwrite an existing generated profile")
    parser.add_argument("--dry-run", action="store_true", help="Write plan artifacts but do not execute steps")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = build_training_pipeline_plan(
            candidate_profile=args.candidate_profile,
            logs=args.logs,
            template_profile=args.template,
            output_root=args.output_root,
            seed=args.seed,
            ridge_lambda=args.ridge_lambda,
            max_test_mae=args.max_test_mae,
            max_test_rmse=args.max_test_rmse,
            max_test_max_abs_error=args.max_test_max_abs_error,
            require_provider=args.require_provider,
            include_model=args.include_model,
            check_model=args.check_model,
            require_model=args.require_model,
            write_predictions=args.write_predictions,
            force_profile=args.force_profile,
        )
        result = run_training_pipeline(plan, dry_run=args.dry_run)
    except Exception as exc:  # pragma: no cover - CLI guard
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Soridormi M6.7 training pipeline")
        print("================================")
        print(f"Candidate profile: {plan.candidate_profile}")
        print(f"Output root: {plan.output_root}")
        print(f"Plan: {result.plan_path}")
        print(f"Report: {result.report_path}")
        if args.dry_run:
            print("Dry run: yes")
            for index, step in enumerate(plan.steps, start=1):
                print(f"{index}. {step.name}: {step.shell}")
        elif result.ok:
            print("Result: OK")
        else:
            print(f"Result: FAILED at {result.failed_step} rc={result.returncode}")
    return 0 if result.ok else result.returncode or 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
