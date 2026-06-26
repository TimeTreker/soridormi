from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.m10_clearance_readiness import (
    DEFAULT_REQUIRED_SCENARIOS,
    _as_float,
    _format_value,
    _mapping,
    build_m10_clearance_readiness,
)
from soridormi_runtime.scenario_curriculum import DEFAULT_SCENARIO_MANIFEST

DEFAULT_SCENARIO_EVAL_ROOT = Path("artifacts/scenario_eval")
DEFAULT_OUTPUT_DIR = Path("artifacts/clearance_history")
DEFAULT_REFERENCE_PROFILE = "clearance_liftscale_stack_s143_step090_offset005"
DEFAULT_REPORT_NAME = "scenario_rollout_report.json"


@dataclass(frozen=True)
class M10ClearanceHistoryCandidate:
    profile: str
    suite_dir: str
    ok: bool
    gate_status: str
    retention_status: str
    candidate_beats_reference: bool | None
    passed_count: int
    failed_count: int
    missing_count: int
    summary_metrics: dict[str, Any]
    reference_comparison: dict[str, Any] | None = None
    scenario_comparisons: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M10ClearanceHistoryReport:
    ok: bool
    reference_profile: str
    reference_suite_dir: str | None
    scenario_eval_root: str
    scenario_manifest: str
    scenario_count: int
    candidate_count: int
    ready_count: int
    reference_beating_blocked_count: int
    rejected_count: int
    missing_count: int
    retained_profile: str | None
    best_candidate_profile: str | None
    candidates: list[dict[str, Any]]
    recommendations: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_csv(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            cleaned = part.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _discover_suite_dirs(
    root: str | Path,
    profiles: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    scenario_root = Path(root)
    if profiles:
        return [(profile, scenario_root / profile) for profile in profiles]
    if not scenario_root.exists():
        return []
    discovered: list[tuple[str, Path]] = []
    for child in sorted(scenario_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob(f"*/{DEFAULT_REPORT_NAME}")):
            discovered.append((child.name, child))
    return discovered


def _retention_status(profile: str, reference_profile: str, report: Mapping[str, Any]) -> str:
    if profile == reference_profile:
        return "RETAINED_REFERENCE"
    if int(report.get("missing_count") or 0) > 0:
        return "MISSING_REQUIRED_REPORTS"
    if report.get("ok") is True:
        return "READY_FOR_VISUAL_INSPECTION"
    if report.get("candidate_beats_reference") is True:
        return "BEATS_REFERENCE_BUT_CLEARANCE_BLOCKED"
    comparison = _mapping(report.get("reference_comparison"))
    if comparison:
        low_ratio_regressions = comparison.get("low_clearance_ratio_regressions")
        if isinstance(low_ratio_regressions, list) and low_ratio_regressions:
            return "REJECT_REFERENCE_REGRESSION"
        blockers = comparison.get("blockers")
        if isinstance(blockers, list) and blockers:
            return "REJECT_REFERENCE_BLOCKED"
    return "BLOCKED_BY_CLEARANCE_GATE"


def _candidate_sort_key(item: M10ClearanceHistoryCandidate) -> tuple[float, float, float, str]:
    status_order = {
        "READY_FOR_VISUAL_INSPECTION": 0.0,
        "BEATS_REFERENCE_BUT_CLEARANCE_BLOCKED": 1.0,
        "RETAINED_REFERENCE": 2.0,
        "BLOCKED_BY_CLEARANCE_GATE": 3.0,
        "REJECT_REFERENCE_BLOCKED": 4.0,
        "REJECT_REFERENCE_REGRESSION": 5.0,
        "MISSING_REQUIRED_REPORTS": 6.0,
    }
    metrics = item.summary_metrics
    low_ratio = _as_float(metrics.get("max_low_clearance_ratio"))
    distance = _as_float(metrics.get("total_forward_distance_m"))
    return (
        status_order.get(item.retention_status, 9.0),
        low_ratio if low_ratio is not None else float("inf"),
        -(distance if distance is not None else -float("inf")),
        item.profile,
    )


def _best_candidate(
    candidates: Sequence[M10ClearanceHistoryCandidate],
) -> M10ClearanceHistoryCandidate | None:
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_sort_key)[0]


def _recommendations(
    *,
    reference_profile: str,
    best: M10ClearanceHistoryCandidate | None,
    ready_count: int,
    reference_beating_blocked_count: int,
) -> list[str]:
    if best is None:
        return [
            "Run the required MuJoCo scenario suite before choosing another clearance candidate.",
            "Use the retained s143 profile as the default reference once scenario reports exist.",
        ]
    if ready_count:
        return [
            f"Run follow-camera visual inspection for `{best.profile}` before any promotion claim.",
            "Compare the visually accepted candidate against the official Open Duck teacher suite.",
            "Keep the readiness, visual review, and teacher-comparison artifacts together.",
        ]
    if reference_beating_blocked_count:
        return [
            "Keep candidates blocked until the absolute G10 clearance gate passes.",
            "Use the best reference-beating blocked candidate as evidence for "
            "the next broader redesign.",
            "Do not promote a candidate on reference improvement alone.",
        ]
    return [
        f"Keep `{reference_profile}` as the retained blocked reference.",
        "Do not continue narrow scalar, reflex, guard, or startup-tail retunes "
        "as the primary M10 path.",
        "Use a broader clearance redesign or acquire a higher-clearance teacher "
        "before the next live training run.",
    ]


def build_m10_clearance_history(
    *,
    scenario_eval_root: str | Path = DEFAULT_SCENARIO_EVAL_ROOT,
    profiles: Sequence[str] | None = None,
    scenarios: Sequence[str] = DEFAULT_REQUIRED_SCENARIOS,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    reference_suite_dir: str | Path | None = None,
) -> M10ClearanceHistoryReport:
    scenario_ids = list(scenarios)
    root = Path(scenario_eval_root)
    resolved_reference_suite_dir = (
        Path(reference_suite_dir) if reference_suite_dir is not None else root / reference_profile
    )
    candidates: list[M10ClearanceHistoryCandidate] = []
    warnings: list[str] = []
    for profile, suite_dir in _discover_suite_dirs(root, profiles):
        readiness = build_m10_clearance_readiness(
            profile=profile,
            suite_dir=suite_dir,
            scenarios=scenario_ids,
            manifest_path=scenario_manifest,
            reference_profile=reference_profile,
            reference_suite_dir=resolved_reference_suite_dir,
        )
        readiness_dict = readiness.to_dict()
        status = _retention_status(profile, reference_profile, readiness_dict)
        candidates.append(
            M10ClearanceHistoryCandidate(
                profile=profile,
                suite_dir=str(suite_dir),
                ok=readiness.ok,
                gate_status=readiness.gate_status,
                retention_status=status,
                candidate_beats_reference=readiness.candidate_beats_reference,
                passed_count=readiness.passed_count,
                failed_count=readiness.failed_count,
                missing_count=readiness.missing_count,
                summary_metrics=readiness.summary_metrics,
                reference_comparison=readiness.reference_comparison,
                scenario_comparisons=readiness.scenario_comparisons,
                blockers=readiness.blockers,
                warnings=readiness.warnings,
            )
        )
        warnings.extend(f"{profile}: {warning}" for warning in readiness.warnings)

    candidates = sorted(candidates, key=_candidate_sort_key)
    best = _best_candidate(candidates)
    ready_count = sum(
        1 for item in candidates if item.retention_status == "READY_FOR_VISUAL_INSPECTION"
    )
    reference_beating_blocked_count = sum(
        1 for item in candidates if item.retention_status == "BEATS_REFERENCE_BUT_CLEARANCE_BLOCKED"
    )
    rejected_count = sum(1 for item in candidates if item.retention_status.startswith("REJECT_"))
    missing_count = sum(
        1 for item in candidates if item.retention_status == "MISSING_REQUIRED_REPORTS"
    )
    retained_profile = best.profile if ready_count else reference_profile
    return M10ClearanceHistoryReport(
        ok=ready_count > 0,
        reference_profile=reference_profile,
        reference_suite_dir=(
            str(resolved_reference_suite_dir) if resolved_reference_suite_dir else None
        ),
        scenario_eval_root=str(root),
        scenario_manifest=str(scenario_manifest),
        scenario_count=len(scenario_ids),
        candidate_count=len(candidates),
        ready_count=ready_count,
        reference_beating_blocked_count=reference_beating_blocked_count,
        rejected_count=rejected_count,
        missing_count=missing_count,
        retained_profile=retained_profile,
        best_candidate_profile=best.profile if best else None,
        candidates=[item.to_dict() for item in candidates],
        recommendations=_recommendations(
            reference_profile=reference_profile,
            best=best,
            ready_count=ready_count,
            reference_beating_blocked_count=reference_beating_blocked_count,
        ),
        warnings=sorted(set(warnings)),
    )


def render_markdown(report: M10ClearanceHistoryReport) -> str:
    lines = [
        "# Soridormi M10 clearance candidate history",
        "",
        f"Reference profile: `{report.reference_profile}`",
        f"Reference suite: `{report.reference_suite_dir or 'n/a'}`",
        f"Candidate count: {report.candidate_count}",
        f"Ready candidates: {report.ready_count}",
        f"Reference-beating but blocked: {report.reference_beating_blocked_count}",
        f"Retained profile: `{report.retained_profile or 'n/a'}`",
        f"Best candidate in this report: `{report.best_candidate_profile or 'n/a'}`",
        "",
        "## Candidate ranking",
        "",
        "| Profile | Status | Gate | Beats ref | Max low ratio | Min p50 m | "
        "Distance m | Falls | Missing |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.candidates:
        metrics = _mapping(item.get("summary_metrics"))
        lines.append(
            (
                "| {profile} | {status} | {gate} | {beats} | {low_ratio} | {p50} | "
                "{distance} | {falls} | {missing} |"
            ).format(
                profile=item.get("profile"),
                status=item.get("retention_status"),
                gate=item.get("gate_status"),
                beats=_format_value(item.get("candidate_beats_reference")),
                low_ratio=_format_value(metrics.get("max_low_clearance_ratio")),
                p50=_format_value(metrics.get("min_swing_clearance_p50_m")),
                distance=_format_value(metrics.get("total_forward_distance_m")),
                falls=_format_value(metrics.get("fallen_count")),
                missing=_format_value(item.get("missing_count")),
            )
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.recommendations)
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize M10 clearance candidate history.")
    parser.add_argument("--scenario-eval-root", type=Path, default=DEFAULT_SCENARIO_EVAL_ROOT)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Profile/suite directory name to include; repeat or comma-separate.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Required scenario id; repeat or comma-separate.",
    )
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--reference-profile-name", default=DEFAULT_REFERENCE_PROFILE)
    parser.add_argument("--reference-suite-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when no candidate is ready for visual inspection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = _normalise_csv(args.scenario) or list(DEFAULT_REQUIRED_SCENARIOS)
    profiles = _normalise_csv(args.profile)
    report = build_m10_clearance_history(
        scenario_eval_root=args.scenario_eval_root,
        profiles=profiles or None,
        scenarios=scenarios,
        scenario_manifest=args.scenario_manifest,
        reference_profile=args.reference_profile_name,
        reference_suite_dir=args.reference_suite_dir,
    )
    output_dir = args.output_dir
    json_output = args.json_output or output_dir / "clearance_candidate_history.json"
    markdown_output = args.output or output_dir / "clearance_candidate_history.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
