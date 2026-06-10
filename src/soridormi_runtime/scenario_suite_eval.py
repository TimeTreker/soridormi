from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from soridormi_runtime.scenario_curriculum import (
    COLLECTOR_READY_STATUSES,
    DEFAULT_SCENARIO_MANIFEST,
    ScenarioDefinition,
    list_scenarios,
)
from soridormi_runtime.scenario_rollout_eval import build_scenario_run_plan


DEFAULT_SUITE_STATUSES = tuple(sorted(COLLECTOR_READY_STATUSES))
SUPPORTED_LOCOMOTION_SKILLS = frozenset(
    {"walk_velocity", "curve_walk", "turn_in_place", "stand", "stop", "stand_idle"}
)
CLEARANCE_CHECK_NAMES = frozenset(
    {
        "foot_metrics_present",
        "touchdown_count",
        "low_clearance_swing_ratio",
        "swing_clearance_p50_m",
    }
)


@dataclass(frozen=True)
class ScenarioSuiteSelection:
    scenario_id: str
    title: str
    status: str
    family: str
    priority: int
    primary_skill: str | None
    selected: bool
    reason: str
    run_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSuitePlan:
    scenario_ids: list[str]
    selected: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    filters: dict[str, Any]
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSuiteReport:
    ok: bool
    scenario_count: int
    passed_count: int
    failed_count: int
    missing_count: int
    scenario_results: list[dict[str, Any]]
    summary_metrics: dict[str, Any]
    report_paths: list[str]
    output_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

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


def _matches_filters(
    scenario: ScenarioDefinition,
    *,
    scenario_ids: set[str] | None,
    statuses: set[str] | None,
    families: set[str] | None,
) -> tuple[bool, str]:
    if scenario_ids is not None and scenario.id not in scenario_ids:
        return False, "scenario_id filter"
    if statuses is not None and scenario.status not in statuses:
        return False, f"status {scenario.status!r} not selected"
    if families is not None and scenario.family not in families:
        return False, f"family {scenario.family!r} not selected"
    return True, "selected"


def build_scenario_suite_plan(
    *,
    manifest_path: str | Path = DEFAULT_SCENARIO_MANIFEST,
    scenarios: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    include_planned: bool = False,
    profile: str = "open_duck_forward",
    duration_s: float | None = None,
    steps: int | None = None,
    control_hz: float = 50.0,
    log_dir: str = "/data/logs",
    log_prefix_root: str = "scenario_suite",
) -> ScenarioSuitePlan:
    """Return the deterministic M9C scenario suite selection and run plans.

    M9C is intentionally locomotion-only.  Social scenarios live in the same
    curriculum but are covered by the scripted-social readiness gates instead.
    """

    scenario_filter = set(_normalise_csv(scenarios)) if scenarios else None
    family_filter = set(_normalise_csv(families)) if families else None
    if statuses:
        status_filter = set(_normalise_csv(statuses))
    elif include_planned:
        status_filter = None
    else:
        status_filter = set(DEFAULT_SUITE_STATUSES)

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scenario in list_scenarios(manifest_path, include_planned=True):
        matches, reason = _matches_filters(
            scenario,
            scenario_ids=scenario_filter,
            statuses=status_filter,
            families=family_filter,
        )
        if not matches:
            skipped.append(
                ScenarioSuiteSelection(
                    scenario_id=scenario.id,
                    title=scenario.title,
                    status=scenario.status,
                    family=scenario.family,
                    priority=scenario.priority,
                    primary_skill=scenario.primary_skill,
                    selected=False,
                    reason=reason,
                ).to_dict()
            )
            continue
        if scenario.primary_skill not in SUPPORTED_LOCOMOTION_SKILLS:
            skipped.append(
                ScenarioSuiteSelection(
                    scenario_id=scenario.id,
                    title=scenario.title,
                    status=scenario.status,
                    family=scenario.family,
                    priority=scenario.priority,
                    primary_skill=scenario.primary_skill,
                    selected=False,
                    reason=f"primary skill {scenario.primary_skill!r} is not supported by M9 locomotion rollout eval",
                ).to_dict()
            )
            continue
        try:
            plan = build_scenario_run_plan(
                scenario.id,
                manifest_path=manifest_path,
                profile=profile,
                duration_s=duration_s,
                steps=steps,
                control_hz=control_hz,
                log_prefix=f"{log_prefix_root}_{scenario.id}",
                log_dir=log_dir,
            ).to_dict()
        except Exception as exc:  # pragma: no cover - defensive path for broken manifests.
            skipped.append(
                ScenarioSuiteSelection(
                    scenario_id=scenario.id,
                    title=scenario.title,
                    status=scenario.status,
                    family=scenario.family,
                    priority=scenario.priority,
                    primary_skill=scenario.primary_skill,
                    selected=False,
                    reason=f"could not build run plan: {exc}",
                ).to_dict()
            )
            continue
        selected.append(
            ScenarioSuiteSelection(
                scenario_id=scenario.id,
                title=scenario.title,
                status=scenario.status,
                family=scenario.family,
                priority=scenario.priority,
                primary_skill=scenario.primary_skill,
                selected=True,
                reason="selected",
                run_plan=plan,
            ).to_dict()
        )

    return ScenarioSuitePlan(
        scenario_ids=[item["scenario_id"] for item in selected],
        selected=selected,
        skipped=skipped,
        filters={
            "scenario_ids": sorted(scenario_filter) if scenario_filter is not None else None,
            "statuses": sorted(status_filter) if status_filter is not None else None,
            "families": sorted(family_filter) if family_filter is not None else None,
            "include_planned": include_planned,
            "supported_locomotion_skills": sorted(SUPPORTED_LOCOMOTION_SKILLS),
        },
        manifest_path=str(manifest_path),
    )


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clearance_check_failed(report: Mapping[str, Any]) -> bool:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("name") not in CLEARANCE_CHECK_NAMES:
            continue
        if check.get("ok") is False and str(check.get("severity", "error")) == "error":
            return True
    return False


def _foot_metrics_present(
    metrics: Mapping[str, Any], stride_report: Mapping[str, Any]
) -> tuple[bool, int | None]:
    samples_with_feet = _as_int(metrics.get("samples_with_feet"))
    if samples_with_feet is None:
        samples_with_feet = _as_int(stride_report.get("samples_with_feet"))
    if samples_with_feet is not None:
        return samples_with_feet > 0, samples_with_feet
    # Backward-compatible fallback for historical reports created before
    # metrics.samples_with_feet existed.  If any foot-derived metric is present,
    # treat foot metrics as present but leave the count unknown.
    present = any(
        metrics.get(key) is not None
        for key in (
            "touchdown_count",
            "swing_clearance_p05_m",
            "swing_clearance_p50_m",
            "low_clearance_swing_ratio",
        )
    )
    return present, None


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"scenario report must contain a JSON object: {path}")
    return payload


def build_scenario_suite_report(
    report_paths: Sequence[str | Path],
    *,
    expected_scenarios: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> ScenarioSuiteReport:
    """Aggregate per-scenario M9A reports into one suite report."""

    expected = list(_normalise_csv(expected_scenarios)) if expected_scenarios else []
    expected_set = set(expected)
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    paths: list[str] = []

    for path in report_paths:
        paths.append(str(path))
        try:
            report = _load_report(path)
        except Exception as exc:
            errors.append(f"could not load report {path}: {exc}")
            continue
        scenario_id = str(report.get("scenario_id", ""))
        seen.add(scenario_id)
        metrics = _mapping(report.get("metrics"))
        thresholds = _mapping(report.get("acceptance_thresholds"))
        stride_report = _mapping(report.get("stride_step_report"))
        foot_metrics_present, samples_with_feet = _foot_metrics_present(metrics, stride_report)
        require_foot_metrics = bool(thresholds.get("require_foot_metrics"))
        item = {
            "scenario_id": scenario_id,
            "scenario_title": report.get("scenario_title"),
            "scenario_status": report.get("scenario_status"),
            "scenario_family": report.get("scenario_family"),
            "expected_skill_id": report.get("expected_skill_id"),
            "ok": bool(report.get("ok")),
            "sample_count": report.get("sample_count"),
            "duration_s": report.get("duration_s"),
            "forward_distance_m": metrics.get("forward_distance_m"),
            "mean_forward_speed_mps": metrics.get("mean_forward_speed_mps"),
            "stuck_ratio": metrics.get("stuck_ratio"),
            "fallen": metrics.get("fallen"),
            "samples_with_feet": samples_with_feet,
            "touchdown_count": metrics.get("touchdown_count"),
            "swing_clearance_p05_m": metrics.get("swing_clearance_p05_m"),
            "swing_clearance_p50_m": metrics.get("swing_clearance_p50_m"),
            "low_clearance_swing_ratio": metrics.get("low_clearance_swing_ratio"),
            "require_foot_metrics": require_foot_metrics,
            "foot_metrics_present": foot_metrics_present,
            "clearance_failed": _clearance_check_failed(report),
            "report_path": str(path),
            "error_count": len(report.get("errors", []) or []),
            "warning_count": len(report.get("warnings", []) or []),
        }
        if not item["ok"]:
            errors.append(f"scenario {scenario_id} failed acceptance")
        results.append(item)

    for scenario_id in expected:
        if scenario_id not in seen:
            errors.append(f"expected scenario {scenario_id} has no report")
            results.append(
                {
                    "scenario_id": scenario_id,
                    "ok": False,
                    "missing": True,
                    "report_path": None,
                    "error_count": 1,
                    "warning_count": 0,
                }
            )

    # Deterministic order: expected order first, then any extra reports by id.
    rank = {scenario_id: index for index, scenario_id in enumerate(expected)}
    results.sort(key=lambda item: (rank.get(str(item.get("scenario_id")), 10_000), str(item.get("scenario_id"))))

    passed = sum(1 for item in results if item.get("ok") is True)
    failed = sum(1 for item in results if item.get("ok") is False and not item.get("missing"))
    missing = sum(1 for item in results if item.get("missing"))
    distances = [_as_float(item.get("forward_distance_m")) for item in results]
    speeds = [_as_float(item.get("mean_forward_speed_mps")) for item in results]
    stuck_ratios = [_as_float(item.get("stuck_ratio")) for item in results]
    distances_f = [value for value in distances if value is not None]
    speeds_f = [value for value in speeds if value is not None]
    stuck_f = [value for value in stuck_ratios if value is not None]
    swing_p50 = [_as_float(item.get("swing_clearance_p50_m")) for item in results]
    swing_p05 = [_as_float(item.get("swing_clearance_p05_m")) for item in results]
    low_clearance_ratios = [_as_float(item.get("low_clearance_swing_ratio")) for item in results]
    swing_p50_f = [value for value in swing_p50 if value is not None]
    swing_p05_f = [value for value in swing_p05 if value is not None]
    low_clearance_f = [value for value in low_clearance_ratios if value is not None]

    summary_metrics = {
        "total_forward_distance_m": sum(distances_f) if distances_f else None,
        "mean_forward_distance_m": _mean(distances_f),
        "mean_forward_speed_mps": _mean(speeds_f),
        "max_stuck_ratio": max(stuck_f) if stuck_f else None,
        "min_swing_clearance_p50_m": min(swing_p50_f) if swing_p50_f else None,
        "min_swing_clearance_p05_m": min(swing_p05_f) if swing_p05_f else None,
        "max_low_clearance_ratio": max(low_clearance_f) if low_clearance_f else None,
        "clearance_failed_count": sum(
            1 for item in results if item.get("clearance_failed") is True
        ),
        "foot_metrics_missing_count": sum(
            1
            for item in results
            if item.get("require_foot_metrics") is True
            and item.get("foot_metrics_present") is False
        ),
        "fallen_count": sum(1 for item in results if item.get("fallen") is True),
        "total_samples": sum(int(item.get("sample_count") or 0) for item in results),
    }

    return ScenarioSuiteReport(
        ok=not errors,
        scenario_count=len(results),
        passed_count=passed,
        failed_count=failed,
        missing_count=missing,
        scenario_results=results,
        summary_metrics=summary_metrics,
        report_paths=paths,
        output_dir=str(output_dir) if output_dir is not None else None,
        warnings=warnings,
        errors=errors,
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def render_suite_markdown(report: ScenarioSuiteReport) -> str:
    lines = [
        "# Soridormi scenario suite report",
        "",
        f"Result: {'PASS' if report.ok else 'FAILED'}",
        f"Scenarios: {report.scenario_count}",
        f"Passed: {report.passed_count}",
        f"Failed: {report.failed_count}",
        f"Missing: {report.missing_count}",
        "",
        "## Summary metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report.summary_metrics.items():
        lines.append(f"| {key} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## Scenario results",
            "",
            (
                "| Scenario | Result | Skill | Distance m | Mean speed m/s | Stuck ratio "
                "| Swing p50 m | Low-clearance ratio | Fallen | Report |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in report.scenario_results:
        result = "PASS" if item.get("ok") else "FAIL"
        report_path = item.get("report_path") or "n/a"
        lines.append(
            (
                "| {scenario} | {result} | {skill} | {distance} | {speed} | {stuck} "
                "| {swing_p50} | {low_clearance} | {fallen} | {report} |"
            ).format(
                scenario=item.get("scenario_id"),
                result=result,
                skill=item.get("expected_skill_id") or "n/a",
                distance=_format_value(item.get("forward_distance_m")),
                speed=_format_value(item.get("mean_forward_speed_mps")),
                stuck=_format_value(item.get("stuck_ratio")),
                swing_p50=_format_value(item.get("swing_clearance_p50_m")),
                low_clearance=_format_value(item.get("low_clearance_swing_ratio")),
                fallen=_format_value(item.get("fallen")),
                report=report_path,
            )
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _expand_report_paths(values: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for raw in values:
        matches = sorted(str(path) for path in Path().glob(raw)) if any(ch in raw for ch in "*?[") else []
        expanded.extend(matches or [raw])
    return expanded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or aggregate Soridormi scenario rollout suites.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--scenario", action="append", default=[], help="Scenario id to include; repeat or comma-separate.")
    parser.add_argument("--status", action="append", default=[], help="Scenario status to include; repeat or comma-separate.")
    parser.add_argument("--family", action="append", default=[], help="Scenario family to include; repeat or comma-separate.")
    parser.add_argument("--include-planned", action="store_true", help="Include planned scenarios unless --status is provided.")
    parser.add_argument("--profile", default="open_duck_forward")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--log-dir", default="/data/logs")
    parser.add_argument("--log-prefix-root", default="scenario_suite")
    parser.add_argument("--print-suite-plan", action="store_true")
    parser.add_argument("--reports", nargs="*", default=[], help="Per-scenario report JSON files or shell glob patterns.")
    parser.add_argument("--expected-scenario", action="append", default=[], help="Expected scenario id; repeat or comma-separate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown suite report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON suite report path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.print_suite_plan:
        plan = build_scenario_suite_plan(
            manifest_path=args.scenario_manifest,
            scenarios=args.scenario,
            statuses=args.status,
            families=args.family,
            include_planned=args.include_planned,
            profile=args.profile,
            duration_s=args.duration_s,
            steps=args.steps,
            control_hz=args.control_hz,
            log_dir=args.log_dir,
            log_prefix_root=args.log_prefix_root,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0 if plan.scenario_ids else 1

    if not args.reports:
        raise SystemExit("--reports is required unless --print-suite-plan is used")
    report = build_scenario_suite_report(
        _expand_report_paths(args.reports),
        expected_scenarios=_normalise_csv(args.expected_scenario),
        output_dir=args.json_output.parent if args.json_output is not None else None,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_suite_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_suite_markdown(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
