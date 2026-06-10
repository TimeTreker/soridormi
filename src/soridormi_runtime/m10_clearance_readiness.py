from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.scenario_curriculum import DEFAULT_SCENARIO_MANIFEST, get_scenario_definition
from soridormi_runtime.scenario_rollout_eval import thresholds_from_scenario_manifest
from soridormi_runtime.scenario_suite_eval import CLEARANCE_CHECK_NAMES

DEFAULT_PROFILE = "context_stage1_three_scenario_10ep_e80"
DEFAULT_REQUIRED_SCENARIOS = (
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
)
DEFAULT_REPORT_NAME = "scenario_rollout_report.json"


@dataclass(frozen=True)
class M10ClearanceScenarioReadiness:
    scenario_id: str
    report_path: str | None
    ok: bool
    status: str
    scenario_acceptance_ok: bool
    clearance_ok: bool
    foot_metrics_present: bool
    samples_with_feet: int | None
    thresholds: dict[str, Any]
    metrics: dict[str, Any]
    failed_clearance_checks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M10ClearanceReadinessReport:
    ok: bool
    profile: str
    gate_status: str
    scenario_count: int
    passed_count: int
    failed_count: int
    missing_count: int
    suite_dir: str | None
    scenario_manifest: str
    report_paths: list[str]
    scenarios: list[dict[str, Any]]
    summary_metrics: dict[str, Any]
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    next_required_evidence: list[str] = field(default_factory=list)
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"report must contain a JSON object: {path}")
    return payload


def _report_paths_by_scenario(report_paths: Sequence[str | Path]) -> tuple[dict[str, Path], list[str]]:
    out: dict[str, Path] = {}
    warnings: list[str] = []
    for path in report_paths:
        report_path = Path(path)
        try:
            payload = _load_json(report_path)
        except Exception as exc:
            warnings.append(f"could not inspect report {report_path}: {exc}")
            continue
        scenario_id = str(payload.get("scenario_id") or "").strip()
        if not scenario_id:
            warnings.append(f"could not inspect report {report_path}: missing scenario_id")
            continue
        if scenario_id in out:
            warnings.append(f"duplicate report for {scenario_id}; using {report_path}")
        out[scenario_id] = report_path
    return out, warnings


def _scenario_report_path(suite_dir: str | Path, scenario_id: str) -> Path:
    return Path(suite_dir) / scenario_id / DEFAULT_REPORT_NAME


def _thresholds_for_scenario(scenario_id: str, manifest_path: str | Path) -> dict[str, Any]:
    scenario = get_scenario_definition(scenario_id, manifest_path)
    return thresholds_from_scenario_manifest(scenario).as_dict()


def _foot_metrics_present(metrics: Mapping[str, Any], stride_report: Mapping[str, Any]) -> tuple[bool, int | None]:
    samples_with_feet = _as_int(metrics.get("samples_with_feet"))
    if samples_with_feet is None:
        samples_with_feet = _as_int(stride_report.get("samples_with_feet"))
    if samples_with_feet is not None:
        return samples_with_feet > 0, samples_with_feet
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


def _failed_clearance_checks(report: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return failed
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("name") or "")
        if name not in CLEARANCE_CHECK_NAMES:
            continue
        if check.get("ok") is False and str(check.get("severity", "error")) == "error":
            failed.append(name)
    return sorted(set(failed))


def _scenario_readiness(
    scenario_id: str,
    report_path: Path | None,
    *,
    manifest_path: str | Path,
) -> M10ClearanceScenarioReadiness:
    thresholds = _thresholds_for_scenario(scenario_id, manifest_path)
    if report_path is None or not report_path.exists():
        return M10ClearanceScenarioReadiness(
            scenario_id=scenario_id,
            report_path=str(report_path) if report_path is not None else None,
            ok=False,
            status="MISSING_REPORT",
            scenario_acceptance_ok=False,
            clearance_ok=False,
            foot_metrics_present=False,
            samples_with_feet=None,
            thresholds=thresholds,
            metrics={},
            blockers=[f"missing scenario rollout report for {scenario_id}"],
        )

    try:
        report = _load_json(report_path)
    except Exception as exc:
        return M10ClearanceScenarioReadiness(
            scenario_id=scenario_id,
            report_path=str(report_path),
            ok=False,
            status="INVALID_REPORT",
            scenario_acceptance_ok=False,
            clearance_ok=False,
            foot_metrics_present=False,
            samples_with_feet=None,
            thresholds=thresholds,
            metrics={},
            blockers=[f"could not load scenario rollout report for {scenario_id}: {exc}"],
        )

    metrics = dict(_mapping(report.get("metrics")))
    stride_report = _mapping(report.get("stride_step_report"))
    report_thresholds = dict(_mapping(report.get("acceptance_thresholds")))
    effective_thresholds = {**thresholds, **report_thresholds}
    require_foot_metrics = bool(effective_thresholds.get("require_foot_metrics"))
    min_swing_clearance = _as_float(effective_thresholds.get("min_swing_clearance_m"))
    max_low_ratio = _as_float(effective_thresholds.get("max_low_clearance_ratio"))
    swing_p50 = _as_float(metrics.get("swing_clearance_p50_m"))
    low_ratio = _as_float(metrics.get("low_clearance_swing_ratio"))
    foot_metrics_present, samples_with_feet = _foot_metrics_present(metrics, stride_report)
    failed_checks = _failed_clearance_checks(report)
    blockers: list[str] = []
    warnings: list[str] = []

    if require_foot_metrics and not foot_metrics_present:
        blockers.append(f"{scenario_id}: required foot metrics are missing")
    if failed_checks:
        blockers.append(f"{scenario_id}: failed clearance checks: {', '.join(failed_checks)}")
    if require_foot_metrics and swing_p50 is None:
        blockers.append(f"{scenario_id}: swing_clearance_p50_m is missing")
    if require_foot_metrics and low_ratio is None:
        blockers.append(f"{scenario_id}: low_clearance_swing_ratio is missing")
    if swing_p50 is not None and min_swing_clearance is not None and swing_p50 < min_swing_clearance:
        blockers.append(
            f"{scenario_id}: swing_clearance_p50_m {swing_p50:.4f}m < {min_swing_clearance:.4f}m"
        )
    if low_ratio is not None and max_low_ratio is not None and low_ratio > max_low_ratio:
        blockers.append(
            f"{scenario_id}: low_clearance_swing_ratio {low_ratio:.3f} > {max_low_ratio:.3f}"
        )

    scenario_acceptance_ok = bool(report.get("ok"))
    if not scenario_acceptance_ok:
        blockers.append(f"{scenario_id}: scenario acceptance failed")

    clearance_ok = not any(
        "clearance" in blocker or "foot metrics" in blocker or "swing" in blocker
        for blocker in blockers
    )
    ok = scenario_acceptance_ok and clearance_ok
    if ok:
        status = "PASS"
    elif not clearance_ok:
        status = "FAIL_CLEARANCE_GATE"
    else:
        status = "FAIL_SCENARIO_ACCEPTANCE"

    if str(report.get("scenario_id") or scenario_id) != scenario_id:
        warnings.append(
            f"report scenario_id {report.get('scenario_id')!r} did not match expected {scenario_id!r}"
        )

    return M10ClearanceScenarioReadiness(
        scenario_id=scenario_id,
        report_path=str(report_path),
        ok=ok,
        status=status,
        scenario_acceptance_ok=scenario_acceptance_ok,
        clearance_ok=clearance_ok,
        foot_metrics_present=foot_metrics_present,
        samples_with_feet=samples_with_feet,
        thresholds=effective_thresholds,
        metrics=metrics,
        failed_clearance_checks=failed_checks,
        blockers=sorted(set(blockers)),
        warnings=warnings,
    )


def _min_float(values: Sequence[Any]) -> float | None:
    parsed = [_as_float(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return min(clean) if clean else None


def _max_float(values: Sequence[Any]) -> float | None:
    parsed = [_as_float(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return max(clean) if clean else None


def _recommendations(blockers: Sequence[str]) -> list[str]:
    if blockers:
        return [
            "Keep the current context policy blocked from M10 promotion.",
            "Collect clearance-focused teacher rollouts for flat, start/stop, and curve scenarios.",
            "Retrain a candidate and rerun the required scenario suite before visual inspection.",
        ]
    return [
        "Run follow-camera visual inspection for all required M10 scenarios.",
        "Re-evaluate the candidate against the official Open Duck teacher baseline.",
        "Retain the readiness JSON/Markdown artifacts with the candidate evidence package.",
    ]


def build_m10_clearance_readiness(
    *,
    profile: str = DEFAULT_PROFILE,
    suite_dir: str | Path | None = None,
    report_paths: Sequence[str | Path] | None = None,
    scenarios: Sequence[str] = DEFAULT_REQUIRED_SCENARIOS,
    manifest_path: str | Path = DEFAULT_SCENARIO_MANIFEST,
) -> M10ClearanceReadinessReport:
    scenario_ids = list(scenarios)
    resolved_suite_dir = Path(suite_dir) if suite_dir is not None else Path("artifacts/scenario_eval") / f"{profile}_suite"
    warnings: list[str] = []
    explicit_reports: dict[str, Path] = {}
    if report_paths:
        explicit_reports, warnings = _report_paths_by_scenario(report_paths)

    scenario_reports: list[M10ClearanceScenarioReadiness] = []
    for scenario_id in scenario_ids:
        report_path = explicit_reports.get(scenario_id) if explicit_reports else _scenario_report_path(resolved_suite_dir, scenario_id)
        scenario_reports.append(
            _scenario_readiness(
                scenario_id,
                report_path,
                manifest_path=manifest_path,
            )
        )

    scenario_dicts = [item.to_dict() for item in scenario_reports]
    blockers = sorted({blocker for item in scenario_reports for blocker in item.blockers})
    warnings.extend(warning for item in scenario_reports for warning in item.warnings)
    passed = sum(1 for item in scenario_reports if item.ok)
    failed = sum(1 for item in scenario_reports if not item.ok and item.status != "MISSING_REPORT")
    missing = sum(1 for item in scenario_reports if item.status == "MISSING_REPORT")
    metrics = [item.metrics for item in scenario_reports]
    summary_metrics = {
        "min_swing_clearance_p50_m": _min_float([item.get("swing_clearance_p50_m") for item in metrics]),
        "min_swing_clearance_p05_m": _min_float([item.get("swing_clearance_p05_m") for item in metrics]),
        "max_low_clearance_ratio": _max_float([item.get("low_clearance_swing_ratio") for item in metrics]),
        "clearance_failed_count": sum(1 for item in scenario_reports if not item.clearance_ok),
        "foot_metrics_missing_count": sum(1 for item in scenario_reports if not item.foot_metrics_present),
        "scenario_acceptance_failed_count": sum(1 for item in scenario_reports if not item.scenario_acceptance_ok),
    }
    ok = not blockers and passed == len(scenario_reports)
    gate_status = "READY_FOR_VISUAL_INSPECTION" if ok else "BLOCKED_BY_CLEARANCE_GATE"
    report_paths_out = [item.report_path for item in scenario_reports if item.report_path]
    return M10ClearanceReadinessReport(
        ok=ok,
        profile=profile,
        gate_status=gate_status,
        scenario_count=len(scenario_reports),
        passed_count=passed,
        failed_count=failed,
        missing_count=missing,
        suite_dir=str(resolved_suite_dir) if suite_dir is not None or not report_paths else None,
        scenario_manifest=str(manifest_path),
        report_paths=report_paths_out,
        scenarios=scenario_dicts,
        summary_metrics=summary_metrics,
        blockers=blockers,
        recommendations=_recommendations(blockers),
        next_required_evidence=[
            "follow-camera visual inspection",
            "official-teacher comparison",
        ],
        warnings=warnings,
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def render_markdown(report: M10ClearanceReadinessReport) -> str:
    lines = [
        "# Soridormi M10 clearance readiness report",
        "",
        f"Profile: `{report.profile}`",
        f"Result: {'PASS' if report.ok else 'FAILED'}",
        f"Gate status: `{report.gate_status}`",
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
            "## Scenario readiness",
            "",
            "| Scenario | Status | Acceptance | Clearance | Swing p50 m | Low-clearance ratio | Report |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in report.scenarios:
        metrics = _mapping(item.get("metrics"))
        lines.append(
            "| {scenario} | {status} | {acceptance} | {clearance} | {swing_p50} | {low_ratio} | {path} |".format(
                scenario=item.get("scenario_id"),
                status=item.get("status"),
                acceptance="PASS" if item.get("scenario_acceptance_ok") else "FAIL",
                clearance="PASS" if item.get("clearance_ok") else "FAIL",
                swing_p50=_format_value(metrics.get("swing_clearance_p50_m")),
                low_ratio=_format_value(metrics.get("low_clearance_swing_ratio")),
                path=item.get("report_path") or "n/a",
            )
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {blocker}" for blocker in report.blockers) if report.blockers else lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    lines.extend(["", "## Next required evidence", ""])
    lines.extend(f"- {item}" for item in report.next_required_evidence)
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze M10 context-policy clearance readiness.")
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE, help="Policy profile name used to resolve the default suite dir.")
    parser.add_argument("--suite-dir", type=Path, default=None, help="Directory containing per-scenario subdirectories and reports.")
    parser.add_argument("--report", action="append", default=[], help="Explicit scenario_rollout_report.json path; repeat as needed.")
    parser.add_argument("--scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for JSON and Markdown readiness reports.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when clearance readiness fails.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = _normalise_csv(args.scenario) or list(DEFAULT_REQUIRED_SCENARIOS)
    report = build_m10_clearance_readiness(
        profile=args.profile_name,
        suite_dir=args.suite_dir,
        report_paths=args.report,
        scenarios=scenarios,
        manifest_path=args.scenario_manifest,
    )
    output_dir = args.output_dir or Path("artifacts/m10_clearance_readiness") / args.profile_name
    json_output = args.json_output or output_dir / "m10_clearance_readiness.json"
    markdown_output = args.output or output_dir / "m10_clearance_readiness.md"

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 1 if args.strict and not report.ok else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
