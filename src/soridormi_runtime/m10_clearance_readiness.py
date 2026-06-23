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
    reference_profile: str | None = None
    reference_suite_dir: str | None = None
    reference_summary_metrics: dict[str, Any] | None = None
    scenario_comparisons: list[dict[str, Any]] = field(default_factory=list)
    reference_comparison: dict[str, Any] | None = None
    candidate_beats_reference: bool | None = None

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


def _sum_float(values: Sequence[Any]) -> float | None:
    parsed = [_as_float(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return sum(clean) if clean else None


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _clearance_p50_margin(item: M10ClearanceScenarioReadiness) -> float | None:
    swing_p50 = _as_float(item.metrics.get("swing_clearance_p50_m"))
    threshold = _as_float(item.thresholds.get("min_swing_clearance_m"))
    if swing_p50 is None or threshold is None:
        return None
    return swing_p50 - threshold


def _low_clearance_ratio_excess(item: M10ClearanceScenarioReadiness) -> float | None:
    low_ratio = _as_float(item.metrics.get("low_clearance_swing_ratio"))
    threshold = _as_float(item.thresholds.get("max_low_clearance_ratio"))
    if low_ratio is None or threshold is None:
        return None
    return low_ratio - threshold


def _build_summary_metrics(
    scenario_reports: Sequence[M10ClearanceScenarioReadiness],
) -> dict[str, Any]:
    metrics = [item.metrics for item in scenario_reports]
    return {
        "min_swing_clearance_p50_m": _min_float([item.get("swing_clearance_p50_m") for item in metrics]),
        "min_swing_clearance_p05_m": _min_float([item.get("swing_clearance_p05_m") for item in metrics]),
        "max_low_clearance_ratio": _max_float([item.get("low_clearance_swing_ratio") for item in metrics]),
        "min_swing_clearance_p50_margin_m": _min_float(
            [_clearance_p50_margin(item) for item in scenario_reports]
        ),
        "max_low_clearance_ratio_excess": _max_float(
            [_low_clearance_ratio_excess(item) for item in scenario_reports]
        ),
        "total_forward_distance_m": _sum_float([item.get("forward_distance_m") for item in metrics]),
        "max_stuck_ratio": _max_float([item.get("stuck_ratio") for item in metrics]),
        "fallen_count": sum(1 for item in metrics if _is_true(item.get("fallen"))),
        "clearance_failed_count": sum(1 for item in scenario_reports if not item.clearance_ok),
        "foot_metrics_missing_count": sum(1 for item in scenario_reports if not item.foot_metrics_present),
        "scenario_acceptance_failed_count": sum(1 for item in scenario_reports if not item.scenario_acceptance_ok),
    }


def _delta(candidate_value: Any, reference_value: Any) -> float | None:
    candidate = _as_float(candidate_value)
    reference = _as_float(reference_value)
    if candidate is None or reference is None:
        return None
    return candidate - reference


def _scenario_comparisons(
    candidate_reports: Sequence[M10ClearanceScenarioReadiness],
    reference_reports: Sequence[M10ClearanceScenarioReadiness],
) -> list[dict[str, Any]]:
    references = {item.scenario_id: item for item in reference_reports}
    comparisons: list[dict[str, Any]] = []
    for candidate in candidate_reports:
        reference = references.get(candidate.scenario_id)
        if reference is None:
            comparisons.append(
                {
                    "scenario_id": candidate.scenario_id,
                    "status": "MISSING_REFERENCE",
                    "candidate_status": candidate.status,
                    "reference_status": None,
                }
            )
            continue
        candidate_metrics = candidate.metrics
        reference_metrics = reference.metrics
        comparisons.append(
            {
                "scenario_id": candidate.scenario_id,
                "status": "COMPARED",
                "candidate_status": candidate.status,
                "reference_status": reference.status,
                "candidate_ok": candidate.ok,
                "reference_ok": reference.ok,
                "candidate_low_clearance_ratio": candidate_metrics.get("low_clearance_swing_ratio"),
                "reference_low_clearance_ratio": reference_metrics.get("low_clearance_swing_ratio"),
                "delta_low_clearance_ratio": _delta(
                    candidate_metrics.get("low_clearance_swing_ratio"),
                    reference_metrics.get("low_clearance_swing_ratio"),
                ),
                "candidate_low_clearance_ratio_excess": _low_clearance_ratio_excess(candidate),
                "reference_low_clearance_ratio_excess": _low_clearance_ratio_excess(reference),
                "delta_low_clearance_ratio_excess": _delta(
                    _low_clearance_ratio_excess(candidate),
                    _low_clearance_ratio_excess(reference),
                ),
                "candidate_swing_clearance_p50_m": candidate_metrics.get("swing_clearance_p50_m"),
                "reference_swing_clearance_p50_m": reference_metrics.get("swing_clearance_p50_m"),
                "delta_swing_clearance_p50_m": _delta(
                    candidate_metrics.get("swing_clearance_p50_m"),
                    reference_metrics.get("swing_clearance_p50_m"),
                ),
                "candidate_forward_distance_m": candidate_metrics.get("forward_distance_m"),
                "reference_forward_distance_m": reference_metrics.get("forward_distance_m"),
                "delta_forward_distance_m": _delta(
                    candidate_metrics.get("forward_distance_m"),
                    reference_metrics.get("forward_distance_m"),
                ),
                "candidate_fallen": candidate_metrics.get("fallen"),
                "reference_fallen": reference_metrics.get("fallen"),
            }
        )
    return comparisons


def _reference_comparison(
    *,
    candidate_summary: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    distance_floor_ratio: float = 0.90,
) -> dict[str, Any]:
    candidate_low_excess = _as_float(candidate_summary.get("max_low_clearance_ratio_excess"))
    reference_low_excess = _as_float(reference_summary.get("max_low_clearance_ratio_excess"))
    candidate_p50_margin = _as_float(candidate_summary.get("min_swing_clearance_p50_margin_m"))
    reference_p50_margin = _as_float(reference_summary.get("min_swing_clearance_p50_margin_m"))
    candidate_distance = _as_float(candidate_summary.get("total_forward_distance_m"))
    reference_distance = _as_float(reference_summary.get("total_forward_distance_m"))
    candidate_falls = int(candidate_summary.get("fallen_count") or 0)
    reference_falls = int(reference_summary.get("fallen_count") or 0)
    candidate_clearance_failed = int(candidate_summary.get("clearance_failed_count") or 0)
    reference_clearance_failed = int(reference_summary.get("clearance_failed_count") or 0)

    low_excess_delta = _delta(candidate_low_excess, reference_low_excess)
    p50_margin_delta = _delta(candidate_p50_margin, reference_p50_margin)
    distance_delta = _delta(candidate_distance, reference_distance)
    no_new_falls = candidate_falls <= reference_falls
    if candidate_distance is not None and reference_distance is not None and reference_distance > 0:
        movement_preserved = candidate_distance >= reference_distance * distance_floor_ratio
    else:
        movement_preserved = False

    clearance_better = candidate_clearance_failed < reference_clearance_failed
    if not clearance_better and low_excess_delta is not None:
        clearance_better = low_excess_delta < -1e-9
    if not clearance_better and low_excess_delta is not None and abs(low_excess_delta) <= 1e-9:
        clearance_better = p50_margin_delta is not None and p50_margin_delta > 1e-9

    blockers: list[str] = []
    if not clearance_better:
        blockers.append("candidate does not improve the G10 clearance bottleneck versus reference")
    if not no_new_falls:
        blockers.append("candidate adds falls versus reference")
    if not movement_preserved:
        blockers.append(
            f"candidate total distance is below {distance_floor_ratio:.0%} of the reference distance"
        )

    return {
        "candidate_beats_reference": clearance_better and no_new_falls and movement_preserved,
        "clearance_bottleneck_improved": clearance_better,
        "no_new_falls": no_new_falls,
        "movement_distance_preserved": movement_preserved,
        "distance_floor_ratio": distance_floor_ratio,
        "delta_max_low_clearance_ratio_excess": low_excess_delta,
        "delta_min_swing_clearance_p50_margin_m": p50_margin_delta,
        "delta_total_forward_distance_m": distance_delta,
        "candidate_clearance_failed_count": candidate_clearance_failed,
        "reference_clearance_failed_count": reference_clearance_failed,
        "blockers": blockers,
    }


def _recommendations(blockers: Sequence[str]) -> list[str]:
    if blockers:
        return [
            "Keep the current context policy blocked from promotion.",
            "Do not recollect unchanged teacher behavior when the teacher also fails the clearance target.",
            "Use a teacher that demonstrates the target clearance or train a bounded phase/state-conditioned residual.",
            "Rerun the required scenario suite before visual inspection.",
        ]
    return [
        "Run follow-camera visual inspection for all required scenarios.",
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
    reference_profile: str | None = None,
    reference_suite_dir: str | Path | None = None,
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
    summary_metrics = _build_summary_metrics(scenario_reports)
    ok = not blockers and passed == len(scenario_reports)
    gate_status = "READY_FOR_VISUAL_INSPECTION" if ok else "BLOCKED_BY_CLEARANCE_GATE"
    report_paths_out = [item.report_path for item in scenario_reports if item.report_path]
    reference_reports: list[M10ClearanceScenarioReadiness] = []
    reference_summary_metrics: dict[str, Any] | None = None
    scenario_comparisons: list[dict[str, Any]] = []
    reference_comparison: dict[str, Any] | None = None
    candidate_beats_reference: bool | None = None
    resolved_reference_suite_dir = Path(reference_suite_dir) if reference_suite_dir is not None else None
    if resolved_reference_suite_dir is not None:
        for scenario_id in scenario_ids:
            reference_reports.append(
                _scenario_readiness(
                    scenario_id,
                    _scenario_report_path(resolved_reference_suite_dir, scenario_id),
                    manifest_path=manifest_path,
                )
            )
        reference_summary_metrics = _build_summary_metrics(reference_reports)
        scenario_comparisons = _scenario_comparisons(scenario_reports, reference_reports)
        reference_comparison = _reference_comparison(
            candidate_summary=summary_metrics,
            reference_summary=reference_summary_metrics,
        )
        candidate_beats_reference = bool(reference_comparison.get("candidate_beats_reference"))
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
        reference_profile=reference_profile,
        reference_suite_dir=str(resolved_reference_suite_dir) if resolved_reference_suite_dir is not None else None,
        reference_summary_metrics=reference_summary_metrics,
        scenario_comparisons=scenario_comparisons,
        reference_comparison=reference_comparison,
        candidate_beats_reference=candidate_beats_reference,
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
        "# Soridormi clearance readiness report",
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
    if report.reference_suite_dir:
        lines.extend(
            [
                "",
                "## Reference comparison",
                "",
                f"Reference profile: `{report.reference_profile or 'n/a'}`",
                f"Reference suite: `{report.reference_suite_dir}`",
                f"Candidate beats reference: {_format_value(report.candidate_beats_reference)}",
                "",
                "| Scenario | Delta low ratio | Delta p50 m | Delta distance m | Candidate status | Reference status |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for item in report.scenario_comparisons:
            lines.append(
                "| {scenario} | {low_delta} | {p50_delta} | {distance_delta} | {candidate_status} | {reference_status} |".format(
                    scenario=item.get("scenario_id"),
                    low_delta=_format_value(item.get("delta_low_clearance_ratio")),
                    p50_delta=_format_value(item.get("delta_swing_clearance_p50_m")),
                    distance_delta=_format_value(item.get("delta_forward_distance_m")),
                    candidate_status=item.get("candidate_status") or "n/a",
                    reference_status=item.get("reference_status") or "n/a",
                )
            )
        comparison = _mapping(report.reference_comparison)
        blockers = comparison.get("blockers", [])
        if blockers:
            lines.extend(["", "Reference blockers:", ""])
            lines.extend(f"- {blocker}" for blocker in blockers if isinstance(blocker, str))
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze context-policy clearance readiness.")
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE, help="Policy profile name used to resolve the default suite dir.")
    parser.add_argument("--suite-dir", type=Path, default=None, help="Directory containing per-scenario subdirectories and reports.")
    parser.add_argument("--report", action="append", default=[], help="Explicit scenario_rollout_report.json path; repeat as needed.")
    parser.add_argument("--scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--reference-profile-name", default=None, help="Optional retained reference profile name for comparison.")
    parser.add_argument("--reference-suite-dir", type=Path, default=None, help="Optional retained reference suite directory.")
    parser.add_argument(
        "--require-reference-improvement",
        action="store_true",
        help="Exit nonzero unless the candidate improves the reference without new falls or major distance loss.",
    )
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
        reference_profile=args.reference_profile_name,
        reference_suite_dir=args.reference_suite_dir,
    )
    output_dir = args.output_dir or Path("artifacts/clearance_readiness") / args.profile_name
    json_output = args.json_output or output_dir / "clearance_readiness.json"
    markdown_output = args.output or output_dir / "clearance_readiness.md"

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    if args.strict and not report.ok:
        return 1
    if args.require_reference_improvement:
        if report.candidate_beats_reference is not True:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
