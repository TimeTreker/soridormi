from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.clearance_readiness import DEFAULT_REQUIRED_SCENARIOS

DEFAULT_OUTPUT_ROOT = Path("artifacts/policy_teacher_comparison")


@dataclass(frozen=True)
class M10TeacherComparisonThresholds:
    min_forward_distance_ratio: float = 0.8
    min_forward_speed_ratio: float = 0.8
    max_stuck_ratio_regression: float = 0.1
    max_candidate_fallen_count: int = 0


@dataclass(frozen=True)
class M10TeacherScenarioComparison:
    scenario_id: str
    reference_ok: bool
    candidate_ok: bool
    reference_forward_distance_m: float | None
    candidate_forward_distance_m: float | None
    forward_distance_ratio: float | None
    reference_forward_speed_mps: float | None
    candidate_forward_speed_mps: float | None
    forward_speed_ratio: float | None
    reference_stuck_ratio: float | None
    candidate_stuck_ratio: float | None
    stuck_ratio_regression: float | None
    reference_swing_clearance_p50_m: float | None
    candidate_swing_clearance_p50_m: float | None
    ok: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M10TeacherComparison:
    ok: bool
    status: str
    reference_suite: str
    candidate_suite: str
    required_scenarios: list[str]
    scenarios: list[dict[str, Any]]
    thresholds: dict[str, Any]
    reference_summary: dict[str, Any]
    candidate_summary: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"suite summary must contain a JSON object: {resolved}")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _scenario_map(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = summary.get("scenario_results", [])
    if not isinstance(values, list):
        return {}
    scenarios: dict[str, Mapping[str, Any]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            continue
        scenario_id = str(item.get("scenario_id") or "").strip()
        if scenario_id:
            scenarios[scenario_id] = item
    return scenarios


def _summary_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("summary_metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return {
        "ok": summary.get("ok") is True,
        "scenario_count": int(summary.get("scenario_count") or 0),
        "passed_count": int(summary.get("passed_count") or 0),
        "failed_count": int(summary.get("failed_count") or 0),
        "missing_count": int(summary.get("missing_count") or 0),
        "fallen_count": int(metrics.get("fallen_count") or 0),
        "total_forward_distance_m": _finite_float(metrics.get("total_forward_distance_m")),
        "mean_forward_speed_mps": _finite_float(metrics.get("mean_forward_speed_mps")),
        "max_stuck_ratio": _finite_float(metrics.get("max_stuck_ratio")),
    }


def compare_clearance_teacher_suites(
    reference_suite: Mapping[str, Any],
    candidate_suite: Mapping[str, Any],
    *,
    reference_path: str = "<reference suite>",
    candidate_path: str = "<candidate suite>",
    required_scenarios: Sequence[str] = DEFAULT_REQUIRED_SCENARIOS,
    thresholds: M10TeacherComparisonThresholds | None = None,
    output_dir: str | Path | None = None,
) -> M10TeacherComparison:
    thresholds = thresholds or M10TeacherComparisonThresholds()
    reference_by_id = _scenario_map(reference_suite)
    candidate_by_id = _scenario_map(candidate_suite)
    reference_summary = _summary_view(reference_suite)
    candidate_summary = _summary_view(candidate_suite)
    errors: list[str] = []
    warnings = [
        "This comparison does not replace the absolute swing-clearance readiness gate."
    ]

    if not reference_summary["ok"]:
        errors.append("reference suite is not passing")
    if not candidate_summary["ok"]:
        errors.append("candidate suite is not passing")
    if candidate_summary["fallen_count"] > thresholds.max_candidate_fallen_count:
        errors.append(
            "candidate fallen_count "
            f"{candidate_summary['fallen_count']} exceeds "
            f"{thresholds.max_candidate_fallen_count}"
        )

    scenario_results: list[M10TeacherScenarioComparison] = []
    for scenario_id in required_scenarios:
        reference = reference_by_id.get(scenario_id)
        candidate = candidate_by_id.get(scenario_id)
        scenario_errors: list[str] = []
        if reference is None:
            errors.append(f"{scenario_id}: missing from reference suite")
            continue
        if candidate is None:
            errors.append(f"{scenario_id}: missing from candidate suite")
            continue

        reference_distance = _finite_float(reference.get("forward_distance_m"))
        candidate_distance = _finite_float(candidate.get("forward_distance_m"))
        distance_ratio = _safe_ratio(candidate_distance, reference_distance)
        if distance_ratio is None:
            scenario_errors.append("forward distance ratio is unavailable")
        elif distance_ratio < thresholds.min_forward_distance_ratio:
            scenario_errors.append(
                f"forward distance ratio {distance_ratio:.6g} is below "
                f"{thresholds.min_forward_distance_ratio:.6g}"
            )

        reference_speed = _finite_float(reference.get("mean_forward_speed_mps"))
        candidate_speed = _finite_float(candidate.get("mean_forward_speed_mps"))
        speed_ratio = _safe_ratio(candidate_speed, reference_speed)
        if speed_ratio is None:
            scenario_errors.append("forward speed ratio is unavailable")
        elif speed_ratio < thresholds.min_forward_speed_ratio:
            scenario_errors.append(
                f"forward speed ratio {speed_ratio:.6g} is below "
                f"{thresholds.min_forward_speed_ratio:.6g}"
            )

        reference_stuck = _finite_float(reference.get("stuck_ratio"))
        candidate_stuck = _finite_float(candidate.get("stuck_ratio"))
        stuck_regression = None
        if reference_stuck is None or candidate_stuck is None:
            scenario_errors.append("stuck ratio regression is unavailable")
        else:
            stuck_regression = candidate_stuck - reference_stuck
            if stuck_regression > thresholds.max_stuck_ratio_regression:
                scenario_errors.append(
                    f"stuck ratio regression {stuck_regression:.6g} exceeds "
                    f"{thresholds.max_stuck_ratio_regression:.6g}"
                )

        reference_ok = reference.get("ok") is True
        candidate_ok = candidate.get("ok") is True
        if not reference_ok:
            scenario_errors.append("reference scenario is not passing")
        if not candidate_ok:
            scenario_errors.append("candidate scenario is not passing")

        scenario_results.append(
            M10TeacherScenarioComparison(
                scenario_id=scenario_id,
                reference_ok=reference_ok,
                candidate_ok=candidate_ok,
                reference_forward_distance_m=reference_distance,
                candidate_forward_distance_m=candidate_distance,
                forward_distance_ratio=distance_ratio,
                reference_forward_speed_mps=reference_speed,
                candidate_forward_speed_mps=candidate_speed,
                forward_speed_ratio=speed_ratio,
                reference_stuck_ratio=reference_stuck,
                candidate_stuck_ratio=candidate_stuck,
                stuck_ratio_regression=stuck_regression,
                reference_swing_clearance_p50_m=_finite_float(
                    reference.get("swing_clearance_p50_m")
                ),
                candidate_swing_clearance_p50_m=_finite_float(
                    candidate.get("swing_clearance_p50_m")
                ),
                ok=not scenario_errors,
                errors=scenario_errors,
            )
        )
        errors.extend(f"{scenario_id}: {message}" for message in scenario_errors)

    ok = not errors and len(scenario_results) == len(required_scenarios)
    return M10TeacherComparison(
        ok=ok,
        status="TEACHER_COMPARISON_PASS" if ok else "TEACHER_COMPARISON_FAIL",
        reference_suite=reference_path,
        candidate_suite=candidate_path,
        required_scenarios=list(required_scenarios),
        scenarios=[item.to_dict() for item in scenario_results],
        thresholds=asdict(thresholds),
        reference_summary=reference_summary,
        candidate_summary=candidate_summary,
        errors=errors,
        warnings=warnings,
        output_dir=str(output_dir) if output_dir is not None else None,
    )


def render_markdown(result: M10TeacherComparison) -> str:
    lines = [
        "# Soridormi policy teacher comparison",
        "",
        f"Status: `{result.status}`",
        f"Result: {'PASS' if result.ok else 'FAIL'}",
        f"Reference suite: `{result.reference_suite}`",
        f"Candidate suite: `{result.candidate_suite}`",
        "",
        "## Scenario comparison",
        "",
        "| Scenario | Distance ratio | Speed ratio | Stuck regression | Candidate clearance p50 | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.scenarios:
        lines.append(
            "| {scenario} | {distance} | {speed} | {stuck} | {clearance} | {status} |".format(
                scenario=item["scenario_id"],
                distance=_format_number(item["forward_distance_ratio"]),
                speed=_format_number(item["forward_speed_ratio"]),
                stuck=_format_number(item["stuck_ratio_regression"]),
                clearance=_format_number(item["candidate_swing_clearance_p50_m"], " m"),
                status="PASS" if item["ok"] else "FAIL",
            )
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {item}" for item in result.errors) if result.errors else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in result.warnings) if result.warnings else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _format_number(value: Any, suffix: str = "") -> str:
    parsed = _finite_float(value)
    return "n/a" if parsed is None else f"{parsed:.6g}{suffix}"


def _normalise_csv(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in value.split(",") if part.strip())
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./scripts/compare_policy_teacher_suite.sh",
        description="Compare a candidate scenario suite with the official teacher suite."
    )
    parser.add_argument("reference_suite", type=Path)
    parser.add_argument("candidate_suite", type=Path)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-forward-distance-ratio", type=float, default=0.8)
    parser.add_argument("--min-forward-speed-ratio", type=float, default=0.8)
    parser.add_argument("--max-stuck-ratio-regression", type=float, default=0.1)
    parser.add_argument("--max-candidate-fallen-count", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    required_scenarios = _normalise_csv(args.scenario) or list(DEFAULT_REQUIRED_SCENARIOS)
    result = compare_clearance_teacher_suites(
        _load_json_object(args.reference_suite),
        _load_json_object(args.candidate_suite),
        reference_path=str(args.reference_suite),
        candidate_path=str(args.candidate_suite),
        required_scenarios=required_scenarios,
        thresholds=M10TeacherComparisonThresholds(
            min_forward_distance_ratio=args.min_forward_distance_ratio,
            min_forward_speed_ratio=args.min_forward_speed_ratio,
            max_stuck_ratio_regression=args.max_stuck_ratio_regression,
            max_candidate_fallen_count=args.max_candidate_fallen_count,
        ),
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy_teacher_comparison.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "policy_teacher_comparison.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(result))
    return 1 if args.strict and not result.ok else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
