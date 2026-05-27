from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soridormi_runtime.analyze_policy_log import analyze_policy_log


@dataclass(frozen=True)
class RolloutComparisonThresholds:
    """Pass/fail thresholds for comparing a candidate rollout against a reference."""

    min_candidate_policy_records: int = 1
    min_candidate_duration: float = 0.0
    max_candidate_resets: int = 0
    min_forward_ratio: float | None = 0.5
    min_speed_ratio: float | None = None
    max_lateral_abs: float | None = None
    max_lateral_ratio: float | None = 2.0
    max_action_abs: float | None = 5.0


@dataclass
class RolloutComparisonResult:
    ok: bool
    reference_log: str
    candidate_log: str
    generated_at_utc: str
    reference: dict[str, Any]
    candidate: dict[str, Any]
    comparison: dict[str, Any]
    thresholds: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    robot_time = summary.get("robot_time") or {}
    resets = summary.get("reset_cycles") or {}
    action = summary.get("action") or {}
    displacement = summary.get("base_displacement") or {}

    duration = _finite_number(robot_time.get("duration"))
    forward_x = _finite_number(displacement.get("forward_x"))
    lateral_y = _finite_number(displacement.get("lateral_y"))

    forward_speed = None
    lateral_speed = None
    if duration is not None and duration > 1e-12:
        if forward_x is not None:
            forward_speed = forward_x / duration
        if lateral_y is not None:
            lateral_speed = lateral_y / duration

    return {
        "path": summary.get("path"),
        "policy_records": int(summary.get("policy_records") or 0),
        "robot_duration": duration,
        "reset_count": int(resets.get("count") or 0),
        "action_abs_max": _finite_number(action.get("abs_max")),
        "forward_x": forward_x,
        "lateral_y": lateral_y,
        "lateral_abs": abs(lateral_y) if lateral_y is not None else None,
        "forward_speed": forward_speed,
        "lateral_speed": lateral_speed,
        "diagnosis": list(summary.get("diagnosis") or []),
    }


def compare_rollout_summaries(
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    thresholds: RolloutComparisonThresholds | None = None,
    reference_log: str | None = None,
    candidate_log: str | None = None,
) -> RolloutComparisonResult:
    """Compare analyzed reference/candidate rollout summaries.

    The reference is usually the trusted teacher policy. The candidate is usually
    a newly trained replacement profile. This is intentionally based on actual
    rollout behavior instead of offline action error.
    """

    thresholds = thresholds or RolloutComparisonThresholds()
    ref = _metric_view(reference_summary)
    cand = _metric_view(candidate_summary)

    comparison = {
        "duration_ratio": _safe_ratio(cand["robot_duration"], ref["robot_duration"]),
        "forward_ratio": _safe_ratio(cand["forward_x"], ref["forward_x"]),
        "forward_speed_ratio": _safe_ratio(cand["forward_speed"], ref["forward_speed"]),
        "lateral_abs_ratio": _safe_ratio(cand["lateral_abs"], ref["lateral_abs"]),
        "action_abs_ratio": _safe_ratio(cand["action_abs_max"], ref["action_abs_max"]),
    }

    errors: list[str] = []
    warnings: list[str] = []

    if cand["policy_records"] < thresholds.min_candidate_policy_records:
        errors.append(
            f"candidate policy_records {cand['policy_records']} is below required minimum "
            f"{thresholds.min_candidate_policy_records}"
        )

    cand_duration = _finite_number(cand["robot_duration"])
    if cand_duration is None:
        errors.append("candidate robot duration is unavailable")
    elif cand_duration < thresholds.min_candidate_duration:
        errors.append(
            f"candidate robot duration {cand_duration:.6g}s is below required minimum "
            f"{thresholds.min_candidate_duration:.6g}s"
        )

    if cand["reset_count"] > thresholds.max_candidate_resets:
        errors.append(
            f"candidate reset count {cand['reset_count']} exceeds limit "
            f"{thresholds.max_candidate_resets}"
        )

    if thresholds.max_action_abs is not None:
        action_abs = _finite_number(cand["action_abs_max"])
        if action_abs is None:
            errors.append("candidate action abs_max is unavailable")
        elif action_abs > thresholds.max_action_abs:
            errors.append(
                f"candidate action abs_max {action_abs:.6g} exceeds limit "
                f"{thresholds.max_action_abs:.6g}"
            )

    if thresholds.min_forward_ratio is not None:
        ratio = _finite_number(comparison["forward_ratio"])
        if ratio is None:
            warnings.append("forward ratio is unavailable; reference or candidate forward_x is missing/zero")
        elif ratio < thresholds.min_forward_ratio:
            errors.append(
                f"candidate forward ratio {ratio:.6g} is below required minimum "
                f"{thresholds.min_forward_ratio:.6g}"
            )

    if thresholds.min_speed_ratio is not None:
        ratio = _finite_number(comparison["forward_speed_ratio"])
        if ratio is None:
            warnings.append("forward speed ratio is unavailable")
        elif ratio < thresholds.min_speed_ratio:
            errors.append(
                f"candidate forward speed ratio {ratio:.6g} is below required minimum "
                f"{thresholds.min_speed_ratio:.6g}"
            )

    if thresholds.max_lateral_abs is not None:
        lateral_abs = _finite_number(cand["lateral_abs"])
        if lateral_abs is None:
            warnings.append("candidate lateral displacement is unavailable")
        elif lateral_abs > thresholds.max_lateral_abs:
            errors.append(
                f"candidate lateral abs {lateral_abs:.6g}m exceeds limit "
                f"{thresholds.max_lateral_abs:.6g}m"
            )

    if thresholds.max_lateral_ratio is not None:
        ratio = _finite_number(comparison["lateral_abs_ratio"])
        if ratio is None:
            # A near-zero teacher lateral drift is common; absolute threshold is
            # the more useful guard in that case.
            warnings.append("lateral ratio is unavailable; use --max-lateral-abs for near-zero teacher drift")
        elif ratio > thresholds.max_lateral_ratio:
            errors.append(
                f"candidate lateral abs ratio {ratio:.6g} exceeds limit "
                f"{thresholds.max_lateral_ratio:.6g}"
            )

    return RolloutComparisonResult(
        ok=not errors,
        reference_log=reference_log or str(reference_summary.get("path") or "<summary>"),
        candidate_log=candidate_log or str(candidate_summary.get("path") or "<summary>"),
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        reference=ref,
        candidate=cand,
        comparison=comparison,
        thresholds=asdict(thresholds),
        errors=errors,
        warnings=warnings,
    )


def compare_rollout_logs(
    reference_log: str | Path,
    candidate_log: str | Path,
    *,
    thresholds: RolloutComparisonThresholds | None = None,
) -> RolloutComparisonResult:
    reference_path = Path(reference_log)
    candidate_path = Path(candidate_log)
    return compare_rollout_summaries(
        analyze_policy_log(reference_path),
        analyze_policy_log(candidate_path),
        thresholds=thresholds,
        reference_log=str(reference_path),
        candidate_log=str(candidate_path),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any, unit: str = "") -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}{unit}"


def render_rollout_comparison_report(result: RolloutComparisonResult) -> str:
    ref = result.reference
    cand = result.candidate
    comp = result.comparison

    lines = [
        "# Soridormi policy rollout comparison",
        "",
        f"Result: {'PASS' if result.ok else 'FAIL'}",
        f"Reference log: `{result.reference_log}`",
        f"Candidate log: `{result.candidate_log}`",
        f"Generated: {result.generated_at_utc}",
        "",
        "## Core rollout metrics",
        "",
        "| Metric | Reference | Candidate | Candidate / reference |",
        "| --- | ---: | ---: | ---: |",
        f"| Policy records | {ref['policy_records']} | {cand['policy_records']} | n/a |",
        f"| Robot duration | {_fmt(ref['robot_duration'], ' s')} | {_fmt(cand['robot_duration'], ' s')} | {_fmt(comp['duration_ratio'])} |",
        f"| Reset count | {ref['reset_count']} | {cand['reset_count']} | n/a |",
        f"| Forward displacement x | {_fmt(ref['forward_x'], ' m')} | {_fmt(cand['forward_x'], ' m')} | {_fmt(comp['forward_ratio'])} |",
        f"| Forward speed | {_fmt(ref['forward_speed'], ' m/s')} | {_fmt(cand['forward_speed'], ' m/s')} | {_fmt(comp['forward_speed_ratio'])} |",
        f"| Lateral abs | {_fmt(ref['lateral_abs'], ' m')} | {_fmt(cand['lateral_abs'], ' m')} | {_fmt(comp['lateral_abs_ratio'])} |",
        f"| Action abs max | {_fmt(ref['action_abs_max'])} | {_fmt(cand['action_abs_max'])} | {_fmt(comp['action_abs_ratio'])} |",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in result.thresholds.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {item}" for item in result.errors) if result.errors else lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in result.warnings) if result.warnings else lines.append("- none")

    lines.extend(["", "## Reference diagnosis", ""])
    lines.extend(f"- {item}" for item in ref.get("diagnosis") or []) or lines.append("- none")

    lines.extend(["", "## Candidate diagnosis", ""])
    lines.extend(f"- {item}" for item in cand.get("diagnosis") or []) or lines.append("- none")

    return "\n".join(lines) + "\n"


def write_rollout_comparison_outputs(
    result: RolloutComparisonResult,
    output_dir: str | Path,
) -> RolloutComparisonResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(out)
    _write_json(out / "rollout_comparison.json", result.to_dict())
    (out / "rollout_comparison_report.md").write_text(
        render_rollout_comparison_report(result),
        encoding="utf-8",
    )
    return result


def _default_output_dir(candidate_log: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/data/policy_rollout_comparisons") / f"{candidate_log.stem}_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate policy rollout against a reference rollout."
    )
    parser.add_argument("reference_log", type=Path, help="Teacher/reference .mcap or .jsonl rollout log")
    parser.add_argument("candidate_log", type=Path, help="Candidate .mcap or .jsonl rollout log")
    parser.add_argument("--output-dir", type=Path, help="Output directory for comparison artifacts")
    parser.add_argument("--min-candidate-policy-records", type=int, default=1)
    parser.add_argument("--min-candidate-duration", type=float, default=0.0)
    parser.add_argument("--max-candidate-resets", type=int, default=0)
    parser.add_argument("--min-forward-ratio", type=float, default=0.5)
    parser.add_argument("--disable-forward-ratio", action="store_true")
    parser.add_argument("--min-speed-ratio", type=float)
    parser.add_argument("--max-lateral-abs", type=float)
    parser.add_argument("--max-lateral-ratio", type=float, default=2.0)
    parser.add_argument("--disable-lateral-ratio", action="store_true")
    parser.add_argument("--max-action-abs", type=float, default=5.0)
    parser.add_argument("--disable-action-bound", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = RolloutComparisonThresholds(
        min_candidate_policy_records=args.min_candidate_policy_records,
        min_candidate_duration=args.min_candidate_duration,
        max_candidate_resets=args.max_candidate_resets,
        min_forward_ratio=None if args.disable_forward_ratio else args.min_forward_ratio,
        min_speed_ratio=args.min_speed_ratio,
        max_lateral_abs=args.max_lateral_abs,
        max_lateral_ratio=None if args.disable_lateral_ratio else args.max_lateral_ratio,
        max_action_abs=None if args.disable_action_bound else args.max_action_abs,
    )
    result = compare_rollout_logs(
        args.reference_log,
        args.candidate_log,
        thresholds=thresholds,
    )
    output_dir = args.output_dir or _default_output_dir(args.candidate_log)
    write_rollout_comparison_outputs(result, output_dir)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_rollout_comparison_report(result))
        print(f"Wrote rollout comparison artifacts to: {output_dir}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
