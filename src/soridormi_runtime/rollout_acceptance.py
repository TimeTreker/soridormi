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
class RolloutAcceptanceThresholds:
    min_policy_records: int = 1
    min_robot_duration: float = 0.0
    max_reset_count: int = 0
    max_action_abs: float | None = 5.0
    max_joint_abs: float | None = None
    min_forward_x: float | None = None
    max_lateral_abs: float | None = None


@dataclass
class RolloutAcceptanceResult:
    ok: bool
    profile_name: str | None
    log_path: str
    generated_at_utc: str
    summary: dict[str, Any]
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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _profile_from_summary(summary: dict[str, Any]) -> str | None:
    latest_command = summary.get("latest_command")
    if isinstance(latest_command, dict):
        raw = latest_command.get("profile") or latest_command.get("policy_profile")
        if raw:
            return str(raw)
    # Policy logs usually do not store profile name yet. Keep this nullable.
    return None


def evaluate_rollout_acceptance(
    log: str | Path,
    *,
    thresholds: RolloutAcceptanceThresholds | None = None,
    profile_name: str | None = None,
) -> RolloutAcceptanceResult:
    log_path = Path(log)
    summary = analyze_policy_log(log_path)
    thresholds = thresholds or RolloutAcceptanceThresholds()
    errors: list[str] = []
    warnings: list[str] = []

    policy_records = int(summary.get("policy_records") or 0)
    if policy_records < thresholds.min_policy_records:
        errors.append(
            f"policy_records {policy_records} is below required minimum "
            f"{thresholds.min_policy_records}"
        )

    robot_duration = _finite_number((summary.get("robot_time") or {}).get("duration"))
    if robot_duration is None:
        if thresholds.min_robot_duration > 0.0:
            errors.append("robot_time duration is unavailable")
        else:
            warnings.append("robot_time duration is unavailable")
    elif robot_duration < thresholds.min_robot_duration:
        errors.append(
            f"robot_time duration {robot_duration:.6g}s is below required minimum "
            f"{thresholds.min_robot_duration:.6g}s"
        )

    reset_count = int((summary.get("reset_cycles") or {}).get("count") or 0)
    if reset_count > thresholds.max_reset_count:
        errors.append(f"reset count {reset_count} exceeds limit {thresholds.max_reset_count}")

    action_abs = _finite_number((summary.get("action") or {}).get("abs_max"))
    if thresholds.max_action_abs is not None:
        if action_abs is None:
            errors.append("action abs_max is unavailable")
        elif action_abs > thresholds.max_action_abs:
            errors.append(
                f"action abs_max {action_abs:.6g} exceeds limit {thresholds.max_action_abs:.6g}"
            )

    joint_abs = _finite_number((summary.get("joint_positions") or {}).get("abs_max"))
    if thresholds.max_joint_abs is not None:
        if joint_abs is None:
            warnings.append("joint position abs_max is unavailable")
        elif joint_abs > thresholds.max_joint_abs:
            errors.append(
                f"joint position abs_max {joint_abs:.6g} exceeds limit "
                f"{thresholds.max_joint_abs:.6g}"
            )

    displacement = summary.get("base_displacement") or {}
    forward_x = _finite_number(displacement.get("forward_x"))
    if thresholds.min_forward_x is not None:
        if forward_x is None:
            errors.append("base forward displacement is unavailable")
        elif forward_x < thresholds.min_forward_x:
            errors.append(
                f"base forward_x {forward_x:.6g}m is below required minimum "
                f"{thresholds.min_forward_x:.6g}m"
            )

    lateral_y = _finite_number(displacement.get("lateral_y"))
    if thresholds.max_lateral_abs is not None:
        if lateral_y is None:
            errors.append("base lateral displacement is unavailable")
        elif abs(lateral_y) > thresholds.max_lateral_abs:
            errors.append(
                f"abs(base lateral_y) {abs(lateral_y):.6g}m exceeds limit "
                f"{thresholds.max_lateral_abs:.6g}m"
            )

    resolved_profile = profile_name or _profile_from_summary(summary)
    return RolloutAcceptanceResult(
        ok=not errors,
        profile_name=resolved_profile,
        log_path=str(log_path),
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        summary=summary,
        thresholds=asdict(thresholds),
        errors=errors,
        warnings=warnings,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any, unit: str = "") -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}{unit}"


def render_rollout_acceptance_report(result: RolloutAcceptanceResult) -> str:
    summary = result.summary
    robot_time = summary.get("robot_time") or {}
    reset_cycles = summary.get("reset_cycles") or {}
    action = summary.get("action") or {}
    joints = summary.get("joint_positions") or {}
    displacement = summary.get("base_displacement") or {}

    lines = [
        "# Soridormi policy rollout acceptance",
        "",
        f"Result: {'PASS' if result.ok else 'FAIL'}",
        f"Profile: {result.profile_name or 'unknown'}",
        f"Log: `{result.log_path}`",
        f"Generated: {result.generated_at_utc}",
        "",
        "## Summary",
        "",
        f"- Policy records: {summary.get('policy_records', 0)}",
        f"- Robot duration: {_fmt(robot_time.get('duration'), ' s')}",
        f"- Reset count: {reset_cycles.get('count', 0)}",
        f"- Action abs max: {_fmt(action.get('abs_max'))}",
        f"- Joint position abs max: {_fmt(joints.get('abs_max'))}",
        f"- Forward displacement x: {_fmt(displacement.get('forward_x'), ' m')}",
        f"- Lateral displacement y: {_fmt(displacement.get('lateral_y'), ' m')}",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in result.thresholds.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Errors", ""])
    if result.errors:
        lines.extend(f"- {item}" for item in result.errors)
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    if result.warnings:
        lines.extend(f"- {item}" for item in result.warnings)
    else:
        lines.append("- none")

    diagnosis = summary.get("diagnosis") or []
    lines.extend(["", "## Log diagnosis", ""])
    if diagnosis:
        lines.extend(f"- {item}" for item in diagnosis)
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def write_rollout_acceptance_outputs(
    result: RolloutAcceptanceResult,
    output_dir: str | Path,
) -> RolloutAcceptanceResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(out)
    _write_json(out / "rollout_acceptance.json", result.to_dict())
    (out / "rollout_acceptance_report.md").write_text(
        render_rollout_acceptance_report(result), encoding="utf-8"
    )
    return result


def _default_output_dir(log: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/policy_rollouts") / f"{log.stem}_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a bounded Soridormi policy rollout log.")
    parser.add_argument("log", type=Path, help="Path to a .mcap or .jsonl runtime log")
    parser.add_argument("--profile", help="Optional profile name to store in the report")
    parser.add_argument("--output-dir", type=Path, help="Output directory for JSON/Markdown reports")
    parser.add_argument("--min-policy-records", type=int, default=1)
    parser.add_argument("--min-robot-duration", type=float, default=0.0)
    parser.add_argument("--max-reset-count", type=int, default=0)
    parser.add_argument("--max-action-abs", type=float, default=5.0)
    parser.add_argument("--disable-action-bound", action="store_true")
    parser.add_argument("--max-joint-abs", type=float)
    parser.add_argument("--min-forward-x", type=float)
    parser.add_argument("--max-lateral-abs", type=float)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = RolloutAcceptanceThresholds(
        min_policy_records=args.min_policy_records,
        min_robot_duration=args.min_robot_duration,
        max_reset_count=args.max_reset_count,
        max_action_abs=None if args.disable_action_bound else args.max_action_abs,
        max_joint_abs=args.max_joint_abs,
        min_forward_x=args.min_forward_x,
        max_lateral_abs=args.max_lateral_abs,
    )
    result = evaluate_rollout_acceptance(
        args.log,
        thresholds=thresholds,
        profile_name=args.profile,
    )
    output_dir = args.output_dir or _default_output_dir(args.log)
    write_rollout_acceptance_outputs(result, output_dir)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_rollout_acceptance_report(result))
        print(f"Wrote rollout acceptance artifacts to: {output_dir}")

    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
