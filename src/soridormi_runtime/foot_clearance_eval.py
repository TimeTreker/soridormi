from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FootClearanceThresholds:
    ground_z: float = 0.0
    swing_contact_threshold: float = 0.5
    min_swing_clearance_m: float = 0.015
    target_swing_clearance_m: float = 0.025
    max_low_clearance_ratio: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FootClearanceReport:
    ok: bool
    log_path: str
    sample_count: int
    samples_with_feet: int
    thresholds: dict[str, float]
    left: dict[str, float | int | None]
    right: dict[str, float | int | None]
    combined: dict[str, float | int | None]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_runtime_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if payload.get("type") == "runtime_step" or "state" in payload:
                yield payload


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_m": None,
            "p05_m": None,
            "p50_m": None,
            "mean_m": None,
            "max_m": None,
        }
    return {
        "count": len(values),
        "min_m": float(min(values)),
        "p05_m": _percentile(values, 5.0),
        "p50_m": _percentile(values, 50.0),
        "mean_m": float(sum(values) / len(values)),
        "max_m": float(max(values)),
    }


def _foot_summary(
    all_clearances: list[float],
    swing_clearances: list[float],
    low_swing_steps: int,
) -> dict[str, float | int | None]:
    summary = _stats(all_clearances)
    swing = _stats(swing_clearances)
    swing_count = int(swing["count"] or 0)
    summary.update(
        {
            "swing_count": swing_count,
            "swing_min_m": swing["min_m"],
            "swing_p05_m": swing["p05_m"],
            "swing_p50_m": swing["p50_m"],
            "swing_mean_m": swing["mean_m"],
            "swing_max_m": swing["max_m"],
            "low_clearance_swing_steps": int(low_swing_steps),
            "low_clearance_swing_ratio": (
                float(low_swing_steps) / float(swing_count) if swing_count else None
            ),
        }
    )
    return summary


def evaluate_foot_clearance(
    log_path: str | Path,
    *,
    thresholds: FootClearanceThresholds | None = None,
) -> FootClearanceReport:
    path = Path(log_path)
    cfg = thresholds or FootClearanceThresholds()
    if not path.exists():
        raise FileNotFoundError(path)

    left_all: list[float] = []
    right_all: list[float] = []
    left_swing: list[float] = []
    right_swing: list[float] = []
    left_low = 0
    right_low = 0
    samples = 0
    samples_with_feet = 0
    warnings: list[str] = []
    errors: list[str] = []

    for record in _load_runtime_records(path):
        samples += 1
        state = record.get("state") or {}
        feet = state.get("feet_position_xyz")
        if not isinstance(feet, list) or len(feet) != 2:
            continue
        if not all(isinstance(item, list) and len(item) >= 3 for item in feet):
            continue
        contacts = state.get("feet_contacts") or [0.0, 0.0]
        if not isinstance(contacts, list) or len(contacts) != 2:
            contacts = [0.0, 0.0]

        samples_with_feet += 1
        left_z = _as_float(feet[0][2]) - cfg.ground_z
        right_z = _as_float(feet[1][2]) - cfg.ground_z
        left_all.append(left_z)
        right_all.append(right_z)

        left_contact = _as_float(contacts[0]) >= cfg.swing_contact_threshold
        right_contact = _as_float(contacts[1]) >= cfg.swing_contact_threshold
        if not left_contact:
            left_swing.append(left_z)
            if left_z < cfg.min_swing_clearance_m:
                left_low += 1
        if not right_contact:
            right_swing.append(right_z)
            if right_z < cfg.min_swing_clearance_m:
                right_low += 1

    if samples == 0:
        errors.append("log contains no runtime_step records")
    if samples_with_feet == 0:
        errors.append("log contains no state.feet_position_xyz samples")

    left = _foot_summary(left_all, left_swing, left_low)
    right = _foot_summary(right_all, right_swing, right_low)
    combined_all = left_all + right_all
    combined_swing = left_swing + right_swing
    combined_low = left_low + right_low
    combined = _foot_summary(combined_all, combined_swing, combined_low)

    low_ratio = combined.get("low_clearance_swing_ratio")
    if low_ratio is not None and float(low_ratio) > cfg.max_low_clearance_ratio:
        warnings.append(
            "swing foot clearance is low too often: "
            f"ratio={float(low_ratio):.3f} > max={cfg.max_low_clearance_ratio:.3f}"
        )
    swing_p50 = combined.get("swing_p50_m")
    if swing_p50 is not None and float(swing_p50) < cfg.target_swing_clearance_m:
        warnings.append(
            "median swing clearance is below target: "
            f"p50={float(swing_p50):.4f} m < target={cfg.target_swing_clearance_m:.4f} m"
        )

    return FootClearanceReport(
        ok=not errors,
        log_path=str(path),
        sample_count=samples,
        samples_with_feet=samples_with_feet,
        thresholds=cfg.as_dict(),
        left=left,
        right=right,
        combined=combined,
        warnings=warnings,
        errors=errors,
    )


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.5f}"


def render_markdown(report: FootClearanceReport) -> str:
    lines = [
        "# Soridormi foot-clearance report",
        "",
        f"Result: {'PASS' if report.ok else 'FAILED'}",
        f"Log: {report.log_path}",
        f"Samples: {report.sample_count}",
        f"Samples with feet_position_xyz: {report.samples_with_feet}",
        "",
        "## Thresholds",
        "",
        "| Name | Value |",
        "| --- | ---: |",
    ]
    for key, value in report.thresholds.items():
        lines.append(f"| {key} | {_format_value(value)} |")

    lines.extend(
        [
            "",
            "## Clearance summary",
            "",
            "| Foot | all_min_m | swing_min_m | swing_p05_m | swing_p50_m | swing_mean_m | low_swing_steps | low_swing_ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, summary in (
        ("left", report.left),
        ("right", report.right),
        ("combined", report.combined),
    ):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _format_value(summary.get(key))
                for key in (
                    "min_m",
                    "swing_min_m",
                    "swing_p05_m",
                    "swing_p50_m",
                    "swing_mean_m",
                    "low_clearance_swing_steps",
                    "low_clearance_swing_ratio",
                )
            )
            + " |"
        )

    lines.extend(["", "## Warnings", ""])
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    if report.errors:
        lines.extend(f"- {error}" for error in report.errors)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Soridormi runtime JSONL foot clearance.")
    parser.add_argument("log", type=Path, help="Runtime JSONL log from run_policy_rollout_smoke.sh")
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--min-swing-clearance", type=float, default=0.015)
    parser.add_argument("--target-swing-clearance", type=float, default=0.025)
    parser.add_argument("--max-low-clearance-ratio", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_foot_clearance(
        args.log,
        thresholds=FootClearanceThresholds(
            ground_z=float(args.ground_z),
            min_swing_clearance_m=float(args.min_swing_clearance),
            target_swing_clearance_m=float(args.target_swing_clearance),
            max_low_clearance_ratio=float(args.max_low_clearance_ratio),
        ),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        rendered = render_markdown(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(rendered)
            print(f"Report written: {args.output}")
        else:
            print(rendered)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
