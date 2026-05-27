from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutDiagnosisThresholds:
    """Human-facing thresholds for diagnosing a candidate rollout failure."""

    min_duration_ratio: float = 0.8
    min_forward_ratio: float = 0.7
    min_speed_ratio: float = 0.7
    max_lateral_abs: float = 0.25
    max_lateral_ratio: float = 3.0
    max_action_abs: float = 5.0
    max_action_ratio: float = 2.5
    max_reset_count: int = 0


@dataclass
class RolloutFailureDiagnosis:
    ok: bool
    generated_at_utc: str
    source: str
    summary: str
    primary_failure_modes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_comparison(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected comparison JSON object in {path}")
    return payload


def diagnose_rollout_comparison(
    comparison_payload: dict[str, Any],
    *,
    thresholds: RolloutDiagnosisThresholds | None = None,
    source: str = "<comparison>",
) -> RolloutFailureDiagnosis:
    """Diagnose why a trained policy rollout underperformed a teacher rollout.

    Input is the JSON object written by ``compare_policy_rollouts.sh``. The goal
    is not another pass/fail gate; it is to point the next project step at the
    most likely core issue: stability, forward progression, drift, or policy
    output saturation.
    """

    thresholds = thresholds or RolloutDiagnosisThresholds()
    candidate = comparison_payload.get("candidate") or {}
    comparison = comparison_payload.get("comparison") or {}
    errors = list(comparison_payload.get("errors") or [])
    warnings = list(comparison_payload.get("warnings") or [])

    duration_ratio = _finite(comparison.get("duration_ratio"))
    forward_ratio = _finite(comparison.get("forward_ratio"))
    speed_ratio = _finite(comparison.get("forward_speed_ratio"))
    lateral_abs = _finite(candidate.get("lateral_abs"))
    lateral_ratio = _finite(comparison.get("lateral_abs_ratio"))
    action_abs = _finite(candidate.get("action_abs_max"))
    action_ratio = _finite(comparison.get("action_abs_ratio"))
    reset_count = int(candidate.get("reset_count") or 0)
    policy_records = int(candidate.get("policy_records") or 0)

    evidence = {
        "candidate_policy_records": policy_records,
        "candidate_reset_count": reset_count,
        "duration_ratio": duration_ratio,
        "forward_ratio": forward_ratio,
        "forward_speed_ratio": speed_ratio,
        "candidate_lateral_abs": lateral_abs,
        "lateral_abs_ratio": lateral_ratio,
        "candidate_action_abs_max": action_abs,
        "action_abs_ratio": action_ratio,
        "comparison_errors": errors,
    }

    modes: list[str] = []
    next_steps: list[str] = []

    if reset_count > thresholds.max_reset_count:
        modes.append("stability_or_fall")
        next_steps.append(
            "Inspect the candidate rollout around the first reset/fall; compare base pose, contacts, and action magnitude before retraining."
        )

    if duration_ratio is not None and duration_ratio < thresholds.min_duration_ratio:
        modes.append("early_termination")
        next_steps.append(
            "Run a shorter bounded rollout and inspect the last valid seconds; early termination usually hides the real failure mode."
        )

    if forward_ratio is not None and forward_ratio < thresholds.min_forward_ratio:
        modes.append("weak_forward_locomotion")
        next_steps.append(
            "Increase teacher data coverage for the target forward command and retrain; prioritize loss on hip/knee/ankle action dimensions."
        )

    if speed_ratio is not None and speed_ratio < thresholds.min_speed_ratio:
        if "weak_forward_locomotion" not in modes:
            modes.append("slow_forward_velocity_tracking")
        next_steps.append(
            "Compare commanded velocity against achieved velocity; check whether the candidate under-amplifies actions after normalization/export."
        )

    if lateral_abs is not None and lateral_abs > thresholds.max_lateral_abs:
        modes.append("lateral_drift")
        next_steps.append(
            "Add lateral/yaw drift metrics to training selection; collect more balanced straight-walk teacher segments."
        )

    if lateral_ratio is not None and lateral_ratio > thresholds.max_lateral_ratio:
        if "lateral_drift" not in modes:
            modes.append("relative_lateral_drift")
        next_steps.append(
            "Teacher drift is much lower than candidate drift; inspect left/right action symmetry and foot contact timing."
        )

    if action_abs is not None and action_abs > thresholds.max_action_abs:
        modes.append("action_saturation")
        next_steps.append(
            "Action magnitude is too large; check ONNX normalization, action-scale assumptions, and add output clipping/loss regularization during training."
        )

    if action_ratio is not None and action_ratio > thresholds.max_action_ratio:
        if "action_saturation" not in modes:
            modes.append("action_amplification")
        next_steps.append(
            "Candidate actions are amplified relative to teacher; compare offline predictions on the same observations before running another rollout."
        )

    if errors and not modes:
        modes.append("threshold_failure")
        next_steps.append("Open the rollout comparison report and fix the first failed threshold before changing training code.")

    if not modes:
        modes.append("no_major_rollout_failure_detected")
        next_steps.append("Promote this candidate to longer rollout testing; increase duration/command diversity before hardware work.")

    modes = list(dict.fromkeys(modes))
    next_steps = list(dict.fromkeys(next_steps))

    ok = modes == ["no_major_rollout_failure_detected"] and not errors
    summary = _summary_sentence(modes)

    return RolloutFailureDiagnosis(
        ok=ok,
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        source=source,
        summary=summary,
        primary_failure_modes=modes,
        evidence=evidence,
        recommended_next_steps=next_steps,
        warnings=warnings,
    )


def _summary_sentence(modes: list[str]) -> str:
    if modes == ["no_major_rollout_failure_detected"]:
        return "Candidate rollout looks healthy under the configured diagnostic thresholds."
    labels = {
        "stability_or_fall": "fall/reset stability",
        "early_termination": "early termination",
        "weak_forward_locomotion": "weak forward locomotion",
        "slow_forward_velocity_tracking": "slow velocity tracking",
        "lateral_drift": "lateral drift",
        "relative_lateral_drift": "relative lateral drift",
        "action_saturation": "action saturation",
        "action_amplification": "action amplification",
        "threshold_failure": "generic threshold failure",
    }
    readable = [labels.get(mode, mode) for mode in modes]
    return "Candidate rollout needs work: " + ", ".join(readable) + "."


def render_rollout_failure_diagnosis_report(result: RolloutFailureDiagnosis) -> str:
    lines = [
        "# Soridormi rollout failure diagnosis",
        "",
        f"Result: {'PASS' if result.ok else 'NEEDS WORK'}",
        f"Source: `{result.source}`",
        f"Generated: {result.generated_at_utc}",
        "",
        "## Summary",
        "",
        result.summary,
        "",
        "## Primary failure modes",
        "",
    ]
    lines.extend(f"- {mode}" for mode in result.primary_failure_modes)
    lines.extend(["", "## Evidence", ""])
    for key, value in result.evidence.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommended next steps", ""])
    lines.extend(f"- {step}" for step in result.recommended_next_steps)
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in result.warnings) if result.warnings else lines.append("- none")
    return "\n".join(lines) + "\n"


def write_rollout_failure_diagnosis_outputs(
    result: RolloutFailureDiagnosis,
    output_dir: str | Path,
) -> RolloutFailureDiagnosis:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout_failure_diagnosis.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "rollout_failure_diagnosis_report.md").write_text(
        render_rollout_failure_diagnosis_report(result),
        encoding="utf-8",
    )
    return result


def _default_output_dir(comparison_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/data/policy_rollout_diagnoses") / f"{comparison_path.stem}_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a candidate policy rollout comparison failure.")
    parser.add_argument("comparison_json", type=Path, help="rollout_comparison.json from compare_policy_rollouts.sh")
    parser.add_argument("--output-dir", type=Path, help="Output directory for diagnosis artifacts")
    parser.add_argument("--min-duration-ratio", type=float, default=0.8)
    parser.add_argument("--min-forward-ratio", type=float, default=0.7)
    parser.add_argument("--min-speed-ratio", type=float, default=0.7)
    parser.add_argument("--max-lateral-abs", type=float, default=0.25)
    parser.add_argument("--max-lateral-ratio", type=float, default=3.0)
    parser.add_argument("--max-action-abs", type=float, default=5.0)
    parser.add_argument("--max-action-ratio", type=float, default=2.5)
    parser.add_argument("--max-reset-count", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = RolloutDiagnosisThresholds(
        min_duration_ratio=args.min_duration_ratio,
        min_forward_ratio=args.min_forward_ratio,
        min_speed_ratio=args.min_speed_ratio,
        max_lateral_abs=args.max_lateral_abs,
        max_lateral_ratio=args.max_lateral_ratio,
        max_action_abs=args.max_action_abs,
        max_action_ratio=args.max_action_ratio,
        max_reset_count=args.max_reset_count,
    )
    payload = _load_comparison(args.comparison_json)
    result = diagnose_rollout_comparison(payload, thresholds=thresholds, source=str(args.comparison_json))
    out_dir = args.output_dir or _default_output_dir(args.comparison_json)
    write_rollout_failure_diagnosis_outputs(result, out_dir)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Soridormi rollout failure diagnosis")
        print("====================================")
        print(f"Source: {args.comparison_json}")
        print(f"Output: {out_dir}")
        print(f"Summary: {result.summary}")
        print(f"Result: {'PASS' if result.ok else 'NEEDS WORK'}")
        if result.primary_failure_modes:
            print("Primary modes:")
            for mode in result.primary_failure_modes:
                print(f"  - {mode}")

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
