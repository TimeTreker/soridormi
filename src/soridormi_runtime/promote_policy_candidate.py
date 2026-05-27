"""Promote an offline-evaluated Soridormi policy candidate to a named runtime profile.

M6.9 is the controlled bridge from offline leaderboard artifacts to a runtime
profile YAML that can be selected by ``run_policy_experiment.sh``.  Promotion is
intentionally explicit and auditable: it copies an existing candidate profile,
adds promotion metadata, writes a promotion record/report, and never launches
simulation.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_profiles import PolicyProfile

PROMOTION_SCHEMA_VERSION = 1
DEFAULT_RECORDS_ROOT = Path("data/policy_promotions")
DEFAULT_PROFILE_OUTPUT_DIR = Path("configs/policies")


@dataclass
class PolicyCandidatePromotionResult:
    ok: bool
    generated_at_utc: str
    target_profile: str
    target_profile_path: str
    source_profile: str
    source_profile_path: str
    leaderboard_path: str
    evaluation_path: str | None
    promotion_record_path: str
    promotion_report_path: str
    promotable: bool
    candidate_rank: int | None = None
    model_kind: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_max_abs_error: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = PROMOTION_SCHEMA_VERSION
        payload["promotion_type"] = "soridormi.policy_candidate_promotion.v1"
        return payload


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_leaderboard_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "candidate_leaderboard.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate leaderboard not found: {candidate}")
    return candidate


def _candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("Leaderboard is missing candidates list")
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def select_candidate(payload: dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    candidates = _candidates(payload)
    if profile:
        for candidate in candidates:
            if str(candidate.get("profile_name")) == profile:
                return candidate
        raise ValueError(f"Candidate profile {profile!r} not found in leaderboard")
    for candidate in candidates:
        if bool(candidate.get("promotable")):
            return candidate
    if candidates:
        return candidates[0]
    raise ValueError("Leaderboard has no candidates")


def _safe_metric(candidate: dict[str, Any], key: str) -> float | None:
    value = candidate.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_profile_from_candidate(candidate: dict[str, Any], source_profile: str | Path | None = None) -> PolicyProfile:
    if source_profile is not None:
        return PolicyProfile.load(source_profile)
    profile_path = candidate.get("profile_path")
    if isinstance(profile_path, str) and profile_path.strip():
        path = Path(profile_path)
        if path.exists():
            return PolicyProfile.load(path)
    profile_name = str(candidate.get("profile_name") or "").strip()
    if not profile_name:
        raise ValueError("Candidate does not include profile_name; pass --source-profile")
    return PolicyProfile.load(profile_name)


def _promotion_payload(
    source: PolicyProfile,
    *,
    target_profile: str,
    candidate: dict[str, Any],
    leaderboard_path: Path,
    description: str | None,
    generated_at_utc: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(source.payload)
    payload["name"] = target_profile
    payload["description"] = description or f"Promoted policy candidate from {source.name}."

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(
        {
            "generated_by": "soridormi_m69_policy_candidate_promotion",
            "promoted_from_profile": source.name,
            "promotion_generated_at_utc": generated_at_utc,
            "promotion_leaderboard_path": str(leaderboard_path),
            "promotion_evaluation_path": str(candidate.get("evaluation_path") or ""),
            "promotion_candidate_rank": candidate.get("rank"),
            "promotion_candidate_promotable": bool(candidate.get("promotable")),
            "promotion_model_sha256": candidate.get("model_sha256"),
            "promotion_metrics": {
                "test_mae": _safe_metric(candidate, "test_mae"),
                "test_rmse": _safe_metric(candidate, "test_rmse"),
                "test_max_abs_error": _safe_metric(candidate, "test_max_abs_error"),
            },
        }
    )
    payload["metadata"] = metadata

    logging = payload.get("logging")
    if not isinstance(logging, dict):
        logging = {}
    logging["prefix"] = f"policy_{target_profile}"
    payload["logging"] = logging
    return payload


def _write_report(path: Path, result: PolicyCandidatePromotionResult) -> None:
    lines = [
        "# Soridormi policy candidate promotion",
        "",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        f"Target profile: `{result.target_profile}`",
        f"Target YAML: `{result.target_profile_path}`",
        f"Source profile: `{result.source_profile}`",
        f"Leaderboard: `{result.leaderboard_path}`",
        f"Evaluation: `{result.evaluation_path or 'n/a'}`",
        "",
        "## Candidate",
        "",
        f"- Rank: {result.candidate_rank if result.candidate_rank is not None else 'n/a'}",
        f"- Promotable: {result.promotable}",
        f"- Model kind: `{result.model_kind or 'n/a'}`",
        f"- Model path: `{result.model_path or 'n/a'}`",
        f"- Model SHA256: `{result.model_sha256 or 'n/a'}`",
        f"- Test MAE: {result.test_mae if result.test_mae is not None else 'n/a'}",
        f"- Test RMSE: {result.test_rmse if result.test_rmse is not None else 'n/a'}",
        f"- Test max abs error: {result.test_max_abs_error if result.test_max_abs_error is not None else 'n/a'}",
    ]
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_policy_candidate(
    leaderboard: str | Path,
    *,
    target_profile: str,
    source_profile: str | Path | None = None,
    candidate_profile: str | None = None,
    output_dir: str | Path = DEFAULT_PROFILE_OUTPUT_DIR,
    records_dir: str | Path = DEFAULT_RECORDS_ROOT,
    robot_config_path: str | Path | None = None,
    description: str | None = None,
    force: bool = False,
    allow_non_promotable: bool = False,
) -> PolicyCandidatePromotionResult:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = utc_stamp()
    errors: list[str] = []
    warnings: list[str] = []
    leaderboard_path = resolve_leaderboard_path(leaderboard)
    leaderboard_payload = _load_json(leaderboard_path)
    candidate = select_candidate(leaderboard_payload, profile=candidate_profile)
    promotable = bool(candidate.get("promotable"))
    if not promotable and not allow_non_promotable:
        errors.append("Selected candidate is not promotable; pass --allow-non-promotable to override")

    source = _load_profile_from_candidate(candidate, source_profile=source_profile)
    output_path = Path(output_dir) / f"{target_profile}.yaml"
    if output_path.exists() and not force:
        errors.append(f"Target profile already exists: {output_path}. Pass --force to overwrite.")

    payload = _promotion_payload(
        source,
        target_profile=target_profile,
        candidate=candidate,
        leaderboard_path=leaderboard_path,
        description=description,
        generated_at_utc=generated,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_dir = Path(records_dir) / f"{target_profile}_{stamp}"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "promotion_record.json"
    report_path = record_dir / "promotion_report.md"

    # Validate the promoted payload before writing when possible.  Use a temp file
    # in the record directory so PolicyProfile/contract logic sees the exact YAML.
    temp_profile = record_dir / f"{target_profile}.yaml"
    temp_profile.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:
        contract = build_policy_contract(PolicyProfile.load(temp_profile), robot_config_path=robot_config_path)
    except Exception as exc:  # pragma: no cover - defensive guard for malformed local profiles
        errors.append(f"Promoted profile contract check failed: {exc!r}")
    else:
        if not contract.ok:
            errors.extend(f"contract: {error}" for error in contract.errors)
        warnings.extend(f"contract: {warning}" for warning in contract.warnings)

    result = PolicyCandidatePromotionResult(
        ok=not errors,
        generated_at_utc=generated,
        target_profile=target_profile,
        target_profile_path=str(output_path),
        source_profile=source.name,
        source_profile_path=str(source.path),
        leaderboard_path=str(leaderboard_path),
        evaluation_path=str(candidate.get("evaluation_path")) if candidate.get("evaluation_path") is not None else None,
        promotion_record_path=str(record_path),
        promotion_report_path=str(report_path),
        promotable=promotable,
        candidate_rank=int(candidate["rank"]) if isinstance(candidate.get("rank"), int) else None,
        model_kind=str(candidate.get("model_kind")) if candidate.get("model_kind") is not None else None,
        model_path=str(candidate.get("model_path")) if candidate.get("model_path") is not None else None,
        model_sha256=str(candidate.get("model_sha256")) if candidate.get("model_sha256") is not None else None,
        test_mae=_safe_metric(candidate, "test_mae"),
        test_rmse=_safe_metric(candidate, "test_rmse"),
        test_max_abs_error=_safe_metric(candidate, "test_max_abs_error"),
        errors=errors,
        warnings=warnings,
    )

    if result.ok:
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    record_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def print_promotion_summary(result: PolicyCandidatePromotionResult) -> None:
    print("Soridormi policy candidate promotion")
    print("====================================")
    print(f"Target profile: {result.target_profile}")
    print(f"Target YAML: {result.target_profile_path}")
    print(f"Source profile: {result.source_profile}")
    print(f"Leaderboard: {result.leaderboard_path}")
    print(f"Record: {result.promotion_record_path}")
    print(f"Report: {result.promotion_report_path}")
    print(f"Promotable: {result.promotable}")
    if result.test_mae is not None:
        print(f"Test MAE: {result.test_mae:.6g}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an offline-ranked policy candidate to a named runtime profile.")
    parser.add_argument("leaderboard", type=Path, help="candidate_leaderboard.json or directory containing it")
    parser.add_argument("--target-profile", required=True, help="Name for the promoted runtime profile YAML")
    parser.add_argument("--profile", dest="candidate_profile", default=None, help="Specific candidate profile to promote; defaults to best promotable")
    parser.add_argument("--source-profile", default=None, help="Explicit source profile YAML/name if the leaderboard does not include profile_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROFILE_OUTPUT_DIR, help="Directory for promoted profile YAML")
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_ROOT, help="Directory for promotion records/reports")
    parser.add_argument("--robot-config", default=None, help="Robot config used for promoted profile contract validation")
    parser.add_argument("--description", default=None, help="Description to write into the promoted profile")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target profile YAML")
    parser.add_argument("--allow-non-promotable", action="store_true", help="Allow promotion of a non-promotable candidate")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    result = promote_policy_candidate(
        args.leaderboard,
        target_profile=args.target_profile,
        source_profile=args.source_profile,
        candidate_profile=args.candidate_profile,
        output_dir=args.output_dir,
        records_dir=args.records_dir,
        robot_config_path=args.robot_config,
        description=args.description,
        force=args.force,
        allow_non_promotable=args.allow_non_promotable,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print_promotion_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
