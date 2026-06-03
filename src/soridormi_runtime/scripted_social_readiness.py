"""Readiness reporting for scripted social skill promotion.

This module turns scripted-social acceptance gates into an explicit promotion
report.  It does not mutate the skill manifest.  Promotion remains a human
patch decision, but this report gives a machine-readable checkpoint for moving
skills from ``available_sim_experimental`` to ``available_sim``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scripted_social_acceptance import ScriptedSocialAcceptanceSummary, run_acceptance
from .skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest, skills_by_id


@dataclass(frozen=True)
class ScriptedSocialSkillReadiness:
    skill_id: str
    current_status: str
    category: str
    execution: str
    dry_run_ok: bool
    live_ok: bool | None
    live_executed: bool | None
    fallen: bool | None
    blockers: list[str]
    warnings: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptedSocialReadinessReport:
    ok: bool
    require_live: bool
    candidate_count: int
    ready_count: int
    skills: list[ScriptedSocialSkillReadiness]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "require_live": self.require_live,
            "candidate_count": self.candidate_count,
            "ready_count": self.ready_count,
            "skills": [skill.to_dict() for skill in self.skills],
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _summary_to_dict(summary: ScriptedSocialAcceptanceSummary | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(summary, ScriptedSocialAcceptanceSummary):
        return summary.to_dict()
    return dict(summary)


def _results_by_skill(summary: ScriptedSocialAcceptanceSummary | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if summary is None:
        return {}
    payload = _summary_to_dict(summary)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("acceptance summary results must be a list")
    by_skill: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        skill_id = row.get("skill_id")
        if isinstance(skill_id, str) and skill_id:
            by_skill[skill_id] = dict(row)
    return by_skill


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def build_readiness_report(
    *,
    manifest: Mapping[str, Any],
    dry_acceptance: ScriptedSocialAcceptanceSummary | Mapping[str, Any],
    live_acceptance: ScriptedSocialAcceptanceSummary | Mapping[str, Any] | None = None,
    require_live: bool = False,
) -> ScriptedSocialReadinessReport:
    """Build a promotion-readiness report from acceptance summaries."""

    skill_map = skills_by_id(dict(manifest))
    dry_results = _results_by_skill(dry_acceptance)
    live_results = _results_by_skill(live_acceptance)

    rows: list[ScriptedSocialSkillReadiness] = []
    for skill_id in sorted(dry_results):
        dry = dry_results[skill_id]
        live = live_results.get(skill_id)
        skill = skill_map.get(skill_id, {})
        status = str(skill.get("status", "<missing>"))
        category = str(skill.get("category", "<missing>"))
        execution = str(skill.get("execution", "<missing>"))
        dry_ok = bool(dry.get("ok"))
        live_ok = _bool_or_none(live.get("ok") if live else None)
        live_executed = _bool_or_none(live.get("executed") if live else None)
        fallen = _bool_or_none(live.get("fallen") if live else None)

        blockers: list[str] = []
        warnings: list[str] = []

        if not skill:
            blockers.append("skill is missing from manifest")
        if status != "available_sim_experimental":
            blockers.append(f"current status is {status!r}; expected 'available_sim_experimental' for promotion review")
        if execution != "scripted_keyframe":
            blockers.append(f"execution is {execution!r}; scripted social promotion expects 'scripted_keyframe'")
        if not dry_ok:
            blockers.append("dry-run acceptance failed")
        for warning in dry.get("warnings", []) or []:
            warnings.append(f"dry-run: {warning}")
        for error in dry.get("errors", []) or []:
            blockers.append(f"dry-run: {error}")

        if live is None:
            if require_live:
                blockers.append("live MuJoCo acceptance JSON is required but missing")
            else:
                warnings.append("live MuJoCo acceptance has not been attached; keep experimental until observed")
        else:
            if live_executed is not True:
                blockers.append("live acceptance summary was not executed against MuJoCo")
            if live_ok is not True:
                blockers.append("live MuJoCo acceptance failed")
            if fallen is True:
                blockers.append("live MuJoCo acceptance reported a fall")
            for warning in live.get("warnings", []) or []:
                warnings.append(f"live: {warning}")
            for error in live.get("errors", []) or []:
                blockers.append(f"live: {error}")

        if blockers:
            recommendation = "keep_available_sim_experimental"
        elif live is None:
            recommendation = "dry_run_ready_requires_live_acceptance"
        else:
            recommendation = "candidate_for_available_sim"

        rows.append(
            ScriptedSocialSkillReadiness(
                skill_id=skill_id,
                current_status=status,
                category=category,
                execution=execution,
                dry_run_ok=dry_ok,
                live_ok=live_ok,
                live_executed=live_executed,
                fallen=fallen,
                blockers=blockers,
                warnings=warnings,
                recommendation=recommendation,
            )
        )

    ready_count = sum(1 for row in rows if row.recommendation == "candidate_for_available_sim")
    report_ok = all(not row.blockers for row in rows)
    if require_live:
        report_ok = report_ok and ready_count == len(rows)
    return ScriptedSocialReadinessReport(
        ok=report_ok,
        require_live=require_live,
        candidate_count=len(rows),
        ready_count=ready_count,
        skills=rows,
    )


def render_markdown(report: ScriptedSocialReadinessReport) -> str:
    lines = [
        "# Soridormi scripted social readiness report",
        "",
        f"Overall: **{'PASS' if report.ok else 'NEEDS WORK'}**",
        f"Require live MuJoCo acceptance: `{str(report.require_live).lower()}`",
        f"Candidates ready for `available_sim`: **{report.ready_count}/{report.candidate_count}**",
        "",
        "| Skill | Status | Dry run | Live | Recommendation | Blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.skills:
        live = "n/a" if row.live_ok is None else ("PASS" if row.live_ok else "FAIL")
        blockers = "<br>".join(row.blockers) if row.blockers else "—"
        lines.append(
            "| {skill} | `{status}` | {dry} | {live} | `{rec}` | {blockers} |".format(
                skill=row.skill_id,
                status=row.current_status,
                dry="PASS" if row.dry_run_ok else "FAIL",
                live=live,
                rec=row.recommendation,
                blockers=blockers,
            )
        )
    lines.append("")
    lines.append(
        "Promotion rule: do not edit a skill from `available_sim_experimental` to `available_sim` "
        "unless dry-run acceptance passes and a live MuJoCo acceptance report passes without falls."
    )
    lines.append("")
    return "\n".join(lines)


def write_report_outputs(report: ScriptedSocialReadinessReport, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scripted_social_readiness_report.json"
    md_path = out / "scripted_social_readiness_report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report readiness for scripted social skill promotion.")
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument("--skill", action="append", dest="skills", help="Skill id to include; repeatable.")
    parser.add_argument(
        "--live-acceptance-json",
        help="Optional JSON output from evaluate_scripted_social_skills.sh --execute --json.",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Require live MuJoCo acceptance before reporting overall PASS.",
    )
    parser.add_argument("--output-dir", help="Write JSON and Markdown reports to this directory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Dry-run control frequency.")
    parser.add_argument("--transition-fraction", type=float, default=0.40, help="Dry-run transition fraction.")
    parser.add_argument(
        "--max-head-velocity-radps",
        type=float,
        default=0.35,
        help="Dry-run head velocity limit in rad/s; use 0 to disable.",
    )
    parser.add_argument(
        "--no-auto-stretch-duration",
        action="store_true",
        help="Do not stretch dry-run gestures to satisfy the speed limit.",
    )
    return parser


def _print_human(report: ScriptedSocialReadinessReport) -> None:
    print("Soridormi scripted social readiness")
    print("====================================")
    print(f"Overall: {'PASS' if report.ok else 'NEEDS WORK'}")
    print(f"Ready for available_sim: {report.ready_count}/{report.candidate_count}")
    for row in report.skills:
        print("")
        print(f"{row.skill_id}: {row.recommendation}")
        print(f"  status: {row.current_status}")
        print(f"  dry_run_ok: {row.dry_run_ok}")
        print(f"  live_ok: {row.live_ok}")
        if row.warnings:
            for warning in row.warnings:
                print(f"  warning: {warning}")
        if row.blockers:
            for blocker in row.blockers:
                print(f"  blocker: {blocker}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_skill_manifest(args.manifest)
    dry_acceptance = run_acceptance(
        manifest=args.manifest,
        skill_ids=args.skills,
        execute=False,
        control_hz=args.control_hz,
        transition_fraction=args.transition_fraction,
        max_head_velocity_radps=args.max_head_velocity_radps,
        auto_stretch_duration=not args.no_auto_stretch_duration,
    )
    live_acceptance = _load_json(args.live_acceptance_json) if args.live_acceptance_json else None
    report = build_readiness_report(
        manifest=manifest,
        dry_acceptance=dry_acceptance,
        live_acceptance=live_acceptance,
        require_live=args.require_live,
    )
    if args.output_dir:
        write_report_outputs(report, args.output_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
