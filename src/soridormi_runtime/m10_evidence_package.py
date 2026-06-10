from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.m10_clearance_readiness import DEFAULT_PROFILE, DEFAULT_REQUIRED_SCENARIOS
from soridormi_runtime.scenario_curriculum import DEFAULT_SCENARIO_MANIFEST

DEFAULT_OUTPUT_ROOT = Path("artifacts/m10_evidence")
DEFAULT_READINESS_ROOT = Path("artifacts/m10_clearance_readiness")
DEFAULT_VISUAL_ROOT = Path("artifacts/m10_visual_inspection")
DEFAULT_REVIEW_NAME = "m10_visual_review.json"
VISUAL_REVIEW_FIELDS = (
    "foot_clearance",
    "base_stability",
    "toe_drag",
    "command_transition_quality",
)
PASS_VALUE = "PASS"
FAIL_VALUES = {"FAIL", "UNCLEAR", "MISSING"}


@dataclass(frozen=True)
class M10EvidenceScenario:
    scenario_id: str
    rollout_report: str | None
    rollout_report_present: bool
    clearance_status: str | None
    clearance_ok: bool | None
    visual_review: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M10EvidencePackage:
    ok: bool
    profile: str
    status: str
    output_dir: str
    scenario_manifest: str
    scenarios: list[dict[str, Any]]
    readiness_report: str
    readiness_ok: bool
    readiness_status: str | None
    visual_plan: str
    visual_plan_ok: bool
    visual_plan_status: str | None
    visual_review: str | None
    visual_review_ok: bool | None
    visual_review_status: str | None
    review_template_json: str
    review_template_markdown: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    commands: dict[str, list[str]] = field(default_factory=dict)

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


def _load_json_if_present(path: str | Path) -> tuple[Mapping[str, Any] | None, str | None]:
    resolved = Path(path)
    if not resolved.exists():
        return None, f"missing JSON artifact: {resolved}"
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as exc:
        return None, f"could not load JSON artifact {resolved}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"JSON artifact must contain an object: {resolved}"
    return payload, None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _default_readiness_report(profile: str) -> Path:
    return DEFAULT_READINESS_ROOT / profile / "m10_clearance_readiness.json"


def _default_visual_plan(profile: str) -> Path:
    return DEFAULT_VISUAL_ROOT / profile / "m10_visual_inspection_plan.json"


def _default_visual_review(output_dir: Path) -> Path:
    return output_dir / DEFAULT_REVIEW_NAME


def _scenario_map_from_readiness(readiness: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if readiness is None:
        return {}
    scenarios = readiness.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, Mapping):
            continue
        scenario_id = str(item.get("scenario_id") or "").strip()
        if scenario_id:
            out[scenario_id] = item
    return out


def _scenario_map_from_review(review: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if review is None:
        return {}
    scenarios = review.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, Mapping):
            continue
        scenario_id = str(item.get("scenario_id") or "").strip()
        if scenario_id:
            out[scenario_id] = item
    return out


def _review_field_value(item: Mapping[str, Any], key: str) -> str:
    return str(item.get(key) or "MISSING").strip().upper()


def _evaluate_visual_review(
    review: Mapping[str, Any] | None,
    *,
    scenarios: Sequence[str],
) -> tuple[bool | None, str | None, list[str]]:
    if review is None:
        return None, None, []
    scenario_reviews = _scenario_map_from_review(review)
    blockers: list[str] = []
    for scenario_id in scenarios:
        item = scenario_reviews.get(scenario_id)
        if item is None:
            blockers.append(f"{scenario_id}: visual review is missing")
            continue
        for key in VISUAL_REVIEW_FIELDS:
            value = _review_field_value(item, key)
            if value != PASS_VALUE:
                blockers.append(f"{scenario_id}: visual review {key} is {value}")
    explicit_status = str(review.get("status") or "").strip().upper()
    if explicit_status and explicit_status not in {"VISUAL_PASS", "PASS"}:
        blockers.append(f"visual review status is {explicit_status}")
    ok = not blockers
    return ok, "VISUAL_PASS" if ok else "VISUAL_BLOCKED", blockers


def build_visual_review_template(
    *,
    profile: str,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    return {
        "profile": profile,
        "status": "UNCLEAR",
        "reviewer": "",
        "reviewed_at": "",
        "instructions": [
            "Fill PASS/FAIL/UNCLEAR for each field after follow-camera visual inspection.",
            "Do not mark VISUAL_PASS unless every required scenario field is PASS.",
            "Keep this review with the M10 readiness and rollout reports.",
        ],
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "foot_clearance": "UNCLEAR",
                "base_stability": "UNCLEAR",
                "toe_drag": "UNCLEAR",
                "command_transition_quality": "UNCLEAR",
                "notes": "",
            }
            for scenario_id in scenarios
        ],
    }


def render_visual_review_template_markdown(template: Mapping[str, Any]) -> str:
    profile = str(template.get("profile") or "")
    lines = [
        "# Soridormi M10 visual review template",
        "",
        f"Profile: `{profile}`",
        "",
        "Fill this after follow-camera inspection. Required values are `PASS`, `FAIL`, or `UNCLEAR`.",
        "",
        "| Scenario | Foot clearance | Base stability | Toe drag | Command transitions | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    scenarios = template.get("scenarios", [])
    if isinstance(scenarios, list):
        for item in scenarios:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| {scenario} | {foot} | {base} | {toe} | {transition} | {notes} |".format(
                    scenario=item.get("scenario_id"),
                    foot=item.get("foot_clearance"),
                    base=item.get("base_stability"),
                    toe=item.get("toe_drag"),
                    transition=item.get("command_transition_quality"),
                    notes=item.get("notes") or "",
                )
            )
    lines.extend(
        [
            "",
            "Promotion rule: visual review can only support M10 evidence after all required fields are `PASS`.",
            "Quantitative clearance readiness remains mandatory and cannot be replaced by visual review.",
            "",
        ]
    )
    return "\n".join(lines)


def _scenario_evidence(
    *,
    scenarios: Sequence[str],
    readiness: Mapping[str, Any] | None,
    review: Mapping[str, Any] | None,
) -> list[M10EvidenceScenario]:
    readiness_by_id = _scenario_map_from_readiness(readiness)
    review_by_id = _scenario_map_from_review(review)
    out: list[M10EvidenceScenario] = []
    for scenario_id in scenarios:
        readiness_item = readiness_by_id.get(scenario_id, {})
        report_path = readiness_item.get("report_path")
        report_path_text = str(report_path) if report_path else None
        present = bool(report_path_text and Path(report_path_text).exists())
        out.append(
            M10EvidenceScenario(
                scenario_id=scenario_id,
                rollout_report=report_path_text,
                rollout_report_present=present,
                clearance_status=str(readiness_item.get("status")) if readiness_item else None,
                clearance_ok=bool(readiness_item.get("clearance_ok")) if readiness_item else None,
                visual_review=dict(review_by_id[scenario_id]) if scenario_id in review_by_id else None,
            )
        )
    return out


def _next_steps(
    *,
    readiness_ok: bool,
    visual_plan_ok: bool,
    visual_review_ok: bool | None,
    status: str,
) -> list[str]:
    if not readiness_ok:
        return [
            "Run the required M10 scenario rollouts and regenerate clearance readiness.",
            "Keep the candidate blocked from M10 promotion until clearance readiness passes.",
        ]
    if not visual_plan_ok:
        return [
            "Generate the follow-camera visual inspection plan for all required M10 scenarios.",
        ]
    if visual_review_ok is None:
        return [
            "Run follow-camera visual inspection and fill m10_visual_review.json.",
            "Then rebuild this evidence package with --visual-review.",
        ]
    if visual_review_ok is False:
        return [
            "Keep the candidate blocked and inspect failed visual fields.",
            "Collect or retrain clearance-focused data before re-running M10 evidence.",
        ]
    if status == "READY_FOR_TEACHER_COMPARISON":
        return [
            "Run official-teacher comparison for the candidate.",
            "Keep readiness, visual review, rollout reports, and teacher-comparison artifacts together.",
        ]
    return []


def build_m10_evidence_package(
    *,
    profile: str = DEFAULT_PROFILE,
    scenarios: Sequence[str] = DEFAULT_REQUIRED_SCENARIOS,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    output_dir: str | Path | None = None,
    readiness_report: str | Path | None = None,
    visual_plan: str | Path | None = None,
    visual_review: str | Path | None = None,
    require_clearance_ready: bool = True,
    require_visual_plan: bool = True,
    require_visual_pass: bool = False,
) -> M10EvidencePackage:
    scenario_ids = list(scenarios)
    resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / profile
    readiness_path = Path(readiness_report) if readiness_report is not None else _default_readiness_report(profile)
    visual_plan_path = Path(visual_plan) if visual_plan is not None else _default_visual_plan(profile)
    visual_review_path = Path(visual_review) if visual_review is not None else _default_visual_review(resolved_output_dir)
    template_json_path = resolved_output_dir / "m10_visual_review_template.json"
    template_md_path = resolved_output_dir / "m10_visual_review_template.md"

    blockers: list[str] = []
    warnings: list[str] = []

    readiness, readiness_error = _load_json_if_present(readiness_path)
    if readiness_error:
        if require_clearance_ready:
            blockers.append(readiness_error)
        else:
            warnings.append(readiness_error)
    readiness_ok = bool(readiness and readiness.get("ok") is True)
    readiness_status = str(readiness.get("gate_status")) if readiness else None
    if require_clearance_ready and readiness is not None and not readiness_ok:
        blockers.append(f"clearance readiness is not passing: {readiness_status or 'UNKNOWN'}")

    plan, plan_error = _load_json_if_present(visual_plan_path)
    if plan_error:
        if require_visual_plan:
            blockers.append(plan_error)
        else:
            warnings.append(plan_error)
    visual_plan_ok = bool(plan and plan.get("ok") is True)
    visual_plan_status = str(plan.get("status")) if plan else None
    if require_visual_plan and plan is not None and not visual_plan_ok:
        blockers.append(f"visual inspection plan is blocked: {visual_plan_status or 'UNKNOWN'}")

    review, review_error = _load_json_if_present(visual_review_path)
    if review_error:
        if require_visual_pass:
            blockers.append(review_error)
        else:
            warnings.append(review_error)
        review = None
    visual_review_ok, visual_review_status, review_blockers = _evaluate_visual_review(
        review,
        scenarios=scenario_ids,
    )
    if review_blockers:
        blockers.extend(review_blockers if require_visual_pass else [])
        if not require_visual_pass:
            warnings.extend(review_blockers)

    if blockers:
        if not readiness_ok:
            status = "BLOCKED_BY_CLEARANCE_READINESS"
        elif not visual_plan_ok:
            status = "BLOCKED_BY_VISUAL_PLAN"
        elif visual_review_ok is False:
            status = "BLOCKED_BY_VISUAL_REVIEW"
        else:
            status = "BLOCKED"
    elif readiness_ok and visual_plan_ok and visual_review_ok is True:
        status = "READY_FOR_TEACHER_COMPARISON"
    elif readiness_ok and visual_plan_ok:
        status = "READY_FOR_VISUAL_REVIEW"
    elif readiness_ok:
        status = "READY_FOR_VISUAL_PLAN"
    else:
        status = "BLOCKED_BY_CLEARANCE_READINESS"

    commands = {
        "analyze_clearance_readiness": [
            "./scripts/analyze_m10_clearance_readiness.sh",
            "--profile-name",
            profile,
            "--output-dir",
            str(readiness_path.parent),
            "--json-output",
            str(readiness_path),
            "--json",
            "--strict",
        ],
        "plan_visual_inspection": [
            "./scripts/plan_m10_visual_inspection.sh",
            "--profile-name",
            profile,
            "--output-dir",
            str(visual_plan_path.parent),
            "--readiness-report",
            str(readiness_path),
            "--require-clearance-ready",
            "--json-output",
            str(visual_plan_path),
            "--json",
            "--strict",
        ],
        "rebuild_evidence_package": [
            "./scripts/build_m10_evidence_package.sh",
            "--profile-name",
            profile,
            "--output-dir",
            str(resolved_output_dir),
            "--readiness-report",
            str(readiness_path),
            "--visual-plan",
            str(visual_plan_path),
            "--visual-review",
            str(visual_review_path),
            "--json",
        ],
    }

    scenario_evidence = _scenario_evidence(
        scenarios=scenario_ids,
        readiness=readiness,
        review=review,
    )
    ok = not blockers and (
        status in {"READY_FOR_VISUAL_REVIEW", "READY_FOR_TEACHER_COMPARISON"}
    )
    return M10EvidencePackage(
        ok=ok,
        profile=profile,
        status=status,
        output_dir=str(resolved_output_dir),
        scenario_manifest=str(scenario_manifest),
        scenarios=[item.to_dict() for item in scenario_evidence],
        readiness_report=str(readiness_path),
        readiness_ok=readiness_ok,
        readiness_status=readiness_status,
        visual_plan=str(visual_plan_path),
        visual_plan_ok=visual_plan_ok,
        visual_plan_status=visual_plan_status,
        visual_review=str(visual_review_path) if review else None,
        visual_review_ok=visual_review_ok,
        visual_review_status=visual_review_status,
        review_template_json=str(template_json_path),
        review_template_markdown=str(template_md_path),
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        next_steps=_next_steps(
            readiness_ok=readiness_ok,
            visual_plan_ok=visual_plan_ok,
            visual_review_ok=visual_review_ok,
            status=status,
        ),
        commands=commands,
    )


def render_markdown(package: M10EvidencePackage) -> str:
    lines = [
        "# Soridormi M10 evidence package",
        "",
        f"Profile: `{package.profile}`",
        f"Status: `{package.status}`",
        f"Result: {'PASS' if package.ok else 'BLOCKED'}",
        f"Output dir: `{package.output_dir}`",
        "",
        "## Required artifacts",
        "",
        f"- Clearance readiness: `{package.readiness_report}` ({package.readiness_status or 'n/a'})",
        f"- Visual inspection plan: `{package.visual_plan}` ({package.visual_plan_status or 'n/a'})",
        f"- Visual review: `{package.visual_review or package.review_template_json}` ({package.visual_review_status or 'template pending'})",
        "",
        "## Scenario evidence",
        "",
        "| Scenario | Clearance | Rollout report present | Visual review |",
        "| --- | --- | --- | --- |",
    ]
    for item in package.scenarios:
        visual = item.get("visual_review")
        if isinstance(visual, Mapping):
            visual_status = ", ".join(
                f"{key}={_review_field_value(visual, key)}" for key in VISUAL_REVIEW_FIELDS
            )
        else:
            visual_status = "pending"
        lines.append(
            "| {scenario} | {clearance} | {present} | {visual} |".format(
                scenario=item.get("scenario_id"),
                clearance=item.get("clearance_status") or "n/a",
                present="yes" if item.get("rollout_report_present") else "no",
                visual=visual_status,
            )
        )
    lines.extend(["", "## Commands", ""])
    for name, command in package.commands.items():
        lines.extend([f"### {name}", "", "```bash", _command_text(command), "```", ""])
    lines.extend(["## Blockers", ""])
    lines.extend(f"- {item}" for item in package.blockers) if package.blockers else lines.append("- none")
    if package.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in package.warnings)
    lines.extend(["", "## Next steps", ""])
    lines.extend(f"- {item}" for item in package.next_steps) if package.next_steps else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Soridormi M10 evidence package manifest.")
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE, help="Policy profile to package.")
    parser.add_argument("--scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evidence package artifacts.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown package output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON package output path.")
    parser.add_argument("--readiness-report", type=Path, default=None, help="M10 clearance readiness JSON path.")
    parser.add_argument("--visual-plan", type=Path, default=None, help="M10 visual inspection plan JSON path.")
    parser.add_argument("--visual-review", type=Path, default=None, help="Filled M10 visual review JSON path.")
    parser.add_argument("--no-require-clearance-ready", action="store_true", help="Do not block when clearance readiness is missing/failing.")
    parser.add_argument("--no-require-visual-plan", action="store_true", help="Do not block when visual plan is missing/failing.")
    parser.add_argument("--require-visual-pass", action="store_true", help="Block unless a filled visual review passes every required field.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the evidence package is blocked.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenarios = _normalise_csv(args.scenario) or list(DEFAULT_REQUIRED_SCENARIOS)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.profile_name
    package = build_m10_evidence_package(
        profile=args.profile_name,
        scenarios=scenarios,
        scenario_manifest=args.scenario_manifest,
        output_dir=output_dir,
        readiness_report=args.readiness_report,
        visual_plan=args.visual_plan,
        visual_review=args.visual_review,
        require_clearance_ready=not args.no_require_clearance_ready,
        require_visual_plan=not args.no_require_visual_plan,
        require_visual_pass=args.require_visual_pass,
    )
    json_output = args.json_output or Path(output_dir) / "m10_evidence_package.json"
    markdown_output = args.output or Path(output_dir) / "m10_evidence_package.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(package), encoding="utf-8")

    template = build_visual_review_template(profile=args.profile_name, scenarios=scenarios)
    template_json = Path(package.review_template_json)
    template_md = Path(package.review_template_markdown)
    template_json.parent.mkdir(parents=True, exist_ok=True)
    template_json.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template_md.write_text(render_visual_review_template_markdown(template), encoding="utf-8")

    if args.json:
        print(json.dumps(package.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(package))
    return 1 if args.strict and not package.ok else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
