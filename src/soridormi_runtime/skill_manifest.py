"""Skill manifest loader and CLI for Soridormi body skills.

The skill manifest is intentionally JSON-only for now so it can be read from
host scripts, runtime containers, and future MCP servers without optional YAML
dependencies.  This module does not execute skills; it validates and exposes
what the robot body claims it can do.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILL_MANIFEST = ROOT / "configs" / "skills" / "open_duck_mini_v2_skills.json"
AVAILABLE_STATUSES = {"available_sim", "available_sim_experimental"}
FUTURE_STATUSES = {
    "planned",
    "planned_wrapper",
    "planned_external_target",
    "future",
    "future_pose_teacher",
    "future_residual_rl",
    "future_evaluation",
    "future_perception",
    "future_composite",
}
UNSUPPORTED_STATUSES = {"unsupported_current_robot"}


@dataclass(frozen=True)
class SkillValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class SkillQuery:
    category: str | None = None
    status: str | None = None
    execution: str | None = None
    available_only: bool = False
    future_only: bool = False
    include_unsupported: bool = False


class SkillManifestError(ValueError):
    """Raised when a skill manifest is structurally invalid."""


def load_skill_manifest(path: str | Path = DEFAULT_SKILL_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - message shape from stdlib
        raise SkillManifestError(f"Invalid JSON skill manifest {manifest_path}: {exc}") from exc


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def validate_skill_manifest(manifest: dict[str, Any]) -> SkillValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not manifest.get("robot"):
        errors.append("robot is required")
    if not manifest.get("capability_profile") and not manifest.get("milestone"):
        errors.append("capability_profile is required")
    if manifest.get("milestone") and not manifest.get("capability_profile"):
        warnings.append("milestone is deprecated; use capability_profile")

    statuses = _as_set(manifest.get("status_vocab"))
    executions = _as_set(manifest.get("execution_vocab"))
    if not statuses:
        errors.append("status_vocab must be a non-empty list")
    if not executions:
        errors.append("execution_vocab must be a non-empty list")

    actuator_groups = manifest.get("actuator_groups", {})
    if not isinstance(actuator_groups, dict) or not actuator_groups:
        errors.append("actuator_groups must be a non-empty object")
    supported_groups = {
        name
        for name, config in actuator_groups.items()
        if isinstance(config, dict) and bool(config.get("supported"))
    }
    all_groups = set(actuator_groups)

    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        errors.append("defaults must be an object")
    elif defaults.get("hardware_enabled") is not False:
        errors.append("defaults.hardware_enabled must remain false for sim-first skill work")

    phases = {phase.get("id") for phase in manifest.get("implementation_phases", []) if isinstance(phase, dict)}
    if not phases:
        errors.append("implementation_phases must declare at least one phase")

    skills = manifest.get("skills")
    if not isinstance(skills, list) or not skills:
        errors.append("skills must be a non-empty list")
        return SkillValidationResult(ok=False, errors=tuple(errors), warnings=tuple(warnings))

    seen_ids: set[str] = set()
    available_count = 0
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            errors.append(f"skills[{index}] must be an object")
            continue

        skill_id = skill.get("id")
        if not isinstance(skill_id, str) or not skill_id:
            errors.append(f"skills[{index}].id is required")
            skill_id = f"<missing-{index}>"
        if skill_id in seen_ids:
            errors.append(f"duplicate skill id: {skill_id}")
        seen_ids.add(skill_id)

        for field in ["category", "display_name", "description", "status", "execution", "implementation_phase"]:
            if not skill.get(field):
                errors.append(f"skill {skill_id}: {field} is required")

        status = skill.get("status")
        execution = skill.get("execution")
        if status not in statuses:
            errors.append(f"skill {skill_id}: unknown status {status!r}")
        if execution not in executions:
            errors.append(f"skill {skill_id}: unknown execution {execution!r}")
        if skill.get("implementation_phase") not in phases:
            errors.append(f"skill {skill_id}: implementation_phase is not declared")

        required_groups = _as_set(skill.get("required_actuator_groups"))
        unknown_groups = required_groups - all_groups
        if unknown_groups:
            errors.append(f"skill {skill_id}: unknown actuator groups {sorted(unknown_groups)}")
        if status in AVAILABLE_STATUSES:
            available_count += 1
            unsupported_required = required_groups - supported_groups
            if unsupported_required:
                errors.append(
                    f"skill {skill_id}: available skill requires unsupported actuator groups "
                    f"{sorted(unsupported_required)}"
                )

        safety = skill.get("safety")
        if not isinstance(safety, dict):
            errors.append(f"skill {skill_id}: safety object is required")
        else:
            if "interruptible" not in safety:
                errors.append(f"skill {skill_id}: safety.interruptible is required")
            if "fallback" not in safety:
                errors.append(f"skill {skill_id}: safety.fallback is required")
            if safety.get("hardware_enabled") is not False:
                errors.append(f"skill {skill_id}: hardware execution must remain disabled in manifest")

        parameters = skill.get("parameters", {})
        if not isinstance(parameters, dict):
            errors.append(f"skill {skill_id}: parameters must be an object")
        else:
            for param_name, param in parameters.items():
                if isinstance(param, dict):
                    if "min" in param and "max" in param and param["min"] > param["max"]:
                        errors.append(f"skill {skill_id}: parameter {param_name} min > max")
                    if "default" in param and "min" in param and param["default"] < param["min"]:
                        errors.append(f"skill {skill_id}: parameter {param_name} default < min")
                    if "default" in param and "max" in param and param["default"] > param["max"]:
                        errors.append(f"skill {skill_id}: parameter {param_name} default > max")

    if available_count < 1:
        warnings.append("no skills are currently available in simulation")
    if available_count > 10:
        warnings.append("safe sim executable subset is larger than the current 10-skill checkpoint target")

    return SkillValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def skills_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(skill["id"]): skill for skill in manifest.get("skills", []) if isinstance(skill, dict) and "id" in skill}


def iter_skills(manifest: dict[str, Any], query: SkillQuery | None = None) -> list[dict[str, Any]]:
    query = query or SkillQuery()
    skills = [skill for skill in manifest.get("skills", []) if isinstance(skill, dict)]

    if query.category:
        skills = [skill for skill in skills if skill.get("category") == query.category]
    if query.status:
        skills = [skill for skill in skills if skill.get("status") == query.status]
    if query.execution:
        skills = [skill for skill in skills if skill.get("execution") == query.execution]
    if query.available_only:
        skills = [skill for skill in skills if skill.get("status") in AVAILABLE_STATUSES]
    if query.future_only:
        skills = [skill for skill in skills if skill.get("status") in FUTURE_STATUSES]
    if not query.include_unsupported:
        skills = [skill for skill in skills if skill.get("status") not in UNSUPPORTED_STATUSES]

    return sorted(skills, key=lambda skill: (str(skill.get("category", "")), str(skill.get("id", ""))))


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    skills = [skill for skill in manifest.get("skills", []) if isinstance(skill, dict)]
    by_category: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    for skill in skills:
        by_category[str(skill.get("category", "unknown"))] = by_category.get(str(skill.get("category", "unknown")), 0) + 1
        by_status[str(skill.get("status", "unknown"))] = by_status.get(str(skill.get("status", "unknown")), 0) + 1
        by_phase[str(skill.get("implementation_phase", "unknown"))] = by_phase.get(
            str(skill.get("implementation_phase", "unknown")), 0
        ) + 1
    return {
        "robot": manifest.get("robot"),
        "capability_profile": manifest.get("capability_profile") or manifest.get("milestone"),
        "skill_count": len(skills),
        "available_sim_count": sum(1 for skill in skills if skill.get("status") in AVAILABLE_STATUSES),
        "unsupported_count": sum(1 for skill in skills if skill.get("status") in UNSUPPORTED_STATUSES),
        "by_category": dict(sorted(by_category.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_phase": dict(sorted(by_phase.items())),
    }


def build_llm_skill_context(manifest: dict[str, Any], *, language: str = "en") -> str:
    available = iter_skills(manifest, SkillQuery(available_only=True, include_unsupported=False))
    future = iter_skills(manifest, SkillQuery(future_only=True, include_unsupported=False))
    unsupported = iter_skills(manifest, SkillQuery(include_unsupported=True))
    unsupported = [skill for skill in unsupported if skill.get("status") in UNSUPPORTED_STATUSES]

    if language.lower().startswith("zh"):
        lines = [
            "Soridormi 技能能力摘要",
            "======================",
            "Soridormi 现在把完整技能宇宙先声明出来，但只允许少量安全的 MuJoCo/sim 技能先落地。",
            "",
            "当前可执行/可包装的仿真技能：",
        ]
        for skill in available:
            lines.append(f"- {skill['id']}: {skill.get('description', '')}")
        lines.extend(["", "已规划但还不能执行的技能："])
        for skill in future[:12]:
            lines.append(f"- {skill['id']} ({skill.get('status')}): {skill.get('description', '')}")
        lines.extend(["", "当前硬件不支持的技能："])
        for skill in unsupported:
            lines.append(f"- {skill['id']}: 需要 {', '.join(skill.get('required_actuator_groups', []))}")
        lines.extend(
            [
                "",
                "规则：",
                "- 不要调用 status=unsupported_current_robot 的技能。",
                "- 不要把 planned/future 技能当成已经可执行。",
                "- 普通人类口令优先使用 walk_forward(speed=slow/normal/medium/quick/fast_limited)，不要要求用户给 vx_mps。",
                "- 所有技能默认 sim-first，hardware_enabled=false。",
                "- 社交语音/TTS 属于 Chromie，不属于 Soridormi。",
            ]
        )
        return "\n".join(lines)

    lines = [
        "Soridormi skill capability summary",
        "==================================",
        "Soridormi declares the full skill universe up front, but only a small safe MuJoCo/sim subset is executable now.",
        "",
        "Currently available or wrapper-ready simulation skills:",
    ]
    for skill in available:
        lines.append(f"- {skill['id']}: {skill.get('description', '')}")
    lines.extend(["", "Planned/future skills that are not executable yet:"])
    for skill in future[:12]:
        lines.append(f"- {skill['id']} ({skill.get('status')}): {skill.get('description', '')}")
    lines.extend(["", "Unsupported on the current robot hardware:"])
    for skill in unsupported:
        lines.append(f"- {skill['id']}: requires {', '.join(skill.get('required_actuator_groups', []))}")
    lines.extend(
        [
            "",
            "Rules:",
            "- Do not call skills with status=unsupported_current_robot.",
            "- Do not treat planned/future skills as executable.",
            "- Prefer walk_forward(speed=slow/normal/medium/quick/fast_limited) for ordinary human walking requests; do not ask users for vx_mps.",
            "- All skills are sim-first and hardware_enabled=false by default.",
            "- Social speech/TTS belongs to Chromie, not Soridormi.",
        ]
    )
    return "\n".join(lines)


def _print_text_summary(manifest: dict[str, Any], skills: Iterable[dict[str, Any]]) -> None:
    summary = summarize_manifest(manifest)
    print("Soridormi skill manifest")
    print("=========================")
    print(f"Robot: {summary['robot']}")
    print(f"Capability profile: {summary['capability_profile']}")
    print(f"Skills: {summary['skill_count']}")
    print(f"Available sim skills: {summary['available_sim_count']}")
    print(f"Unsupported current robot skills: {summary['unsupported_count']}")
    print("")
    print("Skills")
    print("------")
    for skill in skills:
        print(f"- {skill['id']} [{skill.get('category')}/{skill.get('status')}]: {skill.get('description', '')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List and validate Soridormi skill manifests.")
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument("--validate-only", action="store_true", help="Validate manifest and print validation result.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--llm-context", action="store_true", help="Print an LLM-readable skill capability summary.")
    parser.add_argument("--language", default="en", help="Language for --llm-context, e.g. en or zh.")
    parser.add_argument("--category", help="Filter by skill category.")
    parser.add_argument("--status", help="Filter by skill status.")
    parser.add_argument("--execution", help="Filter by execution type.")
    parser.add_argument("--available", action="store_true", help="Show only currently available simulation skills.")
    parser.add_argument("--future", action="store_true", help="Show only planned/future skills.")
    parser.add_argument("--include-unsupported", action="store_true", help="Include unsupported current-robot skills.")
    parser.add_argument("--skill", help="Show one skill by id.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifest = load_skill_manifest(args.manifest)
    validation = validate_skill_manifest(manifest)
    if not validation.ok:
        if args.json:
            print(json.dumps({"validation": validation.to_dict()}, indent=2, sort_keys=True))
        else:
            print("Skill manifest validation: FAILED")
            for error in validation.errors:
                print(f"- {error}")
        return 2

    if args.validate_only:
        if args.json:
            print(json.dumps({"validation": validation.to_dict()}, indent=2, sort_keys=True))
        else:
            print("Skill manifest validation: OK")
            for warning in validation.warnings:
                print(f"warning: {warning}")
        return 0

    if args.llm_context:
        print(build_llm_skill_context(manifest, language=args.language))
        return 0

    query = SkillQuery(
        category=args.category,
        status=args.status,
        execution=args.execution,
        available_only=args.available,
        future_only=args.future,
        include_unsupported=args.include_unsupported,
    )
    skills = iter_skills(manifest, query)

    if args.skill:
        skill = skills_by_id(manifest).get(args.skill)
        if skill is None:
            print(f"Unknown skill: {args.skill}")
            return 1
        skills = [skill]

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summarize_manifest(manifest),
                    "validation": validation.to_dict(),
                    "skills": skills,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_text_summary(manifest, skills)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
