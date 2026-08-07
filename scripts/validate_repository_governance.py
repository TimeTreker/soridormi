#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DOCS = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("LLM_CONTEXT.md"),
    Path("docs/README.md"),
    Path("docs/STATUS.md"),
    Path("docs/DOCUMENTATION_GOVERNANCE.md"),
    Path("docs/PROJECT_SOP.md"),
    Path("docs/architecture.md"),
    Path("docs/SORIDORMI_TARGET_AND_ROADMAP.md"),
    Path("docs/SORIDORMI_EXECUTION_ROADMAP.md"),
    Path("docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md"),
    Path("docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md"),
    Path("docs/CHROMIE_SORIDORMI_TASK_AGENT_IMPLEMENTATION_PLAN.md"),
    Path("docs/SORIDORMI_BODY_CONCURRENCY.md"),
    Path("docs/SORIDORMI_MCP_SERVER.md"),
)

TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".h", ".hpp", ".ini", ".json", ".md", ".py",
    ".sh", ".toml", ".txt", ".yaml", ".yml",
}

EXCLUDED_TOP_LEVEL = {".git", "artifacts", "data", "workspace"}
EXCLUDED_DIRECTORY_NAMES = {"__pycache__"}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
OBSOLETE_ACTIVE_NAMES = {
    "NEXT_SESSION_PROMPT.md",
    "PROJECT_STATUS_AFTER_M6.md",
    "ROADMAP_M4_M7.md",
}

SEQUENCE_LABEL = re.compile(r"(?<![A-Za-z0-9_])M[0-9]+[A-Z]?(?![A-Za-z0-9_])")
NUMBERED_STAGE = re.compile(r"\bStage\s+[0-9]+\b", re.IGNORECASE)
NUMBERED_STEP_HEADING = re.compile(r"^#{1,6}\s+Step\s+[0-9]+\b", re.MULTILINE)
LEGACY_IDENTIFIER = re.compile(r"(?:^|[./_-])m[0-9]+[a-z0-9]*(?=[./_-]|$)", re.IGNORECASE)
LEGACY_FILENAME = re.compile(r"(?:^|[_-])(?:m[0-9]+[a-z]?|stage[_ -]?[0-9]+)(?:[_-]|\.)", re.IGNORECASE)

IMPLEMENTATION_PHASE = "_".join(("implementation", "phase"))
IMPLEMENTATION_PHASES = IMPLEMENTATION_PHASE + "s"
PROJECT_MILESTONE = "mile" + "stone"
CONTEXT_STAGE = "context_" + "stage1"

MAX_LINES = {
    Path("LLM_CONTEXT.md"): 180,
    Path("docs/STATUS.md"): 240,
    Path("docs/SORIDORMI_TARGET_AND_ROADMAP.md"): 260,
    Path("docs/SORIDORMI_EXECUTION_ROADMAP.md"): 280,
}


def first_party_files() -> list[Path]:
    result: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if relative.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
            continue
        result.append(relative)
    return sorted(result)


def main() -> int:
    errors: list[str] = []
    files = first_party_files()

    for relative in CANONICAL_DOCS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing canonical document: {relative}")

    for relative in files:
        if relative.name in OBSOLETE_ACTIVE_NAMES or relative.name.startswith("NEXT_SESSION_PROMPT_"):
            errors.append(f"obsolete session/status document remains active: {relative}")
        if LEGACY_FILENAME.search(relative.name):
            errors.append(f"sequence-labelled first-party filename: {relative}")

    for relative in files:
        if relative.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if relative == Path("scripts/validate_repository_governance.py"):
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if SEQUENCE_LABEL.search(text):
            errors.append(f"numbered project label remains in {relative}")
        if NUMBERED_STAGE.search(text):
            errors.append(f"numbered stage label remains in {relative}")
        if NUMBERED_STEP_HEADING.search(text):
            errors.append(f"numbered implementation-step heading remains in {relative}")
        if LEGACY_IDENTIFIER.search(text):
            errors.append(f"sequence-labelled identifier or path remains in {relative}")
        lowered = text.lower()
        if IMPLEMENTATION_PHASE in lowered or IMPLEMENTATION_PHASES in lowered:
            errors.append(f"deprecated implementation-phase field remains in {relative}")
        if PROJECT_MILESTONE in lowered:
            errors.append(f"project-sequence vocabulary remains in {relative}")
        if CONTEXT_STAGE in lowered:
            errors.append(f"legacy context input identifier remains in {relative}")
        if relative.suffix.lower() == ".md":
            fence_lines = re.findall(r"^`{3,}.*$", text, re.MULTILINE)
            if len(fence_lines) % 2:
                errors.append(f"unbalanced Markdown fence: {relative}")

    for relative, maximum in MAX_LINES.items():
        path = ROOT / relative
        if path.is_file():
            count = len(path.read_text(encoding="utf-8").splitlines())
            if count > maximum:
                errors.append(f"authority document is too large: {relative} ({count}>{maximum})")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("data/*", "!data/.gitkeep", "artifacts/*", "!artifacts/.gitkeep"):
        if required not in gitignore:
            errors.append(f".gitignore missing runtime-output rule: {required}")

    task_tools = (ROOT / "src/soridormi_runtime/mcp/task_tools.py").read_text(encoding="utf-8")
    if '"safe_idle": not emergency_stop' in task_tools:
        errors.append("task payload infers safe_idle only from emergency-stop state")
    if 'parameters.get("target_label") or "person"' in task_tools:
        errors.append("task lowering silently invents a person target")

    skill_manifest_path = ROOT / "configs/skills/open_duck_mini_v2_skills.json"
    skill_manifest = skill_manifest_path.read_text(encoding="utf-8")
    if '"target_ref"' in skill_manifest and '"default": "person"' in skill_manifest:
        errors.append("skill manifest silently defaults target_ref to person")
    for required in (
        '"ability_class_vocab"',
        '"control_coupling_vocab"',
        '"physical_resources"',
        '"concurrency"',
    ):
        if required not in skill_manifest:
            errors.append(f"skill manifest missing body-concurrency contract: {required}")

    body_activity_path = ROOT / "src/soridormi_runtime/mcp/body_activity.py"
    if not body_activity_path.is_file():
        errors.append("missing Soridormi body-activity scheduler contract")
    else:
        body_activity = body_activity_path.read_text(encoding="utf-8")
        for required in (
            'speech_owner": "chromie"',
            'one_final_motor_command_authority',
            'max_primary_locomotion_members',
            'one_writer_per_resource',
        ):
            if required not in body_activity:
                errors.append(f"body-activity contract missing invariant: {required}")
        if 'ABILITY_CLASS_SPEECH' in body_activity or '"speech" in BODY_ABILITY_CLASSES' in body_activity:
            errors.append("speech leaked into Soridormi physical ability classes")

    runtime_tools_path = ROOT / "src/soridormi_runtime/mcp/runtime_tools.py"
    runtime_tools = runtime_tools_path.read_text(encoding="utf-8")
    if "_motion_lock" in runtime_tools:
        errors.append("legacy global motion lock remains in runtime body execution")
    for required in (
        "_locomotion_lock",
        "_head_overlay_lock",
        "_independent_output_locks",
        "execute_runtime_activity_plan",
        "active_lanes",
    ):
        if required not in runtime_tools:
            errors.append(f"runtime missing resource-aware body execution: {required}")

    mcp_manifest = (ROOT / "src/soridormi_runtime/mcp/manifest.py").read_text(encoding="utf-8")
    for tool_name in (
        "soridormi.activity.get_capabilities",
        "soridormi.activity.compile",
        "soridormi.activity.execute",
        "soridormi.activity.status",
        "soridormi.activity.cancel",
    ):
        if tool_name not in mcp_manifest:
            errors.append(f"MCP manifest missing body-activity tool: {tool_name}")

    skill_execution = (ROOT / "src/soridormi_runtime/skill_execution.py").read_text(encoding="utf-8")
    if 'or "person"' in skill_execution or 'get("target_ref", "person")' in skill_execution:
        errors.append("skill execution silently invents a person target")

    if errors:
        print("Repository governance validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository governance validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
