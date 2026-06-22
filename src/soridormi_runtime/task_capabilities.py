from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_CAPABILITY_MANIFEST = (
    ROOT / "configs" / "task_capabilities" / "open_duck_mini_v2_task_capabilities.json"
)


class TaskCapabilityManifestError(ValueError):
    """Raised when the task capability manifest is structurally invalid."""


@dataclass(frozen=True)
class TaskCapabilityValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def load_task_capability_manifest(
    path: str | Path = DEFAULT_TASK_CAPABILITY_MANIFEST,
) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - stdlib message
        raise TaskCapabilityManifestError(
            f"Invalid JSON task capability manifest {manifest_path}: {exc}"
        ) from exc


def validate_task_capability_manifest(
    manifest: dict[str, Any],
) -> TaskCapabilityValidationResult:
    errors: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("robot") != "open_duck_mini_v2":
        errors.append("robot must be open_duck_mini_v2")
    if not manifest.get("readiness_profile") and not manifest.get("milestone"):
        errors.append("readiness_profile is required")
    if manifest.get("task_api_no_motion") is not True:
        errors.append("task_api_no_motion must be true")
    if not isinstance(manifest.get("physical_execution_note"), str) or not manifest.get(
        "physical_execution_note",
    ):
        errors.append("physical_execution_note must be a non-empty string")

    ready_subsystems = manifest.get("ready_subsystems")
    if not isinstance(ready_subsystems, list) or not ready_subsystems:
        errors.append("ready_subsystems must be a non-empty list")
        ready_subsystems_set: set[str] = set()
    else:
        ready_subsystems_set = {str(item) for item in ready_subsystems}

    readiness_vocab = manifest.get("readiness_vocab")
    if not isinstance(readiness_vocab, list) or not readiness_vocab:
        errors.append("readiness_vocab must be a non-empty list")
        readiness_vocab_set: set[str] = set()
    else:
        readiness_vocab_set = {str(item) for item in readiness_vocab}

    unsafe_task_types = manifest.get("unsafe_task_types")
    if not isinstance(unsafe_task_types, list) or not unsafe_task_types:
        errors.append("unsafe_task_types must be a non-empty list")
        unsafe_task_type_set: set[str] = set()
    else:
        unsafe_task_type_set = {str(item) for item in unsafe_task_types}

    task_types = manifest.get("task_types")
    if not isinstance(task_types, list) or not task_types:
        errors.append("task_types must be a non-empty list")
        return TaskCapabilityValidationResult(ok=False, errors=tuple(errors))

    seen: set[str] = set()
    for index, task in enumerate(task_types):
        if not isinstance(task, dict):
            errors.append(f"task_types[{index}] must be an object")
            continue
        task_type = task.get("task_type")
        if not isinstance(task_type, str) or not task_type:
            errors.append(f"task_types[{index}].task_type is required")
            task_type = f"<missing-{index}>"
        if task_type in seen:
            errors.append(f"duplicate task_type: {task_type}")
        seen.add(task_type)
        if task_type in unsafe_task_type_set:
            errors.append(f"task {task_type}: unsafe task type must not be executable")

        for field in [
            "description",
            "readiness",
            "execution_modes",
            "required_subsystems",
            "external_dependencies",
            "missing_subsystems",
            "reason_code",
            "recommended_actions",
        ]:
            if field not in task:
                errors.append(f"task {task_type}: {field} is required")

        readiness = task.get("readiness")
        if readiness not in readiness_vocab_set:
            errors.append(f"task {task_type}: unknown readiness {readiness!r}")
        if readiness in {"future_blocked", "safety_redirect"} and not str(
            task.get("reason_code") or "",
        ).strip():
            errors.append(f"task {task_type}: {readiness} requires reason_code")
        if readiness in {"future_blocked", "safety_redirect"} and task.get(
            "persistent_submit_allowed",
        ) is not False:
            errors.append(
                f"task {task_type}: {readiness} must not allow persistent submit"
            )
        if task.get("task_api_no_motion") is False:
            errors.append(f"task {task_type}: task_api_no_motion must not be false")
        if task.get("physical_execution_ready") is True:
            errors.append(f"task {task_type}: physical_execution_ready must not be true")
        if not isinstance(task.get("persistent_submit_allowed"), bool):
            errors.append(f"task {task_type}: persistent_submit_allowed must be boolean")

        for list_field in [
            "execution_modes",
            "required_subsystems",
            "external_dependencies",
            "missing_subsystems",
            "recommended_actions",
        ]:
            if not isinstance(task.get(list_field), list):
                errors.append(f"task {task_type}: {list_field} must be a list")

        required = {str(item) for item in task.get("required_subsystems", [])}
        external = {str(item) for item in task.get("external_dependencies", [])}
        missing = {str(item) for item in task.get("missing_subsystems", [])}
        unknown_external = external - required
        if unknown_external:
            errors.append(
                f"task {task_type}: external_dependencies not in required_subsystems "
                f"{sorted(unknown_external)}"
            )
        unknown_missing = missing - required - {"dedicated_stop_tool_required"}
        if unknown_missing:
            errors.append(
                f"task {task_type}: missing_subsystems not in required_subsystems "
                f"{sorted(unknown_missing)}"
            )
        ready_required = required & ready_subsystems_set
        if ready_required & missing:
            errors.append(
                f"task {task_type}: subsystem cannot be both ready and missing "
                f"{sorted(ready_required & missing)}"
            )
        if external & missing:
            errors.append(
                f"task {task_type}: external dependency cannot be missing subsystem "
                f"{sorted(external & missing)}"
            )

    return TaskCapabilityValidationResult(ok=not errors, errors=tuple(errors))


def task_capabilities_by_type(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(task["task_type"]): task
        for task in manifest.get("task_types", [])
        if isinstance(task, dict) and "task_type" in task
    }
