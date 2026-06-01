from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCENARIO_MANIFEST = Path("configs/scenarios/open_duck_mini_v2_scenarios.json")
COLLECTOR_READY_STATUSES = frozenset({"mujoco_registry_ready", "mujoco_eval_ready", "training_ready"})
UNSUPPORTED_STATUS = "unsupported_current_robot"


class ScenarioCurriculumError(ValueError):
    """Raised when scenario curriculum metadata is missing or invalid."""


@dataclass(frozen=True)
class ScenarioDefinition:
    """Typed view over one entry in the Soridormi scenario curriculum."""

    payload: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload["id"])

    @property
    def title(self) -> str:
        return str(self.payload.get("title", self.id))

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "planned"))

    @property
    def priority(self) -> int:
        return int(self.payload.get("priority", 0))

    @property
    def family(self) -> str:
        return str(self.payload.get("family", ""))

    @property
    def skills(self) -> list[str]:
        raw = self.payload.get("skills", [])
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]

    @property
    def primary_skill(self) -> str | None:
        return self.skills[0] if self.skills else None

    @property
    def task_context(self) -> dict[str, Any]:
        raw = self.payload.get("task_context", {})
        return deepcopy(raw) if isinstance(raw, dict) else {}

    @property
    def environment_context(self) -> dict[str, Any]:
        raw = self.payload.get("environment_context", {})
        return deepcopy(raw) if isinstance(raw, dict) else {}

    @property
    def command_space(self) -> dict[str, Any]:
        raw = self.payload.get("command_space", {})
        return deepcopy(raw) if isinstance(raw, dict) else {}

    @property
    def dataset_tags(self) -> list[str]:
        raw = self.payload.get("dataset_tags", [])
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]

    def command_range(self, field_name: str) -> tuple[float, float]:
        raw = self.command_space.get(field_name)
        if not isinstance(raw, list) or len(raw) != 2:
            raise ScenarioCurriculumError(
                f"scenario {self.id!r} does not define command_space.{field_name} as [min, max]"
            )
        try:
            minimum = float(raw[0])
            maximum = float(raw[1])
        except (TypeError, ValueError) as exc:
            raise ScenarioCurriculumError(
                f"scenario {self.id!r} command_space.{field_name} must contain numbers"
            ) from exc
        if minimum > maximum:
            raise ScenarioCurriculumError(
                f"scenario {self.id!r} command_space.{field_name} minimum exceeds maximum"
            )
        return minimum, maximum

    def command_range_text(self, field_name: str) -> str:
        minimum, maximum = self.command_range(field_name)
        return f"{minimum:g},{maximum:g}"

    def ramp_names(self) -> list[str]:
        raw = self.command_space.get("ramps", [])
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]

    def ramp_name_for_segment(self, segment_index: int) -> str | None:
        names = self.ramp_names()
        if not names:
            return None
        return names[int(segment_index) % len(names)]

    def dataset_metadata(self) -> dict[str, Any]:
        """Return bounded structured context suitable for dataset JSONL rows."""

        return {
            "scenario_id": self.id,
            "scenario_title": self.title,
            "scenario_status": self.status,
            "scenario_family": self.family,
            "scenario_priority": self.priority,
            "skill_id": self.primary_skill,
            "scenario_skills": self.skills,
            "scenario_dataset_tags": self.dataset_tags,
            "task_context": self.task_context,
            "environment_context": self.environment_context,
            "command_space": self.command_space,
        }


def load_scenario_manifest(path: str | Path = DEFAULT_SCENARIO_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise ScenarioCurriculumError(f"scenario manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioCurriculumError(f"scenario manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScenarioCurriculumError(f"scenario manifest must contain a JSON object: {manifest_path}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ScenarioCurriculumError(f"scenario manifest missing scenarios list: {manifest_path}")
    return payload


def iter_scenarios(path: str | Path = DEFAULT_SCENARIO_MANIFEST) -> Iterable[ScenarioDefinition]:
    manifest = load_scenario_manifest(path)
    for raw in manifest.get("scenarios", []):
        if isinstance(raw, dict):
            yield ScenarioDefinition(deepcopy(raw))


def list_scenarios(
    path: str | Path = DEFAULT_SCENARIO_MANIFEST,
    *,
    include_planned: bool = True,
) -> list[ScenarioDefinition]:
    scenarios = list(iter_scenarios(path))
    if not include_planned:
        scenarios = [item for item in scenarios if item.status in COLLECTOR_READY_STATUSES]
    return sorted(scenarios, key=lambda item: item.priority)


def get_scenario_definition(
    scenario_id: str,
    path: str | Path = DEFAULT_SCENARIO_MANIFEST,
) -> ScenarioDefinition:
    needle = str(scenario_id)
    for scenario in iter_scenarios(path):
        if scenario.id == needle:
            return scenario
    known = ", ".join(item.id for item in list_scenarios(path)[:20])
    raise ScenarioCurriculumError(f"unknown scenario_id {needle!r}; known scenarios: {known}")


def validate_scenario_for_teacher_collection(
    scenario: ScenarioDefinition,
    *,
    allow_planned: bool = False,
) -> list[str]:
    """Validate scenario use for live teacher collection and return non-fatal warnings."""

    if scenario.status == UNSUPPORTED_STATUS:
        raise ScenarioCurriculumError(
            f"scenario {scenario.id!r} is unsupported for the current robot actuator contract"
        )
    if scenario.status not in COLLECTOR_READY_STATUSES and not allow_planned:
        raise ScenarioCurriculumError(
            f"scenario {scenario.id!r} status is {scenario.status!r}; pass "
            "--allow-planned-scenario to collect metadata-only rows before MuJoCo eval promotion"
        )

    # The random teacher collector produces velocity-command locomotion data.
    # Non-locomotion scenarios may still live in the curriculum, but they should
    # not be routed through this collector until a dedicated desired-state
    # collector exists.
    for field_name in ("vx_mps", "vy_mps", "yaw_radps"):
        scenario.command_range(field_name)

    warnings: list[str] = []
    if scenario.status not in COLLECTOR_READY_STATUSES:
        warnings.append(
            f"scenario {scenario.id!r} is {scenario.status!r}; rows are metadata-ready but not "
            "evidence of MuJoCo scenario acceptance"
        )
    return warnings
