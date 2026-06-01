from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_FREE_WALK_SUITE = Path("configs/teacher_suites/open_duck_free_walk_eval_v1.yaml")
DEFAULT_MAX_ABS_X = 0.12
DEFAULT_MAX_ABS_Y = 0.05
DEFAULT_MAX_ABS_YAW = 0.20
DEFAULT_MAX_STEPS = 1200
REQUIRED_TAGS = ("stand", "forward", "backward", "yaw", "curve", "lateral")


@dataclass(frozen=True)
class FreeWalkScenarioCheck:
    name: str
    steps: int
    command: dict[str, float]
    tags: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class FreeWalkSuiteCheck:
    ok: bool
    suite_path: str
    suite_name: str
    generated_at_utc: str
    scenario_count: int
    required_tags: list[str]
    present_tags: list[str]
    limits: dict[str, float | int]
    scenarios: list[FreeWalkScenarioCheck]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _float_value(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _scenario_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [str(raw)]
    return [str(item) for item in raw]


def _coverage_errors(present_tags: Iterable[str], required_tags: Iterable[str]) -> list[str]:
    present = set(present_tags)
    missing = [tag for tag in required_tags if tag not in present]
    return [f"missing required scenario tag: {tag}" for tag in missing]


def validate_free_walk_suite(
    suite_path: str | Path = DEFAULT_FREE_WALK_SUITE,
    *,
    max_abs_x: float = DEFAULT_MAX_ABS_X,
    max_abs_y: float = DEFAULT_MAX_ABS_Y,
    max_abs_yaw: float = DEFAULT_MAX_ABS_YAW,
    max_steps: int = DEFAULT_MAX_STEPS,
    required_tags: Iterable[str] = REQUIRED_TAGS,
) -> FreeWalkSuiteCheck:
    """Validate a conservative command-conditioned free-walk suite.

    This is a static gate for M6A. It does not prove locomotion quality by
    itself; it checks that the suite is present, bounded, and covers the command
    categories needed before running MuJoCo rollouts.
    """

    path = Path(suite_path)
    suite = _load_yaml_mapping(path)
    suite_name = str(suite.get("name") or path.stem)
    scenarios_raw = suite.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        return FreeWalkSuiteCheck(
            ok=False,
            suite_path=str(path),
            suite_name=suite_name,
            generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            scenario_count=0,
            required_tags=list(required_tags),
            present_tags=[],
            limits={
                "max_abs_x": max_abs_x,
                "max_abs_y": max_abs_y,
                "max_abs_yaw": max_abs_yaw,
                "max_steps": max_steps,
            },
            scenarios=[],
            errors=["suite must define a non-empty scenarios list"],
        )

    default_steps = int(suite.get("default_steps", 0) or 0)
    checks: list[FreeWalkScenarioCheck] = []
    present_tags: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []

    for index, item in enumerate(scenarios_raw):
        scenario_errors: list[str] = []
        scenario_warnings: list[str] = []
        if not isinstance(item, dict):
            checks.append(
                FreeWalkScenarioCheck(
                    name=f"scenario_{index}",
                    steps=0,
                    command={},
                    tags=[],
                    errors=["scenario must be a mapping"],
                )
            )
            continue

        name = str(item.get("name") or f"scenario_{index}")
        tags = _scenario_tags(item.get("tags"))
        present_tags.update(tags)
        steps = int(item.get("steps", default_steps) or 0)
        command_raw = item.get("command") or {}
        if not isinstance(command_raw, dict):
            command_raw = {}
            scenario_errors.append("command must be a mapping")

        command = {
            "x": _float_value(command_raw.get("x")),
            "y": _float_value(command_raw.get("y")),
            "yaw": _float_value(command_raw.get("yaw")),
            "ramp_seconds": _float_value(command_raw.get("ramp_seconds")),
        }

        if steps <= 0:
            scenario_errors.append("steps must be positive")
        if steps > max_steps:
            scenario_errors.append(f"steps {steps} exceeds max_steps {max_steps}")
        if abs(command["x"]) > max_abs_x:
            scenario_errors.append(f"|command.x| {abs(command['x'])} exceeds max_abs_x {max_abs_x}")
        if abs(command["y"]) > max_abs_y:
            scenario_errors.append(f"|command.y| {abs(command['y'])} exceeds max_abs_y {max_abs_y}")
        if abs(command["yaw"]) > max_abs_yaw:
            scenario_errors.append(f"|command.yaw| {abs(command['yaw'])} exceeds max_abs_yaw {max_abs_yaw}")
        if command["ramp_seconds"] < 0:
            scenario_errors.append("command.ramp_seconds must be non-negative")
        if "free_walk" not in tags:
            scenario_warnings.append("scenario is missing free_walk tag")

        checks.append(
            FreeWalkScenarioCheck(
                name=name,
                steps=steps,
                command=command,
                tags=tags,
                errors=scenario_errors,
                warnings=scenario_warnings,
            )
        )

    errors.extend(_coverage_errors(present_tags, required_tags))
    for check in checks:
        errors.extend(f"{check.name}: {error}" for error in check.errors)
        warnings.extend(f"{check.name}: {warning}" for warning in check.warnings)

    return FreeWalkSuiteCheck(
        ok=not errors,
        suite_path=str(path),
        suite_name=suite_name,
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        scenario_count=len(checks),
        required_tags=list(required_tags),
        present_tags=sorted(present_tags),
        limits={
            "max_abs_x": max_abs_x,
            "max_abs_y": max_abs_y,
            "max_abs_yaw": max_abs_yaw,
            "max_steps": max_steps,
        },
        scenarios=checks,
        errors=errors,
        warnings=warnings,
    )


def render_free_walk_suite_check(result: FreeWalkSuiteCheck) -> str:
    lines = [
        "# Soridormi free-walk suite check",
        "",
        f"Result: {'PASS' if result.ok else 'FAIL'}",
        f"Suite: {result.suite_name}",
        f"Path: {result.suite_path}",
        f"Generated: {result.generated_at_utc}",
        f"Scenarios: {result.scenario_count}",
        "",
        "## Coverage",
        "",
        f"Required tags: {', '.join(result.required_tags)}",
        f"Present tags: {', '.join(result.present_tags)}",
        "",
        "## Scenarios",
        "",
        "| Scenario | Result | Steps | x | y | yaw | Tags |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.scenarios:
        command = item.command
        lines.append(
            f"| {item.name} | {'PASS' if item.ok else 'FAIL'} | {item.steps} | "
            f"{command.get('x', 0.0):.6g} | {command.get('y', 0.0):.6g} | "
            f"{command.get('yaw', 0.0):.6g} | {', '.join(item.tags)} |"
        )

    lines.extend(["", "## Errors", ""])
    if result.errors:
        lines.extend(f"- {error}" for error in result.errors)
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the M6A commanded free-walk evaluation suite.")
    parser.add_argument("--suite", default=str(DEFAULT_FREE_WALK_SUITE))
    parser.add_argument("--max-abs-x", type=float, default=DEFAULT_MAX_ABS_X)
    parser.add_argument("--max-abs-y", type=float, default=DEFAULT_MAX_ABS_Y)
    parser.add_argument("--max-abs-yaw", type=float, default=DEFAULT_MAX_ABS_YAW)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_free_walk_suite(
            args.suite,
            max_abs_x=args.max_abs_x,
            max_abs_y=args.max_abs_y,
            max_abs_yaw=args.max_abs_yaw,
            max_steps=args.max_steps,
        )
    except Exception as exc:
        payload = {"ok": False, "errors": [repr(exc)]}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Soridormi free-walk suite check")
            print("================================")
            print("Result: FAILED")
            print(f"Error: {exc!r}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_free_walk_suite_check(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
