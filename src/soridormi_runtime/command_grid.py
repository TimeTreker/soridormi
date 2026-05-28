from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.policy_profiles import PolicyProfile


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError("name cannot be empty")
    return text


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _container_data_path(path: Path) -> str:
    text = str(path)
    cwd = str(Path.cwd())
    if text == "/data" or text.startswith("/data/"):
        return text
    if text == "data":
        return "/data"
    if text.startswith("data/"):
        return "/data/" + text[len("data/") :]
    if text == cwd + "/data":
        return "/data"
    if text.startswith(cwd + "/data/"):
        return "/data/" + text[len(cwd + "/data/") :]
    return text


def _host_data_path(path_text: str) -> Path:
    if path_text == "/data":
        return Path("data")
    if path_text.startswith("/data/"):
        return Path("data") / path_text[len("/data/") :]
    return Path(path_text)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CommandGridScenario:
    name: str
    tags: list[str]
    command: dict[str, float]
    steps: int
    seconds: float | None
    teacher_profile_path: str
    candidate_profile_path: str
    candidate_profile_name: str


@dataclass
class CommandGridManifest:
    ok: bool
    grid_name: str
    teacher_suite_manifest: str
    candidate_base_profile: str
    output_dir: str
    generated_at_utc: str
    scenarios: list[CommandGridScenario]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommandGridComparisonScenario:
    name: str
    ok: bool
    comparison_path: str
    teacher_log: str | None
    candidate_log: str | None
    forward_ratio: float | None
    speed_ratio: float | None
    lateral_abs: float | None
    reset_count: int | None
    action_abs_max: float | None
    errors: list[str]
    warnings: list[str]


@dataclass
class CommandGridComparisonSummary:
    ok: bool
    output_dir: str
    generated_at_utc: str
    scenario_count: int
    passed_count: int
    failed_count: int
    scenarios: list[CommandGridComparisonScenario]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_candidate_command_grid(
    teacher_suite_manifest: str | Path,
    candidate_profile: str | Path,
    *,
    output_dir: str | Path,
    force: bool = False,
) -> CommandGridManifest:
    """Generate candidate profiles matching every command in a teacher suite.

    The teacher suite already defines safe/interesting command scenarios. This
    function clones the candidate policy profile once per teacher scenario and
    injects the same command values. The result is a fair teacher-vs-candidate
    rollout grid where both policies receive identical command requests.
    """

    manifest_path = Path(teacher_suite_manifest)
    teacher = _load_json(manifest_path)
    scenarios_raw = teacher.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("teacher suite manifest must contain non-empty scenarios list")

    candidate = PolicyProfile.load(candidate_profile)
    out = Path(output_dir)
    profiles_dir = out / "candidate_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    grid_name = slugify(f"{candidate.name}_vs_{teacher.get('suite_name') or manifest_path.stem}")
    scenarios: list[CommandGridScenario] = []

    for item in scenarios_raw:
        if not isinstance(item, dict):
            raise ValueError("teacher suite scenario must be a mapping")
        name = slugify(str(item.get("name") or "scenario"))
        command = item.get("command") or {}
        if not isinstance(command, dict):
            raise ValueError(f"scenario {name}: command must be a mapping")
        steps = int(item.get("steps") or 1000)
        seconds_raw = item.get("seconds")
        seconds = float(seconds_raw) if seconds_raw not in {None, ""} else None
        tags = [str(tag) for tag in (item.get("tags") or [])]
        teacher_profile_path = str(item.get("profile_path") or "")
        if not teacher_profile_path:
            raise ValueError(f"scenario {name}: profile_path is required")

        candidate_profile_name = slugify(f"{candidate.name}_{name}")
        candidate_profile_path = profiles_dir / f"{candidate_profile_name}.yaml"
        if candidate_profile_path.exists() and not force:
            raise FileExistsError(
                f"Candidate grid profile already exists: {candidate_profile_path}. Pass --force to overwrite."
            )

        payload = deepcopy(candidate.payload)
        payload["name"] = candidate_profile_name
        payload["description"] = f"Candidate {candidate.name} under teacher-suite scenario {name}"
        payload.setdefault("command", {})
        payload["command"].update(command)
        payload.setdefault("logging", {})
        payload["logging"].update(
            {
                "enabled": True,
                "format": "mcap",
                "every_n": 1,
                "prefix": f"grid_{candidate_profile_name}",
            }
        )
        payload["command_grid"] = {
            "grid": grid_name,
            "candidate_base_profile": candidate.name,
            "teacher_suite_manifest": _container_data_path(manifest_path),
            "scenario": name,
            "tags": tags,
        }

        with candidate_profile_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

        scenarios.append(
            CommandGridScenario(
                name=name,
                tags=tags,
                command={str(k): float(v) for k, v in command.items()},
                steps=steps,
                seconds=seconds,
                teacher_profile_path=teacher_profile_path,
                candidate_profile_path=_container_data_path(candidate_profile_path),
                candidate_profile_name=candidate_profile_name,
            )
        )

    result = CommandGridManifest(
        ok=True,
        grid_name=grid_name,
        teacher_suite_manifest=_container_data_path(manifest_path),
        candidate_base_profile=candidate.name,
        output_dir=str(out),
        generated_at_utc=utc_stamp(),
        scenarios=scenarios,
    )
    _write_json(out / "command_grid_manifest.json", result.to_dict())
    return result


def _find_comparison_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(path.glob("*/rollout_comparison.json")) + sorted(path.glob("rollout_comparison.json"))


def summarize_command_grid_comparisons(
    comparisons_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> CommandGridComparisonSummary:
    """Aggregate per-scenario rollout_comparison.json files into one grid report."""

    root = Path(comparisons_dir)
    paths = _find_comparison_jsons(root)
    if not paths:
        raise FileNotFoundError(f"No rollout_comparison.json files found under {root}")

    scenarios: list[CommandGridComparisonScenario] = []
    warnings: list[str] = []
    for path in paths:
        payload = _load_json(path)
        candidate = payload.get("candidate") or {}
        comparison = payload.get("comparison") or {}
        scenario_name = path.parent.name if path.name == "rollout_comparison.json" else path.stem
        # Prefer explicit command-grid scenario when available in the output dir
        scenario_name = slugify(scenario_name.replace("comparison_", ""))
        ok = bool(payload.get("ok"))
        scenarios.append(
            CommandGridComparisonScenario(
                name=scenario_name,
                ok=ok,
                comparison_path=str(path),
                teacher_log=str(payload.get("reference_log") or "") or None,
                candidate_log=str(payload.get("candidate_log") or "") or None,
                forward_ratio=_float_or_none(comparison.get("forward_ratio")),
                speed_ratio=_float_or_none(comparison.get("forward_speed_ratio")),
                lateral_abs=_float_or_none(candidate.get("lateral_abs")),
                reset_count=int(candidate.get("reset_count")) if candidate.get("reset_count") is not None else None,
                action_abs_max=_float_or_none(candidate.get("action_abs_max")),
                errors=[str(item) for item in (payload.get("errors") or [])],
                warnings=[str(item) for item in (payload.get("warnings") or [])],
            )
        )
        warnings.extend(str(item) for item in (payload.get("warnings") or []))

    passed = sum(1 for item in scenarios if item.ok)
    failed = len(scenarios) - passed
    out = Path(output_dir) if output_dir is not None else root
    result = CommandGridComparisonSummary(
        ok=failed == 0,
        output_dir=str(out),
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        scenario_count=len(scenarios),
        passed_count=passed,
        failed_count=failed,
        scenarios=scenarios,
        warnings=sorted(set(warnings)),
    )
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "command_grid_summary.json", result.to_dict())
    (out / "command_grid_report.md").write_text(render_command_grid_report(result), encoding="utf-8")
    return result


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"


def render_command_grid_report(result: CommandGridComparisonSummary) -> str:
    lines = [
        "# Soridormi command-grid rollout comparison",
        "",
        f"Result: {'PASS' if result.ok else 'FAIL'}",
        f"Generated: {result.generated_at_utc}",
        f"Scenarios: {result.scenario_count}",
        f"Passed: {result.passed_count}",
        f"Failed: {result.failed_count}",
        "",
        "| Scenario | Result | Forward ratio | Speed ratio | Lateral abs | Resets | Action abs max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.scenarios:
        lines.append(
            f"| {item.name} | {'PASS' if item.ok else 'FAIL'} | {_fmt(item.forward_ratio)} | "
            f"{_fmt(item.speed_ratio)} | {_fmt(item.lateral_abs)} | "
            f"{item.reset_count if item.reset_count is not None else 'n/a'} | {_fmt(item.action_abs_max)} |"
        )
    failed = [item for item in result.scenarios if not item.ok]
    if failed:
        lines.extend(["", "## Failures", ""])
        for item in failed:
            lines.append(f"### {item.name}")
            lines.extend(f"- {error}" for error in item.errors) if item.errors else lines.append("- unknown failure")
            lines.append("")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def _print_manifest(result: CommandGridManifest) -> None:
    print("Soridormi command-grid profile generator")
    print("=========================================")
    print(f"Grid: {result.grid_name}")
    print(f"Candidate: {result.candidate_base_profile}")
    print(f"Output: {result.output_dir}")
    print(f"Scenarios: {len(result.scenarios)}")
    for scenario in result.scenarios:
        command = scenario.command
        print(
            f"  - {scenario.name}: profile={scenario.candidate_profile_path} steps={scenario.steps} "
            f"cmd=(x={command.get('x', 0.0)}, y={command.get('y', 0.0)}, yaw={command.get('yaw', 0.0)})"
        )


def _print_summary(result: CommandGridComparisonSummary) -> None:
    print(render_command_grid_report(result))
    print(f"Wrote command-grid artifacts to: {result.output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Command-grid teacher-vs-candidate rollout utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate candidate profiles matching a teacher suite manifest.")
    generate.add_argument("candidate_profile")
    generate.add_argument("--teacher-manifest", default="data/teacher_suites/open_duck_teacher_v1/teacher_suite_manifest.json")
    generate.add_argument("--output-dir", default="data/command_grids/open_duck_teacher_v1")
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--json", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize per-scenario rollout comparisons.")
    summarize.add_argument("comparisons_dir")
    summarize.add_argument("--output-dir")
    summarize.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = build_candidate_command_grid(
                args.teacher_manifest,
                args.candidate_profile,
                output_dir=args.output_dir,
                force=args.force,
            )
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            else:
                _print_manifest(result)
            return 0
        if args.command == "summarize":
            result = summarize_command_grid_comparisons(args.comparisons_dir, output_dir=args.output_dir)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            else:
                _print_summary(result)
            return 0
    except Exception as exc:
        payload = {"ok": False, "errors": [repr(exc)]}
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Soridormi command-grid utility")
            print("===============================")
            print("Result: FAILED")
            print(f"Error: {exc!r}")
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
