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

_ALLOWED_COMMAND_KEYS = {
    "x",
    "y",
    "yaw",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "ramp_seconds",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError("scenario/profile name cannot be empty")
    return text


@dataclass
class TeacherScenarioResult:
    name: str
    profile_name: str
    profile_path: str
    steps: int
    seconds: float | None
    command: dict[str, float]
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TeacherSuiteResult:
    ok: bool
    suite_name: str
    suite_path: str
    output_dir: str
    manifest_path: str
    generated_at_utc: str
    scenarios: list[TeacherScenarioResult]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _float_command(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"command.{key} must be numeric, got {value!r}") from exc


def _scenario_command(raw: dict[str, Any]) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("scenario.command must be a mapping")
    unknown = sorted(set(raw) - _ALLOWED_COMMAND_KEYS)
    if unknown:
        raise ValueError(f"unsupported command key(s): {', '.join(unknown)}")
    command = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "neck_pitch": 0.0,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "ramp_seconds": 0.0,
    }
    for key, value in raw.items():
        command[key] = _float_command(value, key=key)
    return command


def _relative_to_cwd(path_text: str, *, suite_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return suite_path.parent / path


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


def build_teacher_suite(
    suite_path: str | Path,
    *,
    output_dir: str | Path,
    force: bool = False,
) -> TeacherSuiteResult:
    suite_file = Path(suite_path)
    suite = _load_yaml_mapping(suite_file)
    suite_name = slugify(str(suite.get("name") or suite_file.stem))
    base_profile_path = _relative_to_cwd(str(suite.get("base_profile", "configs/policies/open_duck_forward.yaml")), suite_path=suite_file)
    base_profile = _load_yaml_mapping(base_profile_path)

    scenarios_raw = suite.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("teacher suite must define a non-empty scenarios list")

    out = Path(output_dir)
    profiles_dir = out / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    default_steps = int(suite.get("default_steps", 1000))
    default_seconds_raw = suite.get("default_seconds")
    default_seconds = float(default_seconds_raw) if default_seconds_raw not in {None, ""} else None
    generated: list[TeacherScenarioResult] = []
    warnings: list[str] = []

    for item in scenarios_raw:
        if not isinstance(item, dict):
            raise ValueError("each scenario must be a mapping")
        scenario_name = slugify(str(item.get("name") or "scenario"))
        profile_name = slugify(str(item.get("profile_name") or f"{suite_name}_{scenario_name}"))
        profile_path = profiles_dir / f"{profile_name}.yaml"
        if profile_path.exists() and not force:
            raise FileExistsError(f"Profile already exists: {profile_path}. Pass --force to overwrite.")

        command = _scenario_command(item.get("command", {}))
        profile = deepcopy(base_profile)
        profile["name"] = profile_name
        profile["description"] = str(item.get("description") or f"Teacher scenario {scenario_name} from {suite_name}")
        profile.setdefault("command", {})
        profile["command"].update(command)
        profile.setdefault("logging", {})
        profile["logging"].update({"enabled": True, "format": "mcap", "every_n": 1, "prefix": f"teacher_{profile_name}"})
        profile["teacher_suite"] = {
            "suite": suite_name,
            "scenario": scenario_name,
            "tags": list(item.get("tags") or []),
        }

        with profile_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(profile, f, sort_keys=False)

        steps = int(item.get("steps", default_steps))
        seconds_raw = item.get("seconds", default_seconds)
        seconds = float(seconds_raw) if seconds_raw not in {None, ""} else None
        generated.append(
            TeacherScenarioResult(
                name=scenario_name,
                profile_name=profile_name,
                profile_path=_container_data_path(profile_path),
                steps=steps,
                seconds=seconds,
                command=command,
                description=str(item.get("description") or ""),
                tags=[str(tag) for tag in (item.get("tags") or [])],
            )
        )
        if "bow" in set(generated[-1].tags):
            warnings.append(
                "head_bow_cue uses head/neck command fields only; full-body bow should be a separate Soridormi skill."
            )

    manifest_path = out / "teacher_suite_manifest.json"
    result = TeacherSuiteResult(
        ok=True,
        suite_name=suite_name,
        suite_path=str(suite_file),
        output_dir=str(out),
        manifest_path=str(manifest_path),
        generated_at_utc=utc_stamp(),
        scenarios=generated,
        warnings=sorted(set(warnings)),
    )
    manifest = asdict(result)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return result


def _print_text(result: TeacherSuiteResult) -> None:
    print("Soridormi teacher suite generator")
    print("==================================")
    print(f"Suite: {result.suite_name}")
    print(f"Output: {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Scenarios: {len(result.scenarios)}")
    for scenario in result.scenarios:
        command = scenario.command
        print(
            f"  - {scenario.name}: profile={scenario.profile_path} steps={scenario.steps} "
            f"cmd=(x={command['x']}, y={command['y']}, yaw={command['yaw']})"
        )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate command-conditioned teacher policy profiles.")
    parser.add_argument("--suite", default="configs/teacher_suites/open_duck_teacher_v1.yaml")
    parser.add_argument("--output-dir", default="data/teacher_suites/open_duck_teacher_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = build_teacher_suite(args.suite, output_dir=args.output_dir, force=args.force)
    except Exception as exc:
        payload = {
            "ok": False,
            "errors": [repr(exc)],
            "suite": args.suite,
            "output_dir": args.output_dir,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Soridormi teacher suite generator")
            print("==================================")
            print("Result: FAILED")
            print(f"Error: {exc!r}")
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
