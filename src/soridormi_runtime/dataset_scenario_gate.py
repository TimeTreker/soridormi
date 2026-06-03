from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from soridormi_runtime.dataset_coverage_report import (
    _command_value,
    _dataset_tags,
    _failure_flags,
    _float_or_none,
    _iter_jsonl,
    _resolve_input_paths,
    _terrain_type,
)
from soridormi_runtime.scenario_curriculum import (
    COLLECTOR_READY_STATUSES,
    DEFAULT_SCENARIO_MANIFEST,
    ScenarioDefinition,
    iter_scenarios,
)
from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file
from soridormi_runtime.training_dataset_prepare import validate_training_sample

DATASET_SCENARIO_GATE_SCHEMA_VERSION = 1
DEFAULT_COMMAND_SOURCE = "applied_command"
DEFAULT_MIN_COMMAND_RANGE_FRACTION = 0.20
DEFAULT_MIN_SAMPLES_PER_SCENARIO = 1
DEFAULT_MAX_FAILURE_RATIO = 0.50
DEFAULT_REQUIRED_COMMANDS = ("vx_mps", "vy_mps", "yaw_radps")
SUPPORTED_COMMAND_SOURCES = ("applied_command", "desired_command", "policy_command")
SUPPORTED_LOCOMOTION_SKILLS = frozenset({"walk_velocity", "curve_walk", "turn_in_place", "stand", "stop", "stand_idle"})


@dataclass
class CommandGateStats:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    span: float | None = None
    manifest_minimum: float | None = None
    manifest_maximum: float | None = None
    manifest_span: float | None = None
    covered_fraction: float | None = None


@dataclass
class ScenarioGateCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class ScenarioDatasetGateEntry:
    scenario_id: str
    title: str | None
    status: str | None
    required: bool
    known_in_manifest: bool
    sample_count: int
    split_counts: dict[str, int]
    skill_counts: dict[str, int]
    terrain_counts: dict[str, int]
    tag_counts: dict[str, int]
    ramp_name_counts: dict[str, int]
    ramp_alpha_count: int
    ramp_alpha_minimum: float | None
    ramp_alpha_maximum: float | None
    task_context_count: int
    environment_context_count: int
    failure_flag_count: int
    fall_count: int
    stuck_count: int
    terminated_count: int
    failure_count: int
    failure_ratio: float
    command_stats: dict[str, CommandGateStats]
    checks: list[ScenarioGateCheck] = field(default_factory=list)
    ok: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DatasetScenarioGateResult:
    ok: bool
    inputs: list[str]
    output_dir: str
    summary_path: str
    report_path: str
    scenario_manifest: str
    required_scenarios: list[str]
    present_scenarios: list[str]
    min_samples_per_scenario: int
    command_source: str
    required_commands: list[str]
    min_command_range_fraction: float
    require_ramp_alpha: bool
    require_task_context: bool
    require_environment_context: bool
    require_failure_flags: bool
    max_failure_ratio: float | None
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    scenario_results: list[ScenarioDatasetGateEntry]
    sha256_by_input: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _normalise_csv(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            cleaned = part.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _scenario_map(path: str | Path) -> dict[str, ScenarioDefinition]:
    return {scenario.id: scenario for scenario in iter_scenarios(path)}


def _default_required_ready_locomotion(path: str | Path) -> list[str]:
    out: list[str] = []
    for scenario in iter_scenarios(path):
        if scenario.status not in COLLECTOR_READY_STATUSES:
            continue
        if scenario.primary_skill not in SUPPORTED_LOCOMOTION_SKILLS:
            continue
        out.append(scenario.id)
    return out


class _ScenarioAccumulator:
    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.sample_count = 0
        self.split_counts: dict[str, int] = {}
        self.skill_counts: dict[str, int] = {}
        self.terrain_counts: dict[str, int] = {}
        self.tag_counts: dict[str, int] = {}
        self.ramp_name_counts: dict[str, int] = {}
        self.ramp_alpha_values: list[float] = []
        self.task_context_count = 0
        self.environment_context_count = 0
        self.failure_flag_count = 0
        self.fall_count = 0
        self.stuck_count = 0
        self.terminated_count = 0
        self.failure_count = 0
        self.command_values: dict[str, dict[str, list[float]]] = {
            source: {key: [] for key in DEFAULT_REQUIRED_COMMANDS} for source in SUPPORTED_COMMAND_SOURCES
        }

    def add(self, split_name: str, sample: dict[str, Any]) -> None:
        self.sample_count += 1
        _count(self.split_counts, split_name)
        _count(self.skill_counts, str(sample.get("skill_id") or "unknown"))
        _count(self.terrain_counts, _terrain_type(sample))
        for tag in _dataset_tags(sample):
            _count(self.tag_counts, tag)
        ramp_name = str(sample.get("command_ramp_name") or sample.get("ramp_name") or "unknown")
        _count(self.ramp_name_counts, ramp_name)
        alpha = _float_or_none(sample.get("command_ramp_alpha"))
        if alpha is not None:
            self.ramp_alpha_values.append(alpha)
        if isinstance(sample.get("task_context"), dict) and sample["task_context"]:
            self.task_context_count += 1
        if isinstance(sample.get("environment_context"), dict) and sample["environment_context"]:
            self.environment_context_count += 1
        if _has_failure_metadata(sample):
            self.failure_flag_count += 1
        fall, terminated, stuck, failure = _failure_flags(sample)
        self.fall_count += int(fall)
        self.terminated_count += int(terminated)
        self.stuck_count += int(stuck)
        self.failure_count += int(failure)
        for source in SUPPORTED_COMMAND_SOURCES:
            for key in DEFAULT_REQUIRED_COMMANDS:
                value = _command_value(sample, source, key)
                if value is not None:
                    self.command_values[source][key].append(value)


def _count(counts: dict[str, int], value: str) -> None:
    key = value if value else "unknown"
    counts[key] = counts.get(key, 0) + 1


def _has_failure_metadata(sample: dict[str, Any]) -> bool:
    keys = {
        "fall",
        "fallen",
        "fell",
        "fall_detected",
        "is_fall",
        "terminated",
        "done",
        "episode_terminated",
        "stuck",
        "is_stuck",
        "stuck_detected",
        "stuck_ratio",
        "failure",
        "failed",
        "is_failure",
        "success",
    }
    payloads = [sample]
    for key in ("metrics", "policy_debug", "failure_metadata", "evaluation", "eval_metrics"):
        value = sample.get(key)
        if isinstance(value, dict):
            payloads.append(value)
    return any(any(key in payload for key in keys) for payload in payloads)


def _command_gate_stats(values: list[float], manifest_range: tuple[float, float] | None) -> CommandGateStats:
    if values:
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
    else:
        minimum = maximum = span = None
    if manifest_range is None:
        return CommandGateStats(count=len(values), minimum=minimum, maximum=maximum, span=span)
    manifest_minimum, manifest_maximum = manifest_range
    manifest_span = manifest_maximum - manifest_minimum
    if span is None or manifest_span <= 0.0:
        covered_fraction = None
    else:
        clipped_low = max(minimum if minimum is not None else manifest_minimum, manifest_minimum)
        clipped_high = min(maximum if maximum is not None else manifest_maximum, manifest_maximum)
        covered_fraction = max(0.0, clipped_high - clipped_low) / manifest_span
    return CommandGateStats(
        count=len(values),
        minimum=minimum,
        maximum=maximum,
        span=span,
        manifest_minimum=manifest_minimum,
        manifest_maximum=manifest_maximum,
        manifest_span=manifest_span,
        covered_fraction=covered_fraction,
    )


def _manifest_command_range(scenario: ScenarioDefinition | None, key: str) -> tuple[float, float] | None:
    if scenario is None:
        return None
    try:
        return scenario.command_range(key)
    except Exception:
        return None


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _make_check(checks: list[ScenarioGateCheck], errors: list[str], name: str, ok: bool, detail: str) -> None:
    checks.append(ScenarioGateCheck(name=name, ok=ok, detail=detail))
    if not ok:
        errors.append(f"{name}: {detail}")


def _entry_from_accumulator(
    scenario_id: str,
    acc: _ScenarioAccumulator | None,
    *,
    scenario: ScenarioDefinition | None,
    required: bool,
    min_samples_per_scenario: int,
    command_source: str,
    required_commands: Sequence[str],
    min_command_range_fraction: float,
    require_ramp_alpha: bool,
    require_task_context: bool,
    require_environment_context: bool,
    require_failure_flags: bool,
    max_failure_ratio: float | None,
) -> ScenarioDatasetGateEntry:
    sample_count = acc.sample_count if acc else 0
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[ScenarioGateCheck] = []

    known = scenario is not None
    title = scenario.title if scenario is not None else None
    status = scenario.status if scenario is not None else None

    _make_check(checks, errors, "scenario_known", known, "scenario is present in the manifest" if known else "scenario is not present in the manifest")
    if required or sample_count > 0:
        _make_check(
            checks,
            errors,
            "min_samples_per_scenario",
            sample_count >= min_samples_per_scenario,
            f"{sample_count} samples >= required {min_samples_per_scenario}",
        )

    command_stats: dict[str, CommandGateStats] = {}
    for key in required_commands:
        values = acc.command_values.get(command_source, {}).get(key, []) if acc else []
        stats = _command_gate_stats(values, _manifest_command_range(scenario, key))
        command_stats[key] = stats
        if sample_count == 0:
            continue
        _make_check(checks, errors, f"{command_source}.{key}.present", stats.count > 0, f"{stats.count} command samples")
        if stats.manifest_span is not None and stats.manifest_span > 0.0 and min_command_range_fraction > 0.0:
            fraction = stats.covered_fraction if stats.covered_fraction is not None else 0.0
            _make_check(
                checks,
                errors,
                f"{command_source}.{key}.range_fraction",
                fraction >= min_command_range_fraction,
                f"covered {fraction:.3f} of manifest range; required {min_command_range_fraction:.3f}",
            )

    ramp_alpha_values = acc.ramp_alpha_values if acc else []
    if sample_count > 0 and require_ramp_alpha:
        _make_check(
            checks,
            errors,
            "ramp_alpha_present",
            len(ramp_alpha_values) == sample_count,
            f"{len(ramp_alpha_values)} / {sample_count} samples include command_ramp_alpha",
        )
    if sample_count > 0 and require_task_context:
        task_count = acc.task_context_count if acc else 0
        _make_check(
            checks,
            errors,
            "task_context_present",
            task_count == sample_count,
            f"{task_count} / {sample_count} samples include task_context",
        )
    if sample_count > 0 and require_environment_context:
        env_count = acc.environment_context_count if acc else 0
        _make_check(
            checks,
            errors,
            "environment_context_present",
            env_count == sample_count,
            f"{env_count} / {sample_count} samples include environment_context",
        )
    if sample_count > 0 and require_failure_flags:
        flag_count = acc.failure_flag_count if acc else 0
        _make_check(
            checks,
            errors,
            "failure_flags_present",
            flag_count == sample_count,
            f"{flag_count} / {sample_count} samples include fall/stuck/termination metadata",
        )

    failure_count = acc.failure_count if acc else 0
    failure_ratio = (failure_count / float(sample_count)) if sample_count else 0.0
    if sample_count > 0 and max_failure_ratio is not None:
        _make_check(
            checks,
            errors,
            "max_failure_ratio",
            failure_ratio <= max_failure_ratio,
            f"failure_ratio={failure_ratio:.3f} <= {max_failure_ratio:.3f}",
        )

    manifest_terrain = None
    if scenario is not None:
        env_context = scenario.environment_context
        if isinstance(env_context, dict):
            terrain = env_context.get("terrain_type")
            if terrain:
                manifest_terrain = str(terrain)
    terrain_counts = _sorted_counts(acc.terrain_counts) if acc else {}
    if sample_count > 0 and manifest_terrain and manifest_terrain not in terrain_counts:
        warnings.append(f"manifest terrain_type {manifest_terrain!r} was not observed in dataset terrain coverage")

    return ScenarioDatasetGateEntry(
        scenario_id=scenario_id,
        title=title,
        status=status,
        required=required,
        known_in_manifest=known,
        sample_count=sample_count,
        split_counts=_sorted_counts(acc.split_counts) if acc else {},
        skill_counts=_sorted_counts(acc.skill_counts) if acc else {},
        terrain_counts=terrain_counts,
        tag_counts=_sorted_counts(acc.tag_counts) if acc else {},
        ramp_name_counts=_sorted_counts(acc.ramp_name_counts) if acc else {},
        ramp_alpha_count=len(ramp_alpha_values),
        ramp_alpha_minimum=min(ramp_alpha_values) if ramp_alpha_values else None,
        ramp_alpha_maximum=max(ramp_alpha_values) if ramp_alpha_values else None,
        task_context_count=acc.task_context_count if acc else 0,
        environment_context_count=acc.environment_context_count if acc else 0,
        failure_flag_count=acc.failure_flag_count if acc else 0,
        fall_count=acc.fall_count if acc else 0,
        stuck_count=acc.stuck_count if acc else 0,
        terminated_count=acc.terminated_count if acc else 0,
        failure_count=failure_count,
        failure_ratio=failure_ratio,
        command_stats=command_stats,
        checks=checks,
        ok=not errors,
        warnings=warnings,
        errors=errors,
    )


def evaluate_dataset_scenario_gate(
    inputs: Iterable[str | Path] | str | Path,
    *,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    output_dir: str | Path | None = None,
    required_scenarios: Sequence[str] | None = None,
    require_ready_locomotion: bool = False,
    min_samples_per_scenario: int = DEFAULT_MIN_SAMPLES_PER_SCENARIO,
    command_source: str = DEFAULT_COMMAND_SOURCE,
    required_commands: Sequence[str] = DEFAULT_REQUIRED_COMMANDS,
    min_command_range_fraction: float = DEFAULT_MIN_COMMAND_RANGE_FRACTION,
    require_ramp_alpha: bool = True,
    require_task_context: bool = True,
    require_environment_context: bool = True,
    require_failure_flags: bool = True,
    max_failure_ratio: float | None = DEFAULT_MAX_FAILURE_RATIO,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    max_reported_issues: int = 50,
) -> DatasetScenarioGateResult:
    if command_source not in SUPPORTED_COMMAND_SOURCES:
        raise ValueError(f"unsupported command source {command_source!r}; expected one of {SUPPORTED_COMMAND_SOURCES}")
    input_values: Iterable[str | Path]
    if isinstance(inputs, (str, Path)):
        input_values = [inputs]
    else:
        input_values = inputs
    scenario_manifest_path = Path(scenario_manifest)
    scenarios = _scenario_map(scenario_manifest_path)
    requested = _normalise_csv(required_scenarios)
    if require_ready_locomotion:
        for item in _default_required_ready_locomotion(scenario_manifest_path):
            if item not in requested:
                requested.append(item)

    output = Path(output_dir) if output_dir is not None else Path("artifacts/dataset_coverage/scenario_gate")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "dataset_scenario_gate_summary.json"
    report_path = output / "dataset_scenario_gate_report.md"

    input_items, input_errors = _resolve_input_paths(input_values)
    errors = list(input_errors)
    warnings: list[str] = []
    sample_count = 0
    valid_sample_count = 0
    invalid_sample_count = 0
    sha_by_input: dict[str, str] = {}
    accs: dict[str, _ScenarioAccumulator] = {}

    if not input_items and not errors:
        errors.append("No dataset inputs were provided")

    for split_name, path in input_items:
        if not path.exists():
            errors.append(f"dataset input not found: {path}")
            continue
        sha_by_input[str(path)] = sha256_file(path)
        for line_number, sample, parse_error in _iter_jsonl(path):
            sample_count += 1
            if parse_error is not None or sample is None:
                invalid_sample_count += 1
                if len(errors) < max_reported_issues:
                    errors.append(f"{path}:{parse_error or f'line {line_number}: invalid sample'}")
                continue
            sample_errors, sample_warnings = validate_training_sample(
                sample,
                observation_size=observation_size,
                action_size=action_size,
            )
            if sample_errors:
                invalid_sample_count += 1
                if len(errors) < max_reported_issues:
                    for issue in sample_errors[: max(0, max_reported_issues - len(errors))]:
                        errors.append(f"{path}:line {line_number}: {issue}")
                continue
            valid_sample_count += 1
            if sample_warnings and len(warnings) < max_reported_issues:
                for warning in sample_warnings[: max(0, max_reported_issues - len(warnings))]:
                    warnings.append(f"{path}:line {line_number}: {warning}")
            scenario_id = str(sample.get("scenario_id") or "unknown")
            acc = accs.setdefault(scenario_id, _ScenarioAccumulator(scenario_id))
            acc.add(split_name, sample)

    required_set = set(requested)
    scenario_ids = sorted(set(accs) | required_set, key=lambda item: (0 if item in required_set else 1, item))
    scenario_results = [
        _entry_from_accumulator(
            scenario_id,
            accs.get(scenario_id),
            scenario=scenarios.get(scenario_id),
            required=scenario_id in required_set,
            min_samples_per_scenario=max(0, int(min_samples_per_scenario)),
            command_source=command_source,
            required_commands=list(required_commands),
            min_command_range_fraction=max(0.0, float(min_command_range_fraction)),
            require_ramp_alpha=require_ramp_alpha,
            require_task_context=require_task_context,
            require_environment_context=require_environment_context,
            require_failure_flags=require_failure_flags,
            max_failure_ratio=max_failure_ratio,
        )
        for scenario_id in scenario_ids
    ]
    for entry in scenario_results:
        for warning in entry.warnings:
            warnings.append(f"{entry.scenario_id}: {warning}")
        for error in entry.errors:
            errors.append(f"{entry.scenario_id}: {error}")

    if valid_sample_count == 0 and not errors:
        errors.append("No valid samples were found")

    result = DatasetScenarioGateResult(
        ok=not errors,
        inputs=[str(path) for _split, path in input_items],
        output_dir=str(output),
        summary_path=str(summary_path),
        report_path=str(report_path),
        scenario_manifest=str(scenario_manifest_path),
        required_scenarios=requested,
        present_scenarios=sorted(accs),
        min_samples_per_scenario=max(0, int(min_samples_per_scenario)),
        command_source=command_source,
        required_commands=list(required_commands),
        min_command_range_fraction=max(0.0, float(min_command_range_fraction)),
        require_ramp_alpha=require_ramp_alpha,
        require_task_context=require_task_context,
        require_environment_context=require_environment_context,
        require_failure_flags=require_failure_flags,
        max_failure_ratio=max_failure_ratio,
        sample_count=sample_count,
        valid_sample_count=valid_sample_count,
        invalid_sample_count=invalid_sample_count,
        scenario_results=scenario_results,
        sha256_by_input=sha_by_input,
        warnings=warnings[:max_reported_issues],
        errors=errors[:max_reported_issues],
    )
    payload = asdict(result)
    payload["schema_version"] = DATASET_SCENARIO_GATE_SCHEMA_VERSION
    payload["gate_type"] = "soridormi.policy_supervision.scenario_gate.v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report_path, result)
    return result


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def _write_markdown_report(path: Path, result: DatasetScenarioGateResult) -> None:
    lines = [
        "# Soridormi dataset scenario gate",
        "",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        f"Samples: **{result.valid_sample_count} valid** / {result.sample_count} total",
        f"Command source: `{result.command_source}`",
        f"Required scenarios: {', '.join(f'`{item}`' for item in result.required_scenarios) if result.required_scenarios else '_none_'}",
        "",
        "## Scenario results",
        "",
        "| Scenario | Required | Samples | Failure ratio | Result |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for entry in result.scenario_results:
        lines.append(
            f"| `{entry.scenario_id}` | {'yes' if entry.required else 'no'} | {entry.sample_count} | "
            f"{entry.failure_ratio:.3f} | {'OK' if entry.ok else 'FAILED'} |"
        )
    lines.extend(["", "## Command range coverage", ""])
    for entry in result.scenario_results:
        lines.extend([f"### `{entry.scenario_id}`", "", "| Command | Count | Min | Max | Covered fraction |", "| --- | ---: | ---: | ---: | ---: |"])
        for command in result.required_commands:
            stats = entry.command_stats.get(command, CommandGateStats())
            lines.append(
                f"| `{command}` | {stats.count} | {_fmt(stats.minimum)} | {_fmt(stats.maximum)} | {_fmt(stats.covered_fraction)} |"
            )
        if entry.errors:
            lines.extend(["", "Errors:"])
            lines.extend(f"- {error}" for error in entry.errors)
        if entry.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in entry.warnings)
        lines.append("")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_gate_summary(result: DatasetScenarioGateResult) -> None:
    print("Soridormi dataset scenario gate")
    print("================================")
    print(f"Inputs: {len(result.inputs)}")
    print(f"Output dir: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print(f"Report: {result.report_path}")
    print(f"Samples: {result.valid_sample_count} valid / {result.sample_count} total")
    print(f"Required scenarios: {', '.join(result.required_scenarios) if result.required_scenarios else 'none'}")
    for entry in result.scenario_results:
        print(
            f"- {entry.scenario_id}: {'OK' if entry.ok else 'FAILED'}; "
            f"samples={entry.sample_count}; failure_ratio={entry.failure_ratio:.3f}"
        )
    if result.errors:
        print("Errors:")
        for error in result.errors[:20]:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate Soridormi policy-supervision datasets against scenario manifest coverage requirements."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Dataset JSONL, prepared_manifest.json, or prepared directory")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate")
    parser.add_argument("--require-ready-locomotion", action="store_true", help="Require every registry-ready locomotion scenario")
    parser.add_argument("--min-samples-per-scenario", type=int, default=DEFAULT_MIN_SAMPLES_PER_SCENARIO)
    parser.add_argument("--command-source", choices=SUPPORTED_COMMAND_SOURCES, default=DEFAULT_COMMAND_SOURCE)
    parser.add_argument("--required-command", action="append", default=[], help="Command field to range-check; repeat or comma-separate")
    parser.add_argument("--min-command-range-fraction", type=float, default=DEFAULT_MIN_COMMAND_RANGE_FRACTION)
    parser.add_argument("--no-require-ramp-alpha", action="store_true")
    parser.add_argument("--no-require-task-context", action="store_true")
    parser.add_argument("--no-require-environment-context", action="store_true")
    parser.add_argument("--no-require-failure-flags", action="store_true")
    parser.add_argument("--max-failure-ratio", type=float, default=DEFAULT_MAX_FAILURE_RATIO)
    parser.add_argument("--allow-any-failure-ratio", action="store_true")
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args(argv)

    required_commands = _normalise_csv(args.required_command) or list(DEFAULT_REQUIRED_COMMANDS)
    result = evaluate_dataset_scenario_gate(
        args.inputs,
        scenario_manifest=args.scenario_manifest,
        output_dir=args.output_dir,
        required_scenarios=_normalise_csv(args.require_scenario),
        require_ready_locomotion=args.require_ready_locomotion,
        min_samples_per_scenario=args.min_samples_per_scenario,
        command_source=args.command_source,
        required_commands=required_commands,
        min_command_range_fraction=args.min_command_range_fraction,
        require_ramp_alpha=not args.no_require_ramp_alpha,
        require_task_context=not args.no_require_task_context,
        require_environment_context=not args.no_require_environment_context,
        require_failure_flags=not args.no_require_failure_flags,
        max_failure_ratio=None if args.allow_any_failure_ratio else args.max_failure_ratio,
        observation_size=args.observation_size,
        action_size=args.action_size,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_gate_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
