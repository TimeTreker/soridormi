from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.bc_training_contract import (
    CONTEXT_SAMPLE_TYPE,
    DEFAULT_CONTRACT_PATH,
    load_and_validate_contract,
    validate_sample_against_contract,
)
from soridormi_runtime.dataset_coverage_report import _resolve_input_paths
from soridormi_runtime.scenario_curriculum import DEFAULT_SCENARIO_MANIFEST, ScenarioDefinition, iter_scenarios
from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file

CONTEXT_BC_EXPORT_SCHEMA_VERSION = 1
DEFAULT_CONTEXT_OUTPUT = Path("/data/training_datasets/context_bc/context_bc_dataset.jsonl")


@dataclass
class ContextBcExportResult:
    ok: bool
    output_path: str
    manifest_path: str
    input_paths: list[str]
    sample_count: int
    converted_count: int
    skipped_count: int
    invalid_output_count: int
    scenario_counts: dict[str, int] = field(default_factory=dict)
    skill_counts: dict[str, int] = field(default_factory=dict)
    output_sha256: str | None = None
    output_written: bool = False
    contract_path: str = str(DEFAULT_CONTRACT_PATH)
    scenario_manifest_path: str = str(DEFAULT_SCENARIO_MANIFEST)
    include_short_history: bool = True
    strict_context: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _float_or_none(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None


def _numeric_vector(value: Any, *, size: int, field_name: str) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{field_name} must be a list"
    if len(value) != size:
        return None, f"{field_name} size {len(value)} != expected {size}"
    out: list[float] = []
    for index, item in enumerate(value):
        if not _is_number(item):
            return None, f"{field_name} contains non-finite/non-numeric value at index {index}"
        out.append(float(item))
    return out, None


def _load_scenario_map(path: str | Path) -> tuple[dict[str, ScenarioDefinition], list[str]]:
    errors: list[str] = []
    scenarios: dict[str, ScenarioDefinition] = {}
    try:
        for scenario in iter_scenarios(path):
            scenarios[scenario.id] = scenario
    except Exception as exc:
        errors.append(f"could not load scenario manifest {path}: {exc}")
    return scenarios, errors


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"line {line_number}: invalid JSON: {exc}"
                continue
            if not isinstance(payload, dict):
                yield line_number, None, f"line {line_number}: expected JSON object"
                continue
            yield line_number, payload, None


def _observation_from_sample(sample: dict[str, Any]) -> Any:
    robot_state = sample.get("robot_state")
    if isinstance(robot_state, dict) and "observation" in robot_state:
        return robot_state.get("observation")
    return sample.get("observation")


def _action_from_sample(sample: dict[str, Any]) -> Any:
    if "teacher_action" in sample:
        return sample.get("teacher_action")
    return sample.get("action")


def _command_list_value(values: Any, index: int) -> float | None:
    if isinstance(values, list) and len(values) > index:
        return _float_or_none(values[index])
    return None


def _command_payload_value(payload: Any, aliases: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for alias in aliases:
        if alias in payload:
            value = _float_or_none(payload.get(alias))
            if value is not None:
                return value
    return None


def _extract_command(sample: dict[str, Any], *, preferred_sources: Iterable[str]) -> tuple[dict[str, float] | None, str | None]:
    aliases = {
        "vx_mps": ("vx_mps", "vx", "x_velocity", "x"),
        "vy_mps": ("vy_mps", "vy", "y_velocity", "y"),
        "yaw_radps": ("yaw_radps", "yaw", "yaw_velocity"),
    }
    vector_sources = {
        "policy_command": sample.get("policy_command"),
        "policy_command_target": sample.get("policy_command_target"),
    }
    for source in preferred_sources:
        payload = sample.get(source)
        if isinstance(payload, dict):
            values = {key: _command_payload_value(payload, alias_list) for key, alias_list in aliases.items()}
            if all(value is not None for value in values.values()):
                return {key: float(value) for key, value in values.items() if value is not None}, source
        if source in vector_sources:
            vector = vector_sources[source]
            vx = _command_list_value(vector, 0)
            vy = _command_list_value(vector, 1)
            yaw = _command_list_value(vector, 2)
            if vx is not None and vy is not None and yaw is not None:
                return {"vx_mps": vx, "vy_mps": vy, "yaw_radps": yaw}, source
    return None, None


def _normalise_desired_command(sample: dict[str, Any]) -> tuple[dict[str, float] | None, str | None]:
    return _extract_command(
        sample,
        preferred_sources=("desired_command", "policy_command_target", "applied_command", "policy_command"),
    )


def _normalise_applied_command(sample: dict[str, Any]) -> tuple[dict[str, float] | None, str | None]:
    return _extract_command(
        sample,
        preferred_sources=("applied_command", "policy_command", "desired_command", "policy_command_target"),
    )


def _scenario_id(sample: dict[str, Any]) -> str:
    value = sample.get("scenario_id")
    return str(value) if value is not None and str(value) else "unknown_scenario"


def _skill_id(sample: dict[str, Any], scenario: ScenarioDefinition | None, task_context: dict[str, Any] | None = None) -> str:
    for candidate in (
        sample.get("skill_id"),
        (task_context or {}).get("skill_id"),
        scenario.primary_skill if scenario is not None else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown_skill"


def _scenario_task_context(scenario: ScenarioDefinition | None) -> dict[str, Any]:
    if scenario is None:
        return {}
    return dict(scenario.task_context)


def _scenario_environment_context(scenario: ScenarioDefinition | None) -> dict[str, Any]:
    if scenario is None:
        return {}
    return dict(scenario.environment_context)


def _normalise_task_context(sample: dict[str, Any], scenario: ScenarioDefinition | None) -> dict[str, Any]:
    context: dict[str, Any] = _scenario_task_context(scenario)
    raw = sample.get("task_context")
    if isinstance(raw, dict):
        context.update(raw)
    skill = _skill_id(sample, scenario, context)
    context["skill_id"] = skill
    if scenario is not None:
        context.setdefault("scenario_family", scenario.family)
        context.setdefault("scenario_status", scenario.status)
    return context


def _normalise_environment_context(sample: dict[str, Any], scenario: ScenarioDefinition | None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    context: dict[str, Any] = _scenario_environment_context(scenario)
    raw = sample.get("environment_context")
    if isinstance(raw, dict):
        context.update(raw)
    terrain = sample.get("terrain_type")
    if isinstance(terrain, str) and terrain:
        context["terrain_type"] = terrain
    if "terrain_type" not in context or not isinstance(context.get("terrain_type"), str) or not context.get("terrain_type"):
        context["terrain_type"] = "unknown"
        warnings.append("environment_context.terrain_type missing; defaulted to unknown")
    return context, warnings


def _failure_flags(sample: dict[str, Any]) -> dict[str, bool]:
    raw = sample.get("failure_flags")
    if isinstance(raw, dict):
        return {
            "fallen": bool(raw.get("fallen", raw.get("fall", False))),
            "stuck": bool(raw.get("stuck", False)),
            "terminated": bool(raw.get("terminated", raw.get("done", False))),
        }
    policy_debug = sample.get("policy_debug") if isinstance(sample.get("policy_debug"), dict) else {}
    return {
        "fallen": bool(sample.get("fallen", sample.get("fall", False))),
        "stuck": bool(sample.get("stuck", False)),
        "terminated": bool(sample.get("terminated", sample.get("done", policy_debug.get("terminated", False)))),
    }


def _rollout_id(sample: dict[str, Any], *, fallback_scenario_id: str, input_label: str) -> str:
    value = sample.get("rollout_id")
    if isinstance(value, str) and value:
        return value
    episode = sample.get("episode_index")
    if episode is not None:
        return f"{fallback_scenario_id}:episode_{episode}"
    return f"{fallback_scenario_id}:{input_label}"


def _step_index(sample: dict[str, Any], fallback: int) -> int:
    for key in ("step_index", "timestep", "global_step_index", "episode_step_index"):
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return int(fallback)


def _context_sample_from_source(
    sample: dict[str, Any],
    *,
    input_label: str,
    input_path: Path,
    line_number: int,
    scenario_map: dict[str, ScenarioDefinition],
    previous_by_rollout: dict[str, dict[str, Any]],
    include_short_history: bool,
    strict_context: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    observation, obs_error = _numeric_vector(_observation_from_sample(sample), size=DEFAULT_OBSERVATION_SIZE, field_name="observation")
    if obs_error:
        errors.append(obs_error)
    action, action_error = _numeric_vector(_action_from_sample(sample), size=DEFAULT_ACTION_SIZE, field_name="teacher_action")
    if action_error:
        errors.append(action_error)

    scenario_id = _scenario_id(sample)
    scenario = scenario_map.get(scenario_id)
    if scenario is None and strict_context:
        errors.append(f"scenario_id {scenario_id!r} not found in scenario manifest")
    elif scenario is None and scenario_id != "unknown_scenario":
        warnings.append(f"scenario_id {scenario_id!r} not found in scenario manifest; using row context/defaults")

    desired_command, desired_source = _normalise_desired_command(sample)
    if desired_command is None:
        errors.append("could not resolve desired_command from desired_command, policy_command_target, applied_command, or policy_command")
    applied_command, applied_source = _normalise_applied_command(sample)
    if applied_command is None and desired_command is not None:
        applied_command = dict(desired_command)
        applied_source = "desired_command"
        warnings.append("applied_command missing; copied desired_command")

    task_context = _normalise_task_context(sample, scenario)
    environment_context, env_warnings = _normalise_environment_context(sample, scenario)
    warnings.extend(env_warnings)

    if strict_context and task_context.get("skill_id") == "unknown_skill":
        errors.append("skill_id could not be resolved from row or scenario manifest")
    skill_id = str(task_context.get("skill_id", "unknown_skill"))
    rollout_id = _rollout_id(sample, fallback_scenario_id=scenario_id, input_label=input_label)
    step_index = _step_index(sample, fallback=line_number - 1)

    if errors:
        return None, errors, warnings

    assert observation is not None
    assert action is not None
    assert desired_command is not None
    assert applied_command is not None

    converted: dict[str, Any] = {
        "sample_type": CONTEXT_SAMPLE_TYPE,
        "schema_version": CONTEXT_BC_EXPORT_SCHEMA_VERSION,
        "source_sample_type": sample.get("sample_type"),
        "source_dataset": str(input_path),
        "source_line_number": int(line_number),
        "source_log": sample.get("source_log"),
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "step_index": step_index,
        "timestep": step_index,
        "episode_index": sample.get("episode_index"),
        "episode_step_index": sample.get("episode_step_index"),
        "skill_id": skill_id,
        "robot_state": {"observation": observation},
        "desired_command": desired_command,
        "applied_command": applied_command,
        "task_context": task_context,
        "environment_context": environment_context,
        "teacher_action": action,
        "failure_flags": _failure_flags(sample),
        "adapter_debug": {
            "adapter": "soridormi_runtime.context_bc_dataset_export",
            "desired_command_source": desired_source,
            "applied_command_source": applied_source,
            "scenario_manifest_match": scenario is not None,
        },
    }
    for key in ("command_ramp_alpha", "command_ramp_name", "command_ramp_steps", "command_segment_index", "command_segment_id"):
        if key in sample:
            converted[key] = sample[key]
    if sample.get("robot_time") is not None:
        converted["robot_time"] = sample.get("robot_time")
    if sample.get("next_robot_time") is not None:
        converted["next_robot_time"] = sample.get("next_robot_time")
    if isinstance(sample.get("scenario_dataset_tags"), list):
        converted["scenario_dataset_tags"] = sample.get("scenario_dataset_tags")
    elif scenario is not None and scenario.dataset_tags:
        converted["scenario_dataset_tags"] = scenario.dataset_tags

    if include_short_history:
        previous = previous_by_rollout.get(rollout_id)
        history: dict[str, Any] = {}
        if previous is not None:
            history["previous_action"] = previous.get("teacher_action")
            history["previous_command"] = previous.get("applied_command")
        else:
            history["previous_action"] = [0.0] * DEFAULT_ACTION_SIZE
            history["previous_command"] = {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_radps": 0.0}
        converted["short_history"] = history
    previous_by_rollout[rollout_id] = converted
    return converted, errors, warnings


def _manifest_path_for(output: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    suffix = output.suffix or ".jsonl"
    return output.with_suffix(suffix + ".manifest.json")


def _write_manifest(result: ContextBcExportResult, *, output: Path, manifest: Path, input_digest: dict[str, str | None]) -> None:
    payload = asdict(result)
    payload.update(
        {
            "schema_version": CONTEXT_BC_EXPORT_SCHEMA_VERSION,
            "created_utc": utc_stamp(),
            "dataset_type": CONTEXT_SAMPLE_TYPE,
            "output_path": str(output),
            "input_sha256": input_digest,
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(result: ContextBcExportResult, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Soridormi Context BC Dataset Export Report",
        "",
        f"Result: **{'PASS' if result.ok else 'FAIL'}**",
        "",
        f"Output: `{result.output_path}`",
        f"Manifest: `{result.manifest_path}`",
        f"Samples read: `{result.sample_count}`",
        f"Converted: `{result.converted_count}`",
        f"Skipped: `{result.skipped_count}`",
        f"Invalid converted rows: `{result.invalid_output_count}`",
        f"Output updated: `{result.output_written}`",
        f"SHA256: `{result.output_sha256}`",
        "",
        "## Scenario counts",
        "",
    ]
    if result.scenario_counts:
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(result.scenario_counts.items()))
    else:
        lines.append("No converted scenarios.")
    lines.extend(["", "## Skill counts", ""])
    if result.skill_counts:
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(result.skill_counts.items()))
    else:
        lines.append("No converted skills.")
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in result.errors)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in result.warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def export_context_bc_dataset(
    inputs: Iterable[str | Path],
    *,
    output_path: str | Path = DEFAULT_CONTEXT_OUTPUT,
    manifest_path: str | Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    include_short_history: bool = True,
    strict_context: bool = False,
    max_reported_issues: int = 80,
    max_samples: int | None = None,
) -> ContextBcExportResult:
    output = Path(output_path)
    manifest = _manifest_path_for(output, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp")

    input_specs, input_errors = _resolve_input_paths(inputs)
    scenario_map, scenario_errors = _load_scenario_map(scenario_manifest)
    contract, contract_result = load_and_validate_contract(contract_path)
    errors: list[str] = list(input_errors) + list(scenario_errors)
    warnings: list[str] = []
    if not contract_result.ok:
        errors.extend(f"contract: {item}" for item in contract_result.errors)
    if contract is None:
        contract = {}

    input_paths = [str(path) for _label, path in input_specs]
    input_digest: dict[str, str | None] = {}
    for path in (path for _label, path in input_specs):
        input_digest[str(path)] = sha256_file(path) if path.exists() else None
        if not path.exists():
            errors.append(f"input JSONL not found: {path}")

    sample_count = 0
    converted_count = 0
    skipped_count = 0
    invalid_output_count = 0
    scenario_counts: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    previous_by_rollout: dict[str, dict[str, Any]] = {}

    with tmp_output.open("w", encoding="utf-8") as handle:
        for label, path in input_specs:
            if max_samples is not None and sample_count >= int(max_samples):
                break
            if not path.exists():
                continue
            for line_number, sample, parse_error in _iter_jsonl(path):
                if max_samples is not None and sample_count >= int(max_samples):
                    break
                sample_count += 1
                if parse_error is not None or sample is None:
                    skipped_count += 1
                    if len(errors) < max_reported_issues:
                        errors.append(f"{path}: {parse_error or 'invalid sample'}")
                    continue
                converted, row_errors, row_warnings = _context_sample_from_source(
                    sample,
                    input_label=label,
                    input_path=path,
                    line_number=line_number,
                    scenario_map=scenario_map,
                    previous_by_rollout=previous_by_rollout,
                    include_short_history=include_short_history,
                    strict_context=strict_context,
                )
                if row_warnings and len(warnings) < max_reported_issues:
                    warnings.extend(f"{path}:{line_number}: {warning}" for warning in row_warnings[: max(0, max_reported_issues - len(warnings))])
                if row_errors or converted is None:
                    skipped_count += 1
                    if len(errors) < max_reported_issues:
                        errors.extend(f"{path}:{line_number}: {error}" for error in row_errors[: max(0, max_reported_issues - len(errors))])
                    continue
                sample_errors, sample_warnings, _kind = validate_sample_against_contract(converted, contract, allow_legacy=False)
                if sample_warnings and len(warnings) < max_reported_issues:
                    warnings.extend(f"{path}:{line_number}: contract: {warning}" for warning in sample_warnings[: max(0, max_reported_issues - len(warnings))])
                if sample_errors:
                    invalid_output_count += 1
                    skipped_count += 1
                    if len(errors) < max_reported_issues:
                        errors.extend(f"{path}:{line_number}: converted sample invalid: {error}" for error in sample_errors[: max(0, max_reported_issues - len(errors))])
                    continue
                handle.write(json.dumps(converted, separators=(",", ":"), sort_keys=True) + "\n")
                converted_count += 1
                scenario_id = str(converted.get("scenario_id", "unknown_scenario"))
                skill_id = str(converted.get("skill_id", "unknown_skill"))
                scenario_counts[scenario_id] = scenario_counts.get(scenario_id, 0) + 1
                skill_counts[skill_id] = skill_counts.get(skill_id, 0) + 1

    if sample_count == 0:
        errors.append("no samples read from input paths")
    if converted_count == 0:
        errors.append("no context BC samples were written; output file was not updated")

    ok = not errors and invalid_output_count == 0 and converted_count > 0
    output_written = False
    digest = None
    if ok:
        tmp_output.replace(output)
        output_written = True
        digest = sha256_file(output) if output.exists() else None
    else:
        tmp_output.unlink(missing_ok=True)
        digest = sha256_file(output) if output.exists() else None

    result = ContextBcExportResult(
        ok=ok,
        output_path=str(output),
        manifest_path=str(manifest),
        input_paths=input_paths,
        sample_count=sample_count,
        converted_count=converted_count,
        skipped_count=skipped_count,
        invalid_output_count=invalid_output_count,
        scenario_counts=scenario_counts,
        skill_counts=skill_counts,
        output_sha256=digest,
        output_written=output_written,
        contract_path=str(contract_path),
        scenario_manifest_path=str(scenario_manifest),
        include_short_history=include_short_history,
        strict_context=strict_context,
        errors=errors[:max_reported_issues],
        warnings=warnings[:max_reported_issues],
    )
    _write_manifest(result, output=output, manifest=manifest, input_digest=input_digest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export legacy/scenario-aware teacher JSONL rows into context-conditioned BC rows.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files, directories, or prepared_manifest.json files")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTEXT_OUTPUT, help="Output context BC JSONL path")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional output manifest path")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH, help="BC training contract path")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST, help="Scenario manifest path")
    parser.add_argument("--no-short-history", action="store_true", help="Do not add bounded short_history fields")
    parser.add_argument("--strict-context", action="store_true", help="Fail rows whose scenario/skill context cannot be resolved from the scenario manifest")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum number of source samples to read")
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    result = export_context_bc_dataset(
        args.inputs,
        output_path=args.output,
        manifest_path=args.manifest,
        contract_path=args.contract,
        scenario_manifest=args.scenario_manifest,
        include_short_history=not args.no_short_history,
        strict_context=args.strict_context,
        max_samples=args.max_samples,
    )
    if args.report is not None:
        _write_markdown(result, args.report)
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Soridormi context BC dataset export")
        print("====================================")
        print(f"Output: {result.output_path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Result: {'OK' if result.ok else 'FAILED'}")
        print(f"Converted: {result.converted_count}/{result.sample_count} samples")
        print(f"Output updated: {'yes' if result.output_written else 'no'}")
        if result.errors:
            print("Errors:")
            for error in result.errors[:20]:
                print(f"  - {error}")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings[:20]:
                print(f"  - {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
