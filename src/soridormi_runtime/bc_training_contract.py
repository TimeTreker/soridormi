from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONTRACT_PATH = Path("configs/training/open_duck_mini_v2_context_bc_contract_v1.json")
CONTRACT_SCHEMA_VERSION = "m9.bc_training_contract.v1"
CONTEXT_SAMPLE_TYPE = "soridormi.policy_supervision.context_v1"
LEGACY_SAMPLE_TYPE = "soridormi.policy_supervision.v1"


@dataclass
class ContractValidationResult:
    ok: bool
    contract_path: str
    contract_id: str | None = None
    robot_profile: str | None = None
    schema_version: str | None = None
    stage_count: int = 0
    input_groups: list[str] = field(default_factory=list)
    action_size: int | None = None
    observation_size: int | None = None
    natural_language_allowed: bool | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SampleValidationResult:
    ok: bool
    sample_count: int
    valid_count: int
    invalid_count: int
    context_sample_count: int = 0
    legacy_sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ContractReport:
    ok: bool
    contract: ContractValidationResult
    sample_validation: SampleValidationResult | None = None


def _load_json(path: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    resolved = Path(path)
    if not resolved.exists():
        return None, [f"JSON file not found: {resolved}"]
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON in {resolved}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"Expected top-level JSON object in {resolved}"]
    return payload, []


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _field_by_name(group: dict[str, Any], name: str) -> dict[str, Any] | None:
    fields = group.get("fields")
    if not isinstance(fields, list):
        return None
    for field_spec in fields:
        if isinstance(field_spec, dict) and field_spec.get("name") == name:
            return field_spec
    return None


def _group_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = contract.get("input_groups", [])
    if not isinstance(groups, list):
        return {}
    return {str(group.get("name")): group for group in groups if isinstance(group, dict) and group.get("name")}


def _vector_errors(value: Any, *, size: int | None, field_name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [f"{field_name} must be a list"]
    if size is not None and len(value) != size:
        errors.append(f"{field_name} size {len(value)} != expected {size}")
    bad = [index for index, item in enumerate(value) if not _is_number(item)]
    if bad:
        preview = ", ".join(str(index) for index in bad[:8])
        errors.append(f"{field_name} contains non-finite/non-numeric values at indices {preview}")
    return errors


def _range_errors(value: Any, *, field_name: str, range_spec: Any) -> list[str]:
    if value is None:
        return []
    if not _is_number(value):
        return [f"{field_name} must be numeric"]
    if isinstance(range_spec, list) and len(range_spec) == 2:
        low, high = float(range_spec[0]), float(range_spec[1])
        v = float(value)
        if v < low or v > high:
            return [f"{field_name}={v} outside range [{low}, {high}]"]
    return []


def validate_contract_payload(contract: dict[str, Any], *, path: str | Path = DEFAULT_CONTRACT_PATH) -> ContractValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CONTRACT_SCHEMA_VERSION}")
    if not contract.get("contract_id"):
        errors.append("contract_id is required")
    if contract.get("natural_language_allowed") is not False:
        errors.append("natural_language_allowed must be false for low-level policy contracts")

    policy_io = contract.get("policy_io")
    if not isinstance(policy_io, dict):
        errors.append("policy_io must be an object")
        policy_io = {}
    action_size = policy_io.get("action_size")
    if action_size != 14:
        errors.append("policy_io.action_size must be 14 for Open Duck Mini v2 action_14d")
    if policy_io.get("output_expression") != "action_14d":
        errors.append("policy_io.output_expression must be action_14d")

    output = contract.get("output")
    if not isinstance(output, dict):
        errors.append("output must be an object")
        output = {}
    if output.get("size") != 14:
        errors.append("output.size must be 14")
    if output.get("field") != "teacher_action":
        errors.append("output.field must be teacher_action")

    groups = _group_map(contract)
    required_groups = ["robot_state", "desired_command", "task_context", "environment_context", "short_history"]
    for name in required_groups:
        if name not in groups:
            errors.append(f"missing input group: {name}")

    observation_size: int | None = None
    robot_state = groups.get("robot_state", {})
    obs_spec = _field_by_name(robot_state, "observation")
    if obs_spec is None:
        errors.append("robot_state.observation field spec is required")
    else:
        observation_size = obs_spec.get("size")
        if observation_size != 101:
            errors.append("robot_state.observation size must be 101")

    desired = groups.get("desired_command", {})
    for command_name in ("vx_mps", "vy_mps", "yaw_radps"):
        spec = _field_by_name(desired, command_name)
        if spec is None:
            errors.append(f"desired_command.{command_name} field spec is required")
        elif not (isinstance(spec.get("range"), list) and len(spec["range"]) == 2):
            errors.append(f"desired_command.{command_name} must declare a numeric range")

    dataset_rows = contract.get("dataset_rows")
    if not isinstance(dataset_rows, dict):
        errors.append("dataset_rows must be an object")
        dataset_rows = {}
    required_context_fields = dataset_rows.get("required_context_fields")
    if not isinstance(required_context_fields, list):
        errors.append("dataset_rows.required_context_fields must be a list")
    else:
        for field in ("scenario_id", "skill_id", "robot_state", "desired_command", "task_context", "environment_context", "teacher_action"):
            if field not in required_context_fields:
                errors.append(f"dataset_rows.required_context_fields missing {field}")

    stages = contract.get("adoption_stages")
    if not isinstance(stages, list) or not stages:
        errors.append("adoption_stages must be a non-empty list")
        stages = []
    seen_stage_numbers: set[int] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            errors.append("adoption_stages entries must be objects")
            continue
        try:
            stage_num = int(stage.get("stage"))
        except (TypeError, ValueError):
            errors.append("adoption stage must include integer stage")
            continue
        if stage_num in seen_stage_numbers:
            errors.append(f"duplicate adoption stage {stage_num}")
        seen_stage_numbers.add(stage_num)
        req = stage.get("required_input_groups")
        if not isinstance(req, list) or not req:
            errors.append(f"adoption stage {stage_num} requires required_input_groups")
        else:
            for group_name in req:
                if group_name not in groups:
                    errors.append(f"adoption stage {stage_num} references missing group {group_name}")

    example = contract.get("example_row")
    if isinstance(example, dict):
        sample_errors, sample_warnings, _kind = validate_sample_against_contract(example, contract, allow_legacy=True)
        errors.extend(f"example_row: {error}" for error in sample_errors)
        warnings.extend(f"example_row: {warning}" for warning in sample_warnings)
    else:
        warnings.append("example_row is missing")

    return ContractValidationResult(
        ok=not errors,
        contract_path=str(path),
        contract_id=str(contract.get("contract_id")) if contract.get("contract_id") is not None else None,
        robot_profile=str(contract.get("robot_profile")) if contract.get("robot_profile") is not None else None,
        schema_version=str(contract.get("schema_version")) if contract.get("schema_version") is not None else None,
        stage_count=len(stages),
        input_groups=list(groups.keys()),
        action_size=int(action_size) if isinstance(action_size, int) else None,
        observation_size=int(observation_size) if isinstance(observation_size, int) else None,
        natural_language_allowed=contract.get("natural_language_allowed") if isinstance(contract.get("natural_language_allowed"), bool) else None,
        errors=errors,
        warnings=warnings,
    )


def load_and_validate_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> tuple[dict[str, Any] | None, ContractValidationResult]:
    payload, errors = _load_json(path)
    if payload is None:
        return None, ContractValidationResult(ok=False, contract_path=str(path), errors=errors)
    return payload, validate_contract_payload(payload, path=path)


def _context_observation(sample: dict[str, Any]) -> Any:
    robot_state = sample.get("robot_state")
    if isinstance(robot_state, dict) and "observation" in robot_state:
        return robot_state.get("observation")
    return sample.get("observation")


def _context_action(sample: dict[str, Any]) -> Any:
    if "teacher_action" in sample:
        return sample.get("teacher_action")
    return sample.get("action")


def _sample_type(sample: dict[str, Any]) -> str | None:
    value = sample.get("sample_type")
    return str(value) if value is not None else None


def validate_sample_against_contract(
    sample: dict[str, Any],
    contract: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> tuple[list[str], list[str], str]:
    errors: list[str] = []
    warnings: list[str] = []
    groups = _group_map(contract)
    output = contract.get("output") if isinstance(contract.get("output"), dict) else {}
    sample_type = _sample_type(sample)

    if sample_type == CONTEXT_SAMPLE_TYPE:
        kind = "context"
    elif sample_type == LEGACY_SAMPLE_TYPE and allow_legacy:
        kind = "legacy"
        warnings.append("legacy soridormi.policy_supervision.v1 sample accepted for Stage 1 only")
    else:
        kind = "unknown"
        errors.append(f"sample_type must be {CONTEXT_SAMPLE_TYPE}" + (f" or {LEGACY_SAMPLE_TYPE}" if allow_legacy else ""))

    obs_size = None
    obs_spec = _field_by_name(groups.get("robot_state", {}), "observation")
    if obs_spec is not None and isinstance(obs_spec.get("size"), int):
        obs_size = int(obs_spec["size"])
    errors.extend(_vector_errors(_context_observation(sample), size=obs_size, field_name="robot_state.observation"))

    action_size = output.get("size") if isinstance(output.get("size"), int) else None
    errors.extend(_vector_errors(_context_action(sample), size=action_size, field_name="teacher_action"))

    if kind == "context":
        dataset_rows = contract.get("dataset_rows") if isinstance(contract.get("dataset_rows"), dict) else {}
        required = dataset_rows.get("required_context_fields") if isinstance(dataset_rows.get("required_context_fields"), list) else []
        for field in required:
            if field not in sample:
                errors.append(f"missing required context field {field}")

        desired = sample.get("desired_command")
        if not isinstance(desired, dict):
            errors.append("desired_command must be an object")
        else:
            desired_spec = groups.get("desired_command", {})
            for field_name in ("vx_mps", "vy_mps", "yaw_radps"):
                spec = _field_by_name(desired_spec, field_name) or {}
                if field_name not in desired:
                    errors.append(f"desired_command.{field_name} is required")
                else:
                    errors.extend(_range_errors(desired.get(field_name), field_name=f"desired_command.{field_name}", range_spec=spec.get("range")))

        task_context = sample.get("task_context")
        if not isinstance(task_context, dict):
            errors.append("task_context must be an object")
        else:
            skill_id = task_context.get("skill_id", sample.get("skill_id"))
            if not isinstance(skill_id, str) or not skill_id:
                errors.append("task_context.skill_id must be a non-empty string")

        environment_context = sample.get("environment_context")
        if not isinstance(environment_context, dict):
            errors.append("environment_context must be an object")
        else:
            terrain = environment_context.get("terrain_type")
            if not isinstance(terrain, str) or not terrain:
                errors.append("environment_context.terrain_type must be a non-empty string")

        if sample.get("skill_id") != (task_context.get("skill_id") if isinstance(task_context, dict) else None):
            warnings.append("top-level skill_id differs from task_context.skill_id")

    return errors, warnings, kind


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
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


def validate_sample_jsonl(
    sample_jsonl: str | Path,
    contract: dict[str, Any],
    *,
    allow_legacy: bool = False,
    max_reported_issues: int = 50,
) -> SampleValidationResult:
    path = Path(sample_jsonl)
    if not path.exists():
        return SampleValidationResult(ok=False, sample_count=0, valid_count=0, invalid_count=0, errors=[f"Sample JSONL not found: {path}"])

    sample_count = 0
    valid = 0
    invalid = 0
    context_count = 0
    legacy_count = 0
    errors: list[str] = []
    warnings: list[str] = []

    for line_number, sample, parse_error in iter_jsonl(path):
        sample_count += 1
        if parse_error is not None or sample is None:
            invalid += 1
            if len(errors) < max_reported_issues:
                errors.append(parse_error or f"line {line_number}: invalid sample")
            continue
        sample_errors, sample_warnings, kind = validate_sample_against_contract(sample, contract, allow_legacy=allow_legacy)
        if kind == "context":
            context_count += 1
        elif kind == "legacy":
            legacy_count += 1
        if sample_errors:
            invalid += 1
            for error in sample_errors:
                if len(errors) < max_reported_issues:
                    errors.append(f"line {line_number}: {error}")
            continue
        valid += 1
        for warning in sample_warnings:
            if len(warnings) < max_reported_issues:
                warnings.append(f"line {line_number}: {warning}")

    if sample_count == 0:
        errors.append("Sample JSONL is empty")
    return SampleValidationResult(
        ok=not errors,
        sample_count=sample_count,
        valid_count=valid,
        invalid_count=invalid,
        context_sample_count=context_count,
        legacy_sample_count=legacy_count,
        errors=errors,
        warnings=warnings,
    )


def _write_markdown(report: ContractReport, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = report.contract
    lines = [
        "# Soridormi BC Training Contract Report",
        "",
        f"Result: **{'PASS' if report.ok else 'FAIL'}**",
        "",
        f"Contract: `{contract.contract_path}`",
        f"Contract id: `{contract.contract_id}`",
        f"Schema: `{contract.schema_version}`",
        f"Robot profile: `{contract.robot_profile}`",
        f"Natural language allowed: `{contract.natural_language_allowed}`",
        f"Input groups: {', '.join(contract.input_groups)}",
        f"Observation size: `{contract.observation_size}`",
        f"Action size: `{contract.action_size}`",
        "",
        "## Contract validation",
        "",
    ]
    if contract.errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in contract.errors)
    else:
        lines.append("No contract errors.")
    if contract.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in contract.warnings)
    if report.sample_validation is not None:
        sample = report.sample_validation
        lines.extend([
            "",
            "## Sample JSONL validation",
            "",
            f"Samples: `{sample.sample_count}`",
            f"Valid: `{sample.valid_count}`",
            f"Invalid: `{sample.invalid_count}`",
            f"Context samples: `{sample.context_sample_count}`",
            f"Legacy samples: `{sample.legacy_sample_count}`",
        ])
        if sample.errors:
            lines.append("")
            lines.append("Sample errors:")
            lines.extend(f"- {item}" for item in sample.errors)
        if sample.warnings:
            lines.append("")
            lines.append("Sample warnings:")
            lines.extend(f"- {item}" for item in sample.warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_report(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    sample_jsonl: str | Path | None = None,
    allow_legacy: bool = False,
) -> ContractReport:
    contract, contract_result = load_and_validate_contract(contract_path)
    sample_result = None
    if contract is not None and sample_jsonl is not None:
        sample_result = validate_sample_jsonl(sample_jsonl, contract, allow_legacy=allow_legacy)
    ok = contract_result.ok and (sample_result.ok if sample_result is not None else True)
    return ContractReport(ok=ok, contract=contract_result, sample_validation=sample_result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Soridormi context-conditioned BC training contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH, help="BC training contract JSON path")
    parser.add_argument("--sample-jsonl", type=Path, default=None, help="Optional dataset JSONL to validate against the contract")
    parser.add_argument("--allow-legacy", action="store_true", help="Allow legacy soridormi.policy_supervision.v1 samples for Stage 1 preflight")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report output path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    args = parser.parse_args(argv)

    report = build_report(contract_path=args.contract, sample_jsonl=args.sample_jsonl, allow_legacy=args.allow_legacy)
    if args.output is not None:
        _write_markdown(report, args.output)
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Soridormi BC training contract")
        print("================================")
        print(f"Contract: {report.contract.contract_path}")
        print(f"Result: {'OK' if report.ok else 'FAILED'}")
        if report.contract.errors:
            print("Errors:")
            for error in report.contract.errors:
                print(f"  - {error}")
        if report.contract.warnings:
            print("Warnings:")
            for warning in report.contract.warnings:
                print(f"  - {warning}")
        if report.sample_validation is not None:
            sample = report.sample_validation
            print(f"Samples: {sample.valid_count}/{sample.sample_count} valid")
            if sample.errors:
                print("Sample errors:")
                for error in sample.errors[:20]:
                    print(f"  - {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
