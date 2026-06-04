from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.bc_training_contract import DEFAULT_CONTRACT_PATH, load_and_validate_contract, validate_sample_against_contract
from soridormi_runtime.context_bc_dataset_prepare import PREPARED_CONTEXT_DATASET_TYPE
from soridormi_runtime.scenario_curriculum import COLLECTOR_READY_STATUSES, DEFAULT_SCENARIO_MANIFEST, list_scenarios
from soridormi_runtime.training_dataset import sha256_file

PREPARED_CONTEXT_GATE_SCHEMA_VERSION = "m9.context_bc_prepared_gate.v1"
DEFAULT_PREPARED_GATE_OUTPUT_DIR = Path("artifacts/training/context_bc/prepared_gate")
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitGateSummary:
    name: str
    path: str
    exists: bool
    sample_count: int
    manifest_sample_count: int | None
    group_count: int
    manifest_group_count: int | None
    scenario_counts: dict[str, int] = field(default_factory=dict)
    sha256: str | None = None
    manifest_sha256: str | None = None
    invalid_sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedContextGateResult:
    ok: bool
    schema_version: str
    prepared_manifest_path: str
    prepared_dataset_type: str | None
    manifest_ok: bool | None
    output_dir: str | None
    contract_path: str
    scenario_manifest_path: str
    split_group_field: str
    require_no_group_leakage: bool
    required_scenarios: list[str]
    min_samples_per_required_scenario: int
    min_split_samples: dict[str, int]
    total_sample_count: int
    scenario_counts: dict[str, int]
    split_group_counts: dict[str, int]
    splits: dict[str, SplitGateSummary]
    leaked_groups: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _manifest_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        return path / "prepared_manifest.json"
    return path


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


def _scenario_id(sample: dict[str, Any]) -> str:
    value = sample.get("scenario_id")
    text = str(value).strip() if value is not None else ""
    return text or "unknown_scenario"


def _group_key(sample: dict[str, Any], index: int, field_name: str) -> str:
    if field_name in {"rollout", "rollout_id"}:
        value = sample.get("rollout_id")
        if value is not None and str(value):
            return str(value)
        scenario = _scenario_id(sample)
        episode = sample.get("episode_index", sample.get("episode_id"))
        if episode is not None and str(episode):
            return f"{scenario}:episode:{episode}"
        source = sample.get("source_dataset", sample.get("source_log"))
        if source is not None and str(source):
            return f"{scenario}:source:{source}"
    else:
        value = sample.get(field_name)
        if value is not None and str(value):
            return str(value)
    return f"sample:{index}"


def _scenario_counts(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        scenario = _scenario_id(sample)
        counts[scenario] = counts.get(scenario, 0) + 1
    return dict(sorted(counts.items()))


def _merge_counts(items: Iterable[dict[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            counts[key] = counts.get(key, 0) + int(value)
    return dict(sorted(counts.items()))


def _split_info_from_manifest(manifest: dict[str, Any], split_name: str) -> dict[str, Any]:
    splits = manifest.get("splits")
    if isinstance(splits, dict) and isinstance(splits.get(split_name), dict):
        return dict(splits[split_name])
    raw = manifest.get(split_name)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _required_ready_locomotion_scenarios(path: str | Path) -> list[str]:
    required: list[str] = []
    for scenario in list_scenarios(path, include_planned=False):
        if scenario.status not in COLLECTOR_READY_STATUSES:
            continue
        if scenario.family.startswith("locomotion"):
            required.append(scenario.id)
    return required


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        for piece in str(raw).split(","):
            item = piece.strip()
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _read_split(
    split_name: str,
    split_payload: dict[str, Any],
    *,
    contract: dict[str, Any],
    group_field: str,
    max_reported_issues: int,
) -> tuple[SplitGateSummary, set[str]]:
    raw_path = split_payload.get("path", f"{split_name}.jsonl")
    path = Path(str(raw_path))
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []
    groups: set[str] = set()
    invalid_count = 0

    if not path.exists():
        errors.append(f"split {split_name!r} JSONL not found: {path}")
        return (
            SplitGateSummary(
                name=split_name,
                path=str(path),
                exists=False,
                sample_count=0,
                manifest_sample_count=_int_or_none(split_payload.get("sample_count")),
                group_count=0,
                manifest_group_count=_int_or_none(split_payload.get("group_count")),
                manifest_sha256=_str_or_none(split_payload.get("sha256")),
                invalid_sample_count=0,
                errors=errors,
                warnings=warnings,
            ),
            groups,
        )

    for index, (line_number, sample, parse_error) in enumerate(_iter_jsonl(path)):
        prefix = f"{path}:{line_number}"
        if parse_error is not None or sample is None:
            invalid_count += 1
            if len(errors) < max_reported_issues:
                errors.append(f"{prefix}: {parse_error or 'invalid sample'}")
            continue
        sample_errors, sample_warnings, kind = validate_sample_against_contract(sample, contract, allow_legacy=False)
        if kind != "context":
            sample_errors.append("sample_type must be soridormi.policy_supervision.context_v1")
        if sample_errors:
            invalid_count += 1
            for item in sample_errors:
                if len(errors) < max_reported_issues:
                    errors.append(f"{prefix}: {item}")
            continue
        for item in sample_warnings:
            if len(warnings) < max_reported_issues:
                warnings.append(f"{prefix}: {item}")
        samples.append(sample)
        groups.add(_group_key(sample, index, group_field))

    actual_sha = sha256_file(path)
    actual_counts = _scenario_counts(samples)
    manifest_counts = split_payload.get("scenario_counts") if isinstance(split_payload.get("scenario_counts"), dict) else {}
    manifest_sample_count = _int_or_none(split_payload.get("sample_count"))
    manifest_group_count = _int_or_none(split_payload.get("group_count"))
    manifest_sha = _str_or_none(split_payload.get("sha256"))

    if manifest_sample_count is not None and manifest_sample_count != len(samples):
        errors.append(f"split {split_name!r} manifest sample_count={manifest_sample_count} but JSONL has {len(samples)} valid sample(s)")
    if manifest_group_count is not None and manifest_group_count != len(groups):
        errors.append(f"split {split_name!r} manifest group_count={manifest_group_count} but JSONL has {len(groups)} group(s)")
    if manifest_sha is not None and manifest_sha != actual_sha:
        errors.append(f"split {split_name!r} manifest sha256 does not match current JSONL")
    if manifest_counts and dict(sorted((str(k), int(v)) for k, v in manifest_counts.items())) != actual_counts:
        errors.append(f"split {split_name!r} manifest scenario_counts do not match current JSONL")

    return (
        SplitGateSummary(
            name=split_name,
            path=str(path),
            exists=True,
            sample_count=len(samples),
            manifest_sample_count=manifest_sample_count,
            group_count=len(groups),
            manifest_group_count=manifest_group_count,
            scenario_counts=actual_counts,
            sha256=actual_sha,
            manifest_sha256=manifest_sha,
            invalid_sample_count=invalid_count,
            errors=errors,
            warnings=warnings,
        ),
        groups,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_prepared_context_dataset(
    prepared_manifest_or_dir: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    scenario_manifest_path: str | Path = DEFAULT_SCENARIO_MANIFEST,
    require_scenarios: Iterable[str] = (),
    require_ready_locomotion: bool = False,
    min_samples_per_required_scenario: int = 1,
    min_train_samples: int = 1,
    min_val_samples: int = 1,
    min_test_samples: int = 1,
    require_manifest_ok: bool = True,
    require_no_group_leakage: bool = True,
    max_reported_issues: int = 80,
) -> PreparedContextGateResult:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = _manifest_path(prepared_manifest_or_dir)
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append(f"prepared manifest not found: {manifest_path}")
    else:
        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"failed to read prepared manifest {manifest_path}: {exc}")

    contract, contract_result = load_and_validate_contract(contract_path)
    if contract is None:
        contract = {}
    if not contract_result.ok:
        errors.extend(f"contract: {item}" for item in contract_result.errors)

    dataset_type = _str_or_none(manifest.get("dataset_type")) if manifest else None
    manifest_ok = manifest.get("ok") if manifest else None
    if dataset_type != PREPARED_CONTEXT_DATASET_TYPE:
        errors.append(f"prepared manifest dataset_type must be {PREPARED_CONTEXT_DATASET_TYPE}, got {dataset_type!r}")
    if require_manifest_ok and manifest_ok is not True:
        errors.append("prepared manifest ok is not true; rerun prepare after fixing upstream dataset errors")

    group_field = str(manifest.get("split_group_field", "rollout_id")) if manifest else "rollout_id"
    required = _dedupe(require_scenarios)
    if require_ready_locomotion:
        required = _dedupe([*required, *_required_ready_locomotion_scenarios(scenario_manifest_path)])

    splits: dict[str, SplitGateSummary] = {}
    split_groups: dict[str, set[str]] = {}
    if manifest and contract_result.ok:
        for split_name in SPLIT_NAMES:
            summary, groups = _read_split(
                split_name,
                _split_info_from_manifest(manifest, split_name),
                contract=contract,
                group_field=group_field,
                max_reported_issues=max_reported_issues,
            )
            splits[split_name] = summary
            split_groups[split_name] = groups
            errors.extend(summary.errors)
            warnings.extend(summary.warnings)
    else:
        for split_name in SPLIT_NAMES:
            splits[split_name] = SplitGateSummary(
                name=split_name,
                path=str(Path(str(prepared_manifest_or_dir)) / f"{split_name}.jsonl"),
                exists=False,
                sample_count=0,
                manifest_sample_count=None,
                group_count=0,
                manifest_group_count=None,
            )
            split_groups[split_name] = set()

    min_split_samples = {"train": min_train_samples, "val": min_val_samples, "test": min_test_samples}
    for split_name, minimum in min_split_samples.items():
        if minimum > 0 and splits[split_name].sample_count < minimum:
            errors.append(
                f"split {split_name!r} has {splits[split_name].sample_count} sample(s); minimum required is {minimum}"
            )

    scenario_counts = _merge_counts(summary.scenario_counts for summary in splits.values())
    total_sample_count = sum(summary.sample_count for summary in splits.values())
    for scenario_id in required:
        count = scenario_counts.get(scenario_id, 0)
        if count < min_samples_per_required_scenario:
            errors.append(
                f"required scenario {scenario_id!r} has {count} sample(s); "
                f"minimum required is {min_samples_per_required_scenario}"
            )

    leaked_groups: dict[str, list[str]] = {}
    if require_no_group_leakage:
        owner: dict[str, str] = {}
        for split_name, groups in split_groups.items():
            for group in groups:
                if group in owner:
                    leaked_groups.setdefault(group, [owner[group]]).append(split_name)
                else:
                    owner[group] = split_name
        for group, split_names in sorted(leaked_groups.items()):
            errors.append(f"split group {group!r} appears in multiple splits: {', '.join(split_names)}")

    split_group_counts = {name: summary.group_count for name, summary in splits.items()}
    result = PreparedContextGateResult(
        ok=not errors,
        schema_version=PREPARED_CONTEXT_GATE_SCHEMA_VERSION,
        prepared_manifest_path=str(manifest_path),
        prepared_dataset_type=dataset_type,
        manifest_ok=bool(manifest_ok) if manifest_ok is not None else None,
        output_dir=_str_or_none(manifest.get("output_dir")) if manifest else None,
        contract_path=str(contract_path),
        scenario_manifest_path=str(scenario_manifest_path),
        split_group_field=group_field,
        require_no_group_leakage=require_no_group_leakage,
        required_scenarios=required,
        min_samples_per_required_scenario=min_samples_per_required_scenario,
        min_split_samples=min_split_samples,
        total_sample_count=total_sample_count,
        scenario_counts=scenario_counts,
        split_group_counts=split_group_counts,
        splits=splits,
        leaked_groups=leaked_groups,
        errors=errors,
        warnings=warnings,
    )
    return result


def write_markdown_report(result: PreparedContextGateResult, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Soridormi Prepared Context BC Dataset Gate",
        "",
        f"Result: **{'PASS' if result.ok else 'FAIL'}**",
        "",
        f"Manifest: `{result.prepared_manifest_path}`",
        f"Dataset type: `{result.prepared_dataset_type}`",
        f"Manifest ok: `{result.manifest_ok}`",
        f"Total samples: `{result.total_sample_count}`",
        f"Split group field: `{result.split_group_field}`",
        f"Require no group leakage: `{result.require_no_group_leakage}`",
        "",
        "## Splits",
        "",
        "| Split | Samples | Groups | Exists | SHA256 |",
        "|---|---:|---:|---|---|",
    ]
    for name in SPLIT_NAMES:
        split = result.splits[name]
        lines.append(
            f"| {name} | {split.sample_count} | {split.group_count} | {split.exists} | `{split.sha256}` |"
        )
    if result.scenario_counts:
        lines.extend(["", "## Scenario counts", ""])
        lines.extend(f"- `{scenario}`: {count}" for scenario, count in sorted(result.scenario_counts.items()))
    if result.required_scenarios:
        lines.extend(["", "## Required scenarios", ""])
        lines.extend(f"- `{scenario}`" for scenario in result.required_scenarios)
    if result.leaked_groups:
        lines.extend(["", "## Leaked split groups", ""])
        lines.extend(f"- `{group}`: {', '.join(splits)}" for group, splits in sorted(result.leaked_groups.items()))
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in result.errors)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in result.warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(result: PreparedContextGateResult) -> None:
    print("Soridormi prepared context BC dataset gate")
    print("===========================================")
    print(f"Manifest: {result.prepared_manifest_path}")
    print(f"Samples: {result.total_sample_count}")
    print("Splits:")
    for name in SPLIT_NAMES:
        split = result.splits[name]
        print(f"  {name}: {split.sample_count} samples, {split.group_count} groups")
    if result.errors:
        print("Errors:")
        for item in result.errors[:40]:
            print(f"  - {item}")
    if result.warnings:
        print("Warnings:")
        for item in result.warnings[:40]:
            print(f"  - {item}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate a prepared context BC dataset manifest before BC training.")
    parser.add_argument("prepared", type=Path, help="prepared_manifest.json or a prepared dataset directory")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--require-scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate")
    parser.add_argument("--require-ready-locomotion", action="store_true", help="Require every registry-ready locomotion scenario")
    parser.add_argument("--min-samples-per-required-scenario", type=int, default=1)
    parser.add_argument("--min-train-samples", type=int, default=1)
    parser.add_argument("--min-val-samples", type=int, default=1)
    parser.add_argument("--min-test-samples", type=int, default=1)
    parser.add_argument("--allow-empty-val", action="store_true", help="Set validation split minimum to 0")
    parser.add_argument("--allow-empty-test", action="store_true", help="Set test split minimum to 0")
    parser.add_argument("--allow-manifest-failed", action="store_true", help="Do not fail only because prepared manifest ok=false")
    parser.add_argument("--allow-group-leakage", action="store_true", help="Do not fail when rollout groups appear in multiple splits")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write prepared_context_gate_report.json/md here")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    min_val = 0 if args.allow_empty_val else args.min_val_samples
    min_test = 0 if args.allow_empty_test else args.min_test_samples
    result = validate_prepared_context_dataset(
        args.prepared,
        contract_path=args.contract,
        scenario_manifest_path=args.scenario_manifest,
        require_scenarios=args.require_scenario,
        require_ready_locomotion=args.require_ready_locomotion,
        min_samples_per_required_scenario=args.min_samples_per_required_scenario,
        min_train_samples=args.min_train_samples,
        min_val_samples=min_val,
        min_test_samples=min_test,
        require_manifest_ok=not args.allow_manifest_failed,
        require_no_group_leakage=not args.allow_group_leakage,
    )
    output_path = args.output
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "prepared_context_gate_report.json"
        json_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output_path is None:
            output_path = args.output_dir / "prepared_context_gate_report.md"
    if output_path is not None:
        write_markdown_report(result, output_path)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
