from __future__ import annotations

import argparse
import hashlib
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
from soridormi_runtime.training_dataset import sha256_file

CONTEXT_BC_PREPARE_SCHEMA_VERSION = 1
PREPARED_CONTEXT_DATASET_TYPE = "soridormi.policy_supervision.context_prepared.v1"
DEFAULT_CONTEXT_PREPARED_DIR = Path("/data/training_datasets/context_bc/prepared")
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


@dataclass
class ContextBcSplitInfo:
    name: str
    path: str
    sample_count: int
    group_count: int
    scenario_counts: dict[str, int] = field(default_factory=dict)
    sha256: str | None = None


@dataclass
class ContextBcPrepareResult:
    ok: bool
    output_dir: str
    manifest_path: str
    input_paths: list[str]
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    skipped_invalid_count: int
    train: ContextBcSplitInfo
    val: ContextBcSplitInfo
    test: ContextBcSplitInfo
    seed: int
    shuffle: bool
    ratios: dict[str, float]
    split_group_field: str
    stratify_by_scenario: bool
    split_group_counts: dict[str, int] = field(default_factory=dict)
    scenario_counts: dict[str, int] = field(default_factory=dict)
    input_sha256: dict[str, str | None] = field(default_factory=dict)
    contract_path: str = str(DEFAULT_CONTRACT_PATH)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> list[str]:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios):
        return ["Split ratios must be finite non-negative numbers"]
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        return [f"Split ratios must sum to 1.0, got {total:g}"]
    if train_ratio <= 0:
        return ["train_ratio must be > 0"]
    return []


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if train_count == 0:
        train_count = 1
    if total >= 3 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def _scenario_id(sample: dict[str, Any]) -> str:
    value = sample.get("scenario_id")
    text = str(value).strip() if value is not None else ""
    return text or "unknown_scenario"


def _sample_sort_key(sample: dict[str, Any], index: int) -> str:
    source = sample.get("source_dataset", sample.get("source_log", ""))
    rollout = sample.get("rollout_id", "")
    step = sample.get("step_index", sample.get("timestep", index))
    return f"{source}|{rollout}|{step}|{index}"


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
    elif field_name in {"scenario", "scenario_id"}:
        return _scenario_id(sample)
    elif field_name in {"episode", "episode_id"}:
        scenario = _scenario_id(sample)
        episode = sample.get("episode_index", sample.get("episode_id"))
        if episode is not None and str(episode):
            return f"{scenario}:episode:{episode}"
    elif field_name in {"source", "source_dataset"}:
        value = sample.get("source_dataset", sample.get("source_log"))
        if value is not None and str(value):
            return str(value)
    else:
        value = sample.get(field_name)
        if value is not None and str(value):
            return str(value)
    return f"sample:{_sample_sort_key(sample, index)}"


def _ordered_group_keys(
    groups: dict[str, list[tuple[int, dict[str, Any]]]],
    first_index: dict[str, int],
    *,
    seed: int,
    shuffle: bool,
    salt: str = "",
) -> list[str]:
    keys = list(groups)
    if shuffle:
        keys.sort(key=lambda key: hashlib.sha256(f"{seed}|context_group|{salt}|{key}".encode("utf-8")).hexdigest())
    else:
        keys.sort(key=lambda key: first_index[key])
    return keys


def _assign_group_keys(
    groups: dict[str, list[tuple[int, dict[str, Any]]]],
    first_index: dict[str, int],
    *,
    seed: int,
    shuffle: bool,
    train_ratio: float,
    val_ratio: float,
    stratify_by_scenario: bool,
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    assignments: dict[str, str] = {}

    def assign(keys: list[str]) -> None:
        train_count, val_count, test_count = _split_counts(len(keys), train_ratio, val_ratio)
        train_keys = set(keys[:train_count])
        val_keys = set(keys[train_count : train_count + val_count])
        test_keys = set(keys[train_count + val_count : train_count + val_count + test_count])
        for key in train_keys:
            assignments[key] = "train"
        for key in val_keys:
            assignments[key] = "val"
        for key in test_keys:
            assignments[key] = "test"

    if stratify_by_scenario:
        by_scenario: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
        first_by_scenario: dict[str, dict[str, int]] = {}
        for key, rows in groups.items():
            scenario = _scenario_id(rows[0][1]) if rows else "unknown_scenario"
            by_scenario.setdefault(scenario, {})[key] = rows
            first_by_scenario.setdefault(scenario, {})[key] = first_index[key]
        for scenario, scenario_groups in sorted(by_scenario.items()):
            keys = _ordered_group_keys(
                scenario_groups,
                first_by_scenario[scenario],
                seed=seed,
                shuffle=shuffle,
                salt=scenario,
            )
            if len(keys) < 3 and (val_ratio > 0 or (1.0 - train_ratio - val_ratio) > 0):
                warnings.append(
                    f"Scenario {scenario!r} has only {len(keys)} split group(s); "
                    "validation/test splits may be empty for this scenario. Collect more independent rollouts."
                )
            assign(keys)
    else:
        keys = _ordered_group_keys(groups, first_index, seed=seed, shuffle=shuffle)
        if len(keys) < 3 and (val_ratio > 0 or (1.0 - train_ratio - val_ratio) > 0):
            warnings.append(
                f"Only {len(keys)} split group(s) found; validation/test splits may be empty. "
                "Collect more independent rollouts/scenarios."
            )
        assign(keys)
    return assignments, warnings


def _collect_split_samples(
    groups: dict[str, list[tuple[int, dict[str, Any]]]],
    assignments: dict[str, str],
    *,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for key, samples in groups.items():
        if assignments.get(key) == split:
            rows.extend(samples)
    rows.sort(key=lambda pair: pair[0])
    return [sample for _index, sample in rows]


def _scenario_counts(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        scenario = _scenario_id(sample)
        counts[scenario] = counts.get(scenario, 0) + 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")


def _split_info(name: str, path: Path, samples: list[dict[str, Any]], group_count: int) -> ContextBcSplitInfo:
    return ContextBcSplitInfo(
        name=name,
        path=str(path),
        sample_count=len(samples),
        group_count=group_count,
        scenario_counts=_scenario_counts(samples),
        sha256=sha256_file(path) if path.exists() else None,
    )


def _empty_split_info(name: str, output_dir: Path) -> ContextBcSplitInfo:
    return ContextBcSplitInfo(name=name, path=str(output_dir / f"{name}.jsonl"), sample_count=0, group_count=0)


def _write_manifest(result: ContextBcPrepareResult) -> None:
    manifest_path = Path(result.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload.update(
        {
            "schema_version": CONTEXT_BC_PREPARE_SCHEMA_VERSION,
            "dataset_type": PREPARED_CONTEXT_DATASET_TYPE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "splits": {
                "train": asdict(result.train),
                "val": asdict(result.val),
                "test": asdict(result.test),
            },
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(result: ContextBcPrepareResult, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Soridormi Context BC Dataset Prepare Report",
        "",
        f"Result: **{'PASS' if result.ok else 'FAIL'}**",
        "",
        f"Output dir: `{result.output_dir}`",
        f"Manifest: `{result.manifest_path}`",
        f"Samples read: `{result.sample_count}`",
        f"Valid samples: `{result.valid_sample_count}`",
        f"Invalid samples: `{result.invalid_sample_count}`",
        f"Skipped invalid samples: `{result.skipped_invalid_count}`",
        f"Split group field: `{result.split_group_field}`",
        f"Scenario-stratified: `{result.stratify_by_scenario}`",
        f"Seed: `{result.seed}`",
        f"Shuffle: `{result.shuffle}`",
        "",
        "## Splits",
        "",
    ]
    for split in (result.train, result.val, result.test):
        lines.extend(
            [
                f"### {split.name}",
                "",
                f"Path: `{split.path}`",
                f"Samples: `{split.sample_count}`",
                f"Groups: `{split.group_count}`",
                f"SHA256: `{split.sha256}`",
                "",
            ]
        )
        if split.scenario_counts:
            lines.append("Scenario counts:")
            lines.extend(f"- `{scenario}`: {count}" for scenario, count in sorted(split.scenario_counts.items()))
            lines.append("")
    if result.errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item}" for item in result.errors)
        lines.append("")
    if result.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in result.warnings)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def prepare_context_bc_dataset(
    inputs: Iterable[str | Path],
    *,
    output_dir: str | Path = DEFAULT_CONTEXT_PREPARED_DIR,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 0,
    shuffle: bool = True,
    split_group_field: str = "rollout_id",
    stratify_by_scenario: bool = True,
    skip_invalid: bool = False,
    max_samples: int | None = None,
    max_reported_issues: int = 80,
) -> ContextBcPrepareResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "prepared_manifest.json"
    empty = {
        "train": _empty_split_info("train", output),
        "val": _empty_split_info("val", output),
        "test": _empty_split_info("test", output),
    }

    input_specs, input_errors = _resolve_input_paths(inputs)
    contract, contract_result = load_and_validate_contract(contract_path)
    errors: list[str] = list(input_errors)
    warnings: list[str] = []
    if not contract_result.ok:
        errors.extend(f"contract: {item}" for item in contract_result.errors)
    if contract is None:
        contract = {}
    ratio_errors = _validate_ratios(train_ratio, val_ratio, test_ratio)
    errors.extend(ratio_errors)

    input_paths = [str(path) for _label, path in input_specs]
    input_sha256: dict[str, str | None] = {}
    for path in (path for _label, path in input_specs):
        input_sha256[str(path)] = sha256_file(path) if path.exists() else None
        if not path.exists():
            errors.append(f"input JSONL not found: {path}")

    sample_count = 0
    valid_samples: list[dict[str, Any]] = []
    invalid_count = 0
    skipped_invalid_count = 0

    if not ratio_errors and contract_result.ok:
        for label, path in input_specs:
            if max_samples is not None and sample_count >= max_samples:
                break
            if not path.exists():
                continue
            for line_number, sample, parse_error in _iter_jsonl(path):
                if max_samples is not None and sample_count >= max_samples:
                    break
                sample_count += 1
                prefix = f"{path}:{line_number}"
                if parse_error is not None or sample is None:
                    invalid_count += 1
                    message = f"{prefix}: {parse_error or 'invalid sample'}"
                    target = warnings if skip_invalid else errors
                    if len(target) < max_reported_issues:
                        target.append(message)
                    if skip_invalid:
                        skipped_invalid_count += 1
                    continue
                sample_errors, sample_warnings, kind = validate_sample_against_contract(sample, contract, allow_legacy=False)
                if kind != "context":
                    sample_errors.append(f"sample_type must be {CONTEXT_SAMPLE_TYPE}")
                if sample_errors:
                    invalid_count += 1
                    target = warnings if skip_invalid else errors
                    for item in sample_errors:
                        if len(target) < max_reported_issues:
                            target.append(f"{prefix}: {item}")
                    if skip_invalid:
                        skipped_invalid_count += 1
                    continue
                for item in sample_warnings:
                    if len(warnings) < max_reported_issues:
                        warnings.append(f"{prefix}: {item}")
                valid_samples.append(sample)

    if not input_specs:
        errors.append("no input JSONL files resolved")
    if sample_count == 0 and not any("input JSONL not found" in error for error in errors):
        errors.append("no samples read from input paths")
    if not valid_samples:
        errors.append("no valid context BC samples found")
    if invalid_count and not skip_invalid:
        warnings.append("invalid rows were skipped from split outputs; fix the dataset or pass --skip-invalid intentionally")
    elif invalid_count and skip_invalid:
        warnings.append(f"skipped {invalid_count} invalid row(s) because --skip-invalid was set")

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    first_index: dict[str, int] = {}
    for index, sample in enumerate(valid_samples):
        key = _group_key(sample, index, split_group_field)
        groups.setdefault(key, []).append((index, sample))
        first_index.setdefault(key, index)

    split_group_counts = {"train": 0, "val": 0, "test": 0}
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    if valid_samples and not ratio_errors:
        assignments, split_warnings = _assign_group_keys(
            groups,
            first_index,
            seed=seed,
            shuffle=shuffle,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify_by_scenario=stratify_by_scenario,
        )
        warnings.extend(split_warnings)
        split_group_counts = {
            "train": sum(1 for split in assignments.values() if split == "train"),
            "val": sum(1 for split in assignments.values() if split == "val"),
            "test": sum(1 for split in assignments.values() if split == "test"),
        }
        train_samples = _collect_split_samples(groups, assignments, split="train")
        val_samples = _collect_split_samples(groups, assignments, split="val")
        test_samples = _collect_split_samples(groups, assignments, split="test")

    paths = {"train": output / "train.jsonl", "val": output / "val.jsonl", "test": output / "test.jsonl"}
    _write_jsonl(paths["train"], train_samples)
    _write_jsonl(paths["val"], val_samples)
    _write_jsonl(paths["test"], test_samples)

    result = ContextBcPrepareResult(
        ok=not errors,
        output_dir=str(output),
        manifest_path=str(manifest_path),
        input_paths=input_paths,
        sample_count=sample_count,
        valid_sample_count=len(valid_samples),
        invalid_sample_count=invalid_count,
        skipped_invalid_count=skipped_invalid_count,
        train=_split_info("train", paths["train"], train_samples, split_group_counts["train"]),
        val=_split_info("val", paths["val"], val_samples, split_group_counts["val"]),
        test=_split_info("test", paths["test"], test_samples, split_group_counts["test"]),
        seed=seed,
        shuffle=shuffle,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        split_group_field=split_group_field,
        stratify_by_scenario=stratify_by_scenario,
        split_group_counts=split_group_counts,
        scenario_counts=_scenario_counts(valid_samples),
        input_sha256=input_sha256,
        contract_path=str(contract_path),
        errors=errors,
        warnings=warnings,
    )
    _write_manifest(result)
    return result


def print_prepare_summary(result: ContextBcPrepareResult) -> None:
    print("Soridormi context BC dataset prepare")
    print("=====================================")
    print(f"Output dir: {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Samples: {result.sample_count}")
    print(f"Valid samples: {result.valid_sample_count}")
    print(f"Invalid samples: {result.invalid_sample_count}")
    print(f"Split group field: {result.split_group_field}")
    print(f"Scenario-stratified: {result.stratify_by_scenario}")
    print("Splits:")
    for split in (result.train, result.val, result.test):
        print(f"  {split.name}: {split.sample_count} samples, {split.group_count} groups -> {split.path}")
    if result.warnings:
        print("Warnings:")
        for item in result.warnings[:20]:
            print(f"  - {item}")
    if result.errors:
        print("Errors:")
        for item in result.errors[:40]:
            print(f"  - {item}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare context BC JSONL rows into train/val/test splits without rollout leakage.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input context BC JSONL files, dirs, or prepared manifests")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONTEXT_PREPARED_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Preserve first-seen group ordering instead of deterministic hash shuffling")
    parser.add_argument("--split-group-field", default="rollout_id", help="Leakage boundary field; default: rollout_id")
    parser.add_argument("--no-stratify-by-scenario", action="store_true", help="Disable per-scenario group stratification")
    parser.add_argument("--skip-invalid", action="store_true", help="Allow invalid rows to be skipped without failing the prepare result")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_context_bc_dataset(
        args.inputs,
        output_dir=args.output_dir,
        contract_path=args.contract,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        split_group_field=args.split_group_field,
        stratify_by_scenario=not args.no_stratify_by_scenario,
        skip_invalid=args.skip_invalid,
        max_samples=args.max_samples,
    )
    if args.report is not None:
        write_markdown_report(result, args.report)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_prepare_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
