from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_ACTION_SIZE,
    DEFAULT_OBSERVATION_SIZE,
    sha256_file,
)

PREPARED_DATASET_SCHEMA_VERSION = 1
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


@dataclass
class DatasetValidationSummary:
    ok: bool
    dataset_path: str
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    observation_size: int
    action_size: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetSplitInfo:
    name: str
    path: str
    sample_count: int
    sha256: str | None = None


@dataclass
class DatasetPrepareResult:
    ok: bool
    input_path: str
    output_dir: str
    manifest_path: str
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    train: DatasetSplitInfo
    val: DatasetSplitInfo
    test: DatasetSplitInfo
    seed: int
    shuffle: bool
    ratios: dict[str, float]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_numeric_vector(value: Any, *, size: int, field_name: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{field_name} must be a list"]
    errors: list[str] = []
    if len(value) != size:
        errors.append(f"{field_name} size {len(value)} != expected {size}")
    bad_indices = [index for index, item in enumerate(value) if not _is_finite_number(item)]
    if bad_indices:
        preview = ", ".join(str(index) for index in bad_indices[:8])
        errors.append(f"{field_name} contains non-finite/non-numeric values at indices {preview}")
    return errors


def validate_training_sample(
    sample: dict[str, Any],
    *,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    require_next_state: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate a single supervised policy sample.

    The exporter is intentionally permissive about metadata, but the training
    preflight should be strict about the vectors that a learner consumes.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if sample.get("sample_type") != "soridormi.policy_supervision.v1":
        errors.append("sample_type must be soridormi.policy_supervision.v1")
    if sample.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append(f"schema_version must be {DATASET_SCHEMA_VERSION}")

    errors.extend(_validate_numeric_vector(sample.get("observation"), size=observation_size, field_name="observation"))
    errors.extend(_validate_numeric_vector(sample.get("action"), size=action_size, field_name="action"))

    raw_action = sample.get("raw_action")
    if raw_action is not None:
        errors.extend(_validate_numeric_vector(raw_action, size=action_size, field_name="raw_action"))

    policy_command = sample.get("policy_command")
    if policy_command is not None:
        if not isinstance(policy_command, list):
            errors.append("policy_command must be a list when present")
        elif len(policy_command) != 7:
            warnings.append(f"policy_command size {len(policy_command)} != expected 7")
        elif any(not _is_finite_number(value) for value in policy_command):
            errors.append("policy_command contains non-finite/non-numeric values")

    if sample.get("step_index") is None:
        warnings.append("step_index is missing")
    if sample.get("robot_time") is None:
        warnings.append("robot_time is missing")
    if require_next_state and sample.get("next_state") is None:
        errors.append("next_state is required but missing")

    return errors, warnings


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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


def load_and_validate_dataset(
    dataset_path: str | Path,
    *,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    require_next_state: bool = False,
    max_reported_issues: int = 50,
) -> tuple[list[dict[str, Any]], DatasetValidationSummary]:
    path = Path(dataset_path)
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    total = 0
    invalid = 0

    if not path.exists():
        return [], DatasetValidationSummary(
            ok=False,
            dataset_path=str(path),
            sample_count=0,
            valid_sample_count=0,
            invalid_sample_count=0,
            observation_size=observation_size,
            action_size=action_size,
            errors=[f"Dataset not found: {path}"],
        )

    for line_number, sample, parse_error in _iter_jsonl(path):
        total += 1
        if parse_error is not None or sample is None:
            invalid += 1
            if len(errors) < max_reported_issues:
                errors.append(parse_error or f"line {line_number}: invalid sample")
            continue
        sample_errors, sample_warnings = validate_training_sample(
            sample,
            observation_size=observation_size,
            action_size=action_size,
            require_next_state=require_next_state,
        )
        if sample_errors:
            invalid += 1
            if len(errors) < max_reported_issues:
                for error in sample_errors[: max(0, max_reported_issues - len(errors))]:
                    errors.append(f"line {line_number}: {error}")
            continue
        if sample_warnings and len(warnings) < max_reported_issues:
            for warning in sample_warnings[: max(0, max_reported_issues - len(warnings))]:
                warnings.append(f"line {line_number}: {warning}")
        samples.append(sample)

    if total == 0:
        errors.append("Dataset is empty")
    return samples, DatasetValidationSummary(
        ok=not errors,
        dataset_path=str(path),
        sample_count=total,
        valid_sample_count=len(samples),
        invalid_sample_count=invalid,
        observation_size=observation_size,
        action_size=action_size,
        errors=errors,
        warnings=warnings,
    )


def _sample_key(sample: dict[str, Any], index: int) -> str:
    source_log = sample.get("source_log", "")
    step_index = sample.get("step_index", index)
    robot_time = sample.get("robot_time", "")
    return f"{source_log}|{step_index}|{robot_time}|{index}"


def _ordered_samples(samples: list[dict[str, Any]], *, seed: int, shuffle: bool) -> list[dict[str, Any]]:
    indexed = list(enumerate(samples))
    if shuffle:
        indexed.sort(
            key=lambda pair: hashlib.sha256(f"{seed}|{_sample_key(pair[1], pair[0])}".encode("utf-8")).hexdigest()
        )
    return [sample for _index, sample in indexed]


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
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total > 0 and train_count == 0:
        train_count = 1
    if total >= 3 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")


def split_training_dataset(
    dataset_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 0,
    shuffle: bool = True,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    require_next_state: bool = False,
) -> DatasetPrepareResult:
    input_path = Path(dataset_path)
    output = Path(output_dir) if output_dir is not None else Path("/data/training_datasets/prepared") / input_path.stem
    output.mkdir(parents=True, exist_ok=True)

    ratio_errors = _validate_ratios(train_ratio, val_ratio, test_ratio)
    if ratio_errors:
        empty = DatasetSplitInfo(name="train", path=str(output / "train.jsonl"), sample_count=0)
        return DatasetPrepareResult(
            ok=False,
            input_path=str(input_path),
            output_dir=str(output),
            manifest_path=str(output / "prepared_manifest.json"),
            sample_count=0,
            valid_sample_count=0,
            invalid_sample_count=0,
            train=empty,
            val=DatasetSplitInfo(name="val", path=str(output / "val.jsonl"), sample_count=0),
            test=DatasetSplitInfo(name="test", path=str(output / "test.jsonl"), sample_count=0),
            seed=seed,
            shuffle=shuffle,
            ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
            errors=ratio_errors,
        )

    samples, validation = load_and_validate_dataset(
        input_path,
        observation_size=observation_size,
        action_size=action_size,
        require_next_state=require_next_state,
    )
    ordered = _ordered_samples(samples, seed=seed, shuffle=shuffle)
    train_count, val_count, test_count = _split_counts(len(ordered), train_ratio, val_ratio)
    train_samples = ordered[:train_count]
    val_samples = ordered[train_count : train_count + val_count]
    test_samples = ordered[train_count + val_count : train_count + val_count + test_count]

    split_paths = {
        "train": output / "train.jsonl",
        "val": output / "val.jsonl",
        "test": output / "test.jsonl",
    }
    _write_jsonl(split_paths["train"], train_samples)
    _write_jsonl(split_paths["val"], val_samples)
    _write_jsonl(split_paths["test"], test_samples)

    split_infos = {
        "train": DatasetSplitInfo(
            name="train",
            path=str(split_paths["train"]),
            sample_count=len(train_samples),
            sha256=sha256_file(split_paths["train"]),
        ),
        "val": DatasetSplitInfo(
            name="val",
            path=str(split_paths["val"]),
            sample_count=len(val_samples),
            sha256=sha256_file(split_paths["val"]),
        ),
        "test": DatasetSplitInfo(
            name="test",
            path=str(split_paths["test"]),
            sample_count=len(test_samples),
            sha256=sha256_file(split_paths["test"]),
        ),
    }

    errors = list(validation.errors)
    warnings = list(validation.warnings)
    if validation.valid_sample_count == 0 and not errors:
        errors.append("No valid samples were found")

    manifest_path = output / "prepared_manifest.json"
    result = DatasetPrepareResult(
        ok=not errors,
        input_path=str(input_path),
        output_dir=str(output),
        manifest_path=str(manifest_path),
        sample_count=validation.sample_count,
        valid_sample_count=validation.valid_sample_count,
        invalid_sample_count=validation.invalid_sample_count,
        train=split_infos["train"],
        val=split_infos["val"],
        test=split_infos["test"],
        seed=seed,
        shuffle=shuffle,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        errors=errors,
        warnings=warnings,
    )

    manifest = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_dir": str(output),
        "sample_count": result.sample_count,
        "valid_sample_count": result.valid_sample_count,
        "invalid_sample_count": result.invalid_sample_count,
        "observation_size": observation_size,
        "action_size": action_size,
        "seed": seed,
        "shuffle": shuffle,
        "ratios": result.ratios,
        "splits": {
            "train": asdict(result.train),
            "val": asdict(result.val),
            "test": asdict(result.test),
        },
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def print_prepare_summary(result: DatasetPrepareResult) -> None:
    print("Soridormi training dataset prepare")
    print("===================================")
    print(f"Input: {result.input_path}")
    print(f"Output dir: {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Samples: {result.sample_count}")
    print(f"Valid samples: {result.valid_sample_count}")
    print(f"Invalid samples: {result.invalid_sample_count}")
    print("Splits:")
    for split in (result.train, result.val, result.test):
        print(f"  {split.name}: {split.sample_count} -> {split.path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:20]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split a Soridormi supervised training dataset.")
    parser.add_argument("dataset", type=Path, help="Input supervised dataset JSONL")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for train/val/test JSONL files")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Keep input order instead of deterministic hash shuffling")
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--require-next-state", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args()

    result = split_training_dataset(
        args.dataset,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        observation_size=args.observation_size,
        action_size=args.action_size,
        require_next_state=args.require_next_state,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_prepare_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
