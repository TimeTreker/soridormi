from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file

TRAINING_STATS_SCHEMA_VERSION = 1
DEFAULT_STD_FLOOR = 1e-6


@dataclass
class VectorStats:
    size: int
    count: int
    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]


@dataclass
class SplitStats:
    name: str
    path: str
    sample_count: int
    observation: VectorStats | None = None
    action: VectorStats | None = None
    sha256: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingDatasetStatsResult:
    ok: bool
    prepared_manifest_path: str
    output_dir: str
    stats_path: str
    normalization_path: str
    report_path: str
    sample_count: int
    splits: dict[str, SplitStats]
    normalization: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _VectorAccumulator:
    def __init__(self, size: int) -> None:
        self.size = size
        self.count = 0
        self._sum = [0.0] * size
        self._sumsq = [0.0] * size
        self._min = [math.inf] * size
        self._max = [-math.inf] * size

    def add(self, values: list[float]) -> None:
        self.count += 1
        for index, value in enumerate(values):
            item = float(value)
            self._sum[index] += item
            self._sumsq[index] += item * item
            if item < self._min[index]:
                self._min[index] = item
            if item > self._max[index]:
                self._max[index] = item

    def to_stats(self) -> VectorStats:
        if self.count == 0:
            zeros = [0.0] * self.size
            return VectorStats(size=self.size, count=0, mean=zeros, std=zeros, min=zeros, max=zeros)

        mean = [value / self.count for value in self._sum]
        std: list[float] = []
        for index, value in enumerate(self._sumsq):
            variance = max(0.0, value / self.count - mean[index] * mean[index])
            std.append(math.sqrt(variance))
        return VectorStats(
            size=self.size,
            count=self.count,
            mean=mean,
            std=std,
            min=list(self._min),
            max=list(self._max),
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_vector(value: Any, *, size: int, field_name: str) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{field_name} must be a list"
    if len(value) != size:
        return None, f"{field_name} size {len(value)} != expected {size}"
    bad_indices = [index for index, item in enumerate(value) if not _is_finite_number(item)]
    if bad_indices:
        preview = ", ".join(str(index) for index in bad_indices[:8])
        return None, f"{field_name} contains non-finite/non-numeric values at indices {preview}"
    return [float(item) for item in value], None


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


def _resolve_manifest_path(prepared: str | Path) -> Path:
    path = Path(prepared)
    if path.is_dir():
        return path / "prepared_manifest.json"
    return path


def _path_from_manifest(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_manifest(prepared: str | Path) -> tuple[Path, dict[str, Any], list[str]]:
    manifest_path = _resolve_manifest_path(prepared)
    if not manifest_path.exists():
        return manifest_path, {}, [f"Prepared manifest not found: {manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return manifest_path, {}, [f"Prepared manifest is invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return manifest_path, {}, ["Prepared manifest must be a JSON object"]
    if payload.get("dataset_type") != "soridormi.policy_supervision.prepared.v1":
        return manifest_path, payload, ["Prepared manifest dataset_type must be soridormi.policy_supervision.prepared.v1"]
    return manifest_path, payload, []


def _analyze_split(
    name: str,
    path: Path,
    *,
    expected_count: int | None,
    observation_size: int,
    action_size: int,
    max_reported_issues: int,
) -> SplitStats:
    obs_acc = _VectorAccumulator(observation_size)
    action_acc = _VectorAccumulator(action_size)
    errors: list[str] = []
    warnings: list[str] = []
    sample_count = 0

    if not path.exists():
        return SplitStats(
            name=name,
            path=str(path),
            sample_count=0,
            errors=[f"Split file not found: {path}"],
        )

    for line_number, sample, parse_error in _iter_jsonl(path):
        if parse_error is not None or sample is None:
            if len(errors) < max_reported_issues:
                errors.append(parse_error or f"line {line_number}: invalid sample")
            continue
        obs, obs_error = _validate_vector(sample.get("observation"), size=observation_size, field_name="observation")
        action, action_error = _validate_vector(sample.get("action"), size=action_size, field_name="action")
        sample_errors = [error for error in (obs_error, action_error) if error]
        if sample_errors:
            if len(errors) < max_reported_issues:
                for error in sample_errors[: max(0, max_reported_issues - len(errors))]:
                    errors.append(f"line {line_number}: {error}")
            continue
        assert obs is not None
        assert action is not None
        obs_acc.add(obs)
        action_acc.add(action)
        sample_count += 1

    if expected_count is not None and expected_count != sample_count:
        warnings.append(f"manifest sample_count {expected_count} != observed valid sample_count {sample_count}")

    return SplitStats(
        name=name,
        path=str(path),
        sample_count=sample_count,
        observation=obs_acc.to_stats(),
        action=action_acc.to_stats(),
        sha256=sha256_file(path),
        errors=errors,
        warnings=warnings,
    )


def _floor_std(values: list[float], *, std_floor: float) -> list[float]:
    return [value if value >= std_floor else std_floor for value in values]


def _write_markdown_report(path: Path, result: TrainingDatasetStatsResult) -> None:
    lines = [
        "# Soridormi training dataset statistics",
        "",
        f"Prepared manifest: `{result.prepared_manifest_path}`",
        f"Sample count: **{result.sample_count}**",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        "",
        "## Splits",
        "",
        "| Split | Samples | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for split in result.splits.values():
        digest = split.sha256 or "n/a"
        lines.append(f"| {split.name} | {split.sample_count} | `{digest}` |")
    lines.extend([
        "",
        "## Normalization",
        "",
        "Normalization statistics are computed from the train split only.",
        "",
        f"Observation size: {len(result.normalization.get('observation_mean', []))}",
        f"Action size: {len(result.normalization.get('action_mean', []))}",
    ])
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_prepared_training_dataset(
    prepared: str | Path,
    *,
    output_dir: str | Path | None = None,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    std_floor: float = DEFAULT_STD_FLOOR,
    max_reported_issues: int = 50,
) -> TrainingDatasetStatsResult:
    manifest_path, manifest, manifest_errors = _load_manifest(prepared)
    output = Path(output_dir) if output_dir is not None else manifest_path.parent
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "dataset_stats.json"
    normalization_path = output / "normalization.json"
    report_path = output / "dataset_stats_report.md"

    if manifest_errors:
        result = TrainingDatasetStatsResult(
            ok=False,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            stats_path=str(stats_path),
            normalization_path=str(normalization_path),
            report_path=str(report_path),
            sample_count=0,
            splits={},
            normalization={},
            errors=manifest_errors,
        )
        stats_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown_report(report_path, result)
        return result

    split_payloads = manifest.get("splits")
    if not isinstance(split_payloads, dict):
        result = TrainingDatasetStatsResult(
            ok=False,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            stats_path=str(stats_path),
            normalization_path=str(normalization_path),
            report_path=str(report_path),
            sample_count=0,
            splits={},
            normalization={},
            errors=["Prepared manifest is missing splits"],
        )
        stats_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown_report(report_path, result)
        return result

    splits: dict[str, SplitStats] = {}
    errors: list[str] = []
    warnings: list[str] = []
    total_samples = 0
    for name in ("train", "val", "test"):
        payload = split_payloads.get(name)
        if not isinstance(payload, dict) or not payload.get("path"):
            split = SplitStats(name=name, path="", sample_count=0, errors=[f"Manifest missing {name} split path"])
        else:
            expected_count = payload.get("sample_count") if isinstance(payload.get("sample_count"), int) else None
            split = _analyze_split(
                name,
                _path_from_manifest(manifest_path, str(payload["path"])),
                expected_count=expected_count,
                observation_size=observation_size,
                action_size=action_size,
                max_reported_issues=max_reported_issues,
            )
        splits[name] = split
        total_samples += split.sample_count
        errors.extend(f"{name}: {error}" for error in split.errors)
        warnings.extend(f"{name}: {warning}" for warning in split.warnings)

    train = splits.get("train")
    normalization: dict[str, Any] = {
        "schema_version": TRAINING_STATS_SCHEMA_VERSION,
        "normalization_type": "soridormi.policy_supervision.normalization.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_prepared_manifest": str(manifest_path),
        "std_floor": std_floor,
    }
    if train is None or train.sample_count == 0 or train.observation is None or train.action is None:
        errors.append("train split has no valid samples; cannot compute normalization")
    else:
        normalization.update(
            {
                "sample_count": train.sample_count,
                "observation_mean": train.observation.mean,
                "observation_std": _floor_std(train.observation.std, std_floor=std_floor),
                "action_mean": train.action.mean,
                "action_std": _floor_std(train.action.std, std_floor=std_floor),
            }
        )

    result = TrainingDatasetStatsResult(
        ok=not errors,
        prepared_manifest_path=str(manifest_path),
        output_dir=str(output),
        stats_path=str(stats_path),
        normalization_path=str(normalization_path),
        report_path=str(report_path),
        sample_count=total_samples,
        splits=splits,
        normalization=normalization,
        errors=errors,
        warnings=warnings,
    )

    stats_payload = asdict(result)
    stats_payload["schema_version"] = TRAINING_STATS_SCHEMA_VERSION
    stats_payload["stats_type"] = "soridormi.policy_supervision.stats.v1"
    stats_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report_path, result)
    return result


def print_stats_summary(result: TrainingDatasetStatsResult) -> None:
    print("Soridormi training dataset statistics")
    print("======================================")
    print(f"Prepared manifest: {result.prepared_manifest_path}")
    print(f"Output dir: {result.output_dir}")
    print(f"Stats: {result.stats_path}")
    print(f"Normalization: {result.normalization_path}")
    print(f"Report: {result.report_path}")
    print(f"Samples: {result.sample_count}")
    print("Splits:")
    for split in result.splits.values():
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
    parser = argparse.ArgumentParser(description="Compute training statistics for a prepared Soridormi dataset.")
    parser.add_argument("prepared", type=Path, help="Prepared dataset directory or prepared_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for stats artifacts")
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args()

    result = analyze_prepared_training_dataset(
        args.prepared,
        output_dir=args.output_dir,
        observation_size=args.observation_size,
        action_size=args.action_size,
        std_floor=args.std_floor,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_stats_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
