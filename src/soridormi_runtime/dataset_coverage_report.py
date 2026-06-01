from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file
from soridormi_runtime.training_dataset_prepare import validate_training_sample

DATASET_COVERAGE_SCHEMA_VERSION = 1
DEFAULT_HISTOGRAM_BINS = 8


@dataclass
class NumericCoverage:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    abs_mean: float | None = None
    histogram: list[dict[str, float | int]] = field(default_factory=list)


@dataclass
class CategoricalCoverage:
    total: int = 0
    distinct_count: int = 0
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class FailureCoverage:
    total: int = 0
    fall_count: int = 0
    terminated_count: int = 0
    stuck_count: int = 0
    failure_count: int = 0
    fall_ratio: float = 0.0
    terminated_ratio: float = 0.0
    stuck_ratio: float = 0.0
    failure_ratio: float = 0.0


@dataclass
class DatasetCoverageResult:
    ok: bool
    inputs: list[str]
    output_dir: str
    summary_path: str
    report_path: str
    sample_count: int
    valid_sample_count: int
    invalid_sample_count: int
    scenario_coverage: CategoricalCoverage
    skill_coverage: CategoricalCoverage
    split_coverage: CategoricalCoverage
    terrain_coverage: CategoricalCoverage
    tag_coverage: CategoricalCoverage
    command_coverage: dict[str, dict[str, NumericCoverage]]
    ramp_alpha_coverage: NumericCoverage
    failure_coverage: FailureCoverage
    sha256_by_input: dict[str, str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _NumericAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: Any) -> None:
        number = _float_or_none(value)
        if number is not None:
            self.values.append(number)

    def to_coverage(self, *, bins: int) -> NumericCoverage:
        values = self.values
        if not values:
            return NumericCoverage()
        count = len(values)
        minimum = min(values)
        maximum = max(values)
        mean = sum(values) / count
        abs_mean = sum(abs(value) for value in values) / count
        return NumericCoverage(
            count=count,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            abs_mean=abs_mean,
            histogram=_histogram(values, bins=max(1, int(bins))),
        )


class _CategoricalAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.counts: dict[str, int] = {}

    def add(self, value: Any) -> None:
        text = str(value).strip() if value is not None else ""
        if not text:
            text = "unknown"
        self.total += 1
        self.counts[text] = self.counts.get(text, 0) + 1

    def add_many(self, values: Any) -> None:
        if isinstance(values, list):
            for value in values:
                self.add(value)
            return
        self.add(values)

    def to_coverage(self) -> CategoricalCoverage:
        counts = dict(sorted(self.counts.items(), key=lambda item: (-item[1], item[0])))
        return CategoricalCoverage(total=self.total, distinct_count=len(counts), counts=counts)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on", "failed", "failure", "stuck", "fall"}:
            return True
        if text in {"0", "false", "no", "n", "off", "ok", "success", "none"}:
            return False
    return None


def _histogram(values: list[float], *, bins: int) -> list[dict[str, float | int]]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [{"minimum": minimum, "maximum": maximum, "count": len(values)}]
    width = (maximum - minimum) / float(bins)
    counts = [0 for _ in range(bins)]
    for value in values:
        index = int((value - minimum) / width)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    buckets: list[dict[str, float | int]] = []
    for index, count in enumerate(counts):
        low = minimum + width * index
        high = maximum if index == bins - 1 else minimum + width * (index + 1)
        buckets.append({"minimum": low, "maximum": high, "count": count})
    return buckets


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
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


def _resolve_input_paths(inputs: Iterable[str | Path]) -> tuple[list[tuple[str, Path]], list[str]]:
    resolved: list[tuple[str, Path]] = []
    errors: list[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            manifest = path / "prepared_manifest.json"
            if manifest.exists():
                resolved.extend(_paths_from_prepared_manifest(manifest, errors))
                continue
            jsonl_files = sorted(path.glob("*.jsonl"))
            if jsonl_files:
                resolved.extend((item.stem, item) for item in jsonl_files)
                continue
            errors.append(f"directory has no prepared_manifest.json or JSONL files: {path}")
            continue
        if path.name == "prepared_manifest.json" or path.suffix.lower() == ".json":
            resolved.extend(_paths_from_prepared_manifest(path, errors))
            continue
        resolved.append((path.stem, path))
    return resolved, errors


def _paths_from_prepared_manifest(manifest_path: Path, errors: list[str]) -> list[tuple[str, Path]]:
    if not manifest_path.exists():
        errors.append(f"prepared manifest not found: {manifest_path}")
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"prepared manifest is invalid JSON: {manifest_path}: {exc}")
        return []
    if not isinstance(payload, dict):
        errors.append(f"prepared manifest must be a JSON object: {manifest_path}")
        return []
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        errors.append(f"prepared manifest is missing splits: {manifest_path}")
        return []
    resolved: list[tuple[str, Path]] = []
    for name in ("train", "val", "test"):
        split = splits.get(name)
        if not isinstance(split, dict) or not split.get("path"):
            continue
        split_path = Path(str(split["path"]))
        if not split_path.is_absolute():
            split_path = manifest_path.parent / split_path
        resolved.append((name, split_path))
    return resolved


def _command_value(sample: dict[str, Any], source: str, key: str) -> float | None:
    if source in {"applied_command", "desired_command"}:
        payload = sample.get(source)
        if isinstance(payload, dict):
            aliases = {
                "vx_mps": ("x_velocity", "vx_mps", "vx"),
                "vy_mps": ("y_velocity", "vy_mps", "vy"),
                "yaw_radps": ("yaw_velocity", "yaw_radps", "yaw"),
            }
            for alias in aliases[key]:
                if alias in payload:
                    return _float_or_none(payload.get(alias))
        if source == "desired_command":
            target = sample.get("policy_command_target")
            if isinstance(target, list):
                return _command_list_value(target, key)
        return None
    if source == "policy_command":
        payload = sample.get("policy_command")
        if isinstance(payload, list):
            return _command_list_value(payload, key)
    return None


def _command_list_value(values: list[Any], key: str) -> float | None:
    index_by_key = {"vx_mps": 0, "vy_mps": 1, "yaw_radps": 2}
    index = index_by_key[key]
    if len(values) <= index:
        return None
    return _float_or_none(values[index])


def _terrain_type(sample: dict[str, Any]) -> str:
    env_context = sample.get("environment_context")
    if isinstance(env_context, dict) and env_context.get("terrain_type"):
        return str(env_context["terrain_type"])
    return str(sample.get("terrain_type", "unknown") or "unknown")


def _dataset_tags(sample: dict[str, Any]) -> list[str]:
    tags = sample.get("scenario_dataset_tags", sample.get("dataset_tags", []))
    if isinstance(tags, list):
        return [str(item) for item in tags if str(item).strip()]
    if tags:
        return [str(tags)]
    return ["unknown"]


def _nested_dicts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    out = [sample]
    for key in ("metrics", "policy_debug", "failure_metadata", "evaluation", "eval_metrics"):
        value = sample.get(key)
        if isinstance(value, dict):
            out.append(value)
    return out


def _flag_from_keys(sample: dict[str, Any], keys: Iterable[str]) -> bool:
    for payload in _nested_dicts(sample):
        for key in keys:
            if key not in payload:
                continue
            as_bool = _bool_or_none(payload.get(key))
            if as_bool is not None:
                return as_bool
            as_float = _float_or_none(payload.get(key))
            if as_float is not None:
                return as_float > 0.0
    return False


def _failure_flags(sample: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    fall = _flag_from_keys(sample, ("fall", "fallen", "fell", "fall_detected", "is_fall"))
    terminated = _flag_from_keys(sample, ("terminated", "done", "episode_terminated"))
    stuck = _flag_from_keys(sample, ("stuck", "is_stuck", "stuck_detected"))
    if not stuck:
        for payload in _nested_dicts(sample):
            stuck_ratio = _float_or_none(payload.get("stuck_ratio"))
            if stuck_ratio is not None and stuck_ratio > 0.0:
                stuck = True
                break
    explicit_failure = _flag_from_keys(sample, ("failure", "failed", "is_failure"))
    success = _bool_or_none(sample.get("success"))
    if success is False:
        explicit_failure = True
    failure = bool(explicit_failure or fall or stuck or terminated)
    return fall, terminated, stuck, failure


def analyze_dataset_coverage(
    inputs: Iterable[str | Path] | str | Path,
    *,
    output_dir: str | Path | None = None,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    max_reported_issues: int = 50,
) -> DatasetCoverageResult:
    input_values: Iterable[str | Path]
    if isinstance(inputs, (str, Path)):
        input_values = [inputs]
    else:
        input_values = inputs
    input_items, input_errors = _resolve_input_paths(input_values)
    output = Path(output_dir) if output_dir is not None else Path("/data/training_datasets/coverage")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "dataset_coverage_summary.json"
    report_path = output / "dataset_coverage_report.md"

    scenario_acc = _CategoricalAccumulator()
    skill_acc = _CategoricalAccumulator()
    split_acc = _CategoricalAccumulator()
    terrain_acc = _CategoricalAccumulator()
    tag_acc = _CategoricalAccumulator()
    ramp_alpha_acc = _NumericAccumulator()
    command_acc = {
        "applied_command": {
            "vx_mps": _NumericAccumulator(),
            "vy_mps": _NumericAccumulator(),
            "yaw_radps": _NumericAccumulator(),
        },
        "desired_command": {
            "vx_mps": _NumericAccumulator(),
            "vy_mps": _NumericAccumulator(),
            "yaw_radps": _NumericAccumulator(),
        },
        "policy_command": {
            "vx_mps": _NumericAccumulator(),
            "vy_mps": _NumericAccumulator(),
            "yaw_radps": _NumericAccumulator(),
        },
    }

    errors = list(input_errors)
    warnings: list[str] = []
    sample_count = 0
    valid_sample_count = 0
    invalid_sample_count = 0
    fall_count = 0
    terminated_count = 0
    stuck_count = 0
    failure_count = 0
    sha_by_input: dict[str, str] = {}

    if not input_items and not errors:
        errors.append("No dataset inputs were provided")

    for split_name, path in input_items:
        if not path.exists():
            errors.append(f"dataset input not found: {path}")
            continue
        sha_by_input[str(path)] = sha256_file(path)
        for line_number, sample, parse_error in _iter_jsonl(path):
            sample_count += 1
            split_acc.add(split_name)
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

            scenario_acc.add(sample.get("scenario_id"))
            skill_acc.add(sample.get("skill_id"))
            terrain_acc.add(_terrain_type(sample))
            tag_acc.add_many(_dataset_tags(sample))
            ramp_alpha_acc.add(sample.get("command_ramp_alpha"))
            for source, per_source in command_acc.items():
                for key, acc in per_source.items():
                    acc.add(_command_value(sample, source, key))

            fall, terminated, stuck, failure = _failure_flags(sample)
            fall_count += int(fall)
            terminated_count += int(terminated)
            stuck_count += int(stuck)
            failure_count += int(failure)

    total = float(valid_sample_count) if valid_sample_count else 1.0
    failure_coverage = FailureCoverage(
        total=valid_sample_count,
        fall_count=fall_count,
        terminated_count=terminated_count,
        stuck_count=stuck_count,
        failure_count=failure_count,
        fall_ratio=fall_count / total,
        terminated_ratio=terminated_count / total,
        stuck_ratio=stuck_count / total,
        failure_ratio=failure_count / total,
    )
    command_coverage = {
        source: {key: acc.to_coverage(bins=histogram_bins) for key, acc in per_source.items()}
        for source, per_source in command_acc.items()
    }
    if valid_sample_count == 0 and not errors:
        errors.append("No valid samples were found")

    result = DatasetCoverageResult(
        ok=not errors,
        inputs=[str(path) for _split, path in input_items],
        output_dir=str(output),
        summary_path=str(summary_path),
        report_path=str(report_path),
        sample_count=sample_count,
        valid_sample_count=valid_sample_count,
        invalid_sample_count=invalid_sample_count,
        scenario_coverage=scenario_acc.to_coverage(),
        skill_coverage=skill_acc.to_coverage(),
        split_coverage=split_acc.to_coverage(),
        terrain_coverage=terrain_acc.to_coverage(),
        tag_coverage=tag_acc.to_coverage(),
        command_coverage=command_coverage,
        ramp_alpha_coverage=ramp_alpha_acc.to_coverage(bins=histogram_bins),
        failure_coverage=failure_coverage,
        sha256_by_input=sha_by_input,
        errors=errors,
        warnings=warnings,
    )
    summary_payload = asdict(result)
    summary_payload["schema_version"] = DATASET_COVERAGE_SCHEMA_VERSION
    summary_payload["coverage_type"] = "soridormi.policy_supervision.coverage.v1"
    summary_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report_path, result)
    return result


def _write_markdown_report(path: Path, result: DatasetCoverageResult) -> None:
    lines = [
        "# Soridormi dataset coverage report",
        "",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        f"Samples: **{result.valid_sample_count} valid** / {result.sample_count} total",
        "",
        "## Scenario coverage",
        "",
        "| Scenario | Samples |",
        "| --- | ---: |",
    ]
    for key, count in result.scenario_coverage.counts.items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Skill coverage", "", "| Skill | Samples |", "| --- | ---: |"])
    for key, count in result.skill_coverage.counts.items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Terrain coverage", "", "| Terrain | Samples |", "| --- | ---: |"])
    for key, count in result.terrain_coverage.counts.items():
        lines.append(f"| `{key}` | {count} |")
    applied = result.command_coverage.get("applied_command", {})
    lines.extend(["", "## Applied command distribution", "", "| Command | Count | Min | Mean | Max |", "| --- | ---: | ---: | ---: | ---: |"])
    for key in ("vx_mps", "vy_mps", "yaw_radps"):
        coverage = applied.get(key, NumericCoverage())
        lines.append(
            f"| `{key}` | {coverage.count} | {_fmt(coverage.minimum)} | "
            f"{_fmt(coverage.mean)} | {_fmt(coverage.maximum)} |"
        )
    lines.extend(
        [
            "",
            "## Ramp and failure flags",
            "",
            f"Ramp alpha samples: {result.ramp_alpha_coverage.count}",
            f"Failure ratio: {result.failure_coverage.failure_ratio:.3f}",
            f"Fall ratio: {result.failure_coverage.fall_ratio:.3f}",
            f"Stuck ratio: {result.failure_coverage.stuck_ratio:.3f}",
            f"Terminated ratio: {result.failure_coverage.terminated_ratio:.3f}",
        ]
    )
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_coverage_summary(result: DatasetCoverageResult) -> None:
    print("Soridormi dataset coverage")
    print("===========================")
    print(f"Inputs: {len(result.inputs)}")
    print(f"Output dir: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print(f"Report: {result.report_path}")
    print(f"Samples: {result.valid_sample_count} valid / {result.sample_count} total")
    print(f"Scenarios: {result.scenario_coverage.distinct_count}")
    print(f"Skills: {result.skill_coverage.distinct_count}")
    print(f"Terrains: {result.terrain_coverage.distinct_count}")
    print(
        "Failures: "
        f"fall={result.failure_coverage.fall_count}, "
        f"stuck={result.failure_coverage.stuck_count}, "
        f"terminated={result.failure_coverage.terminated_count}"
    )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:20]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors[:20]:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report scenario, command, terrain, ramp, and failure coverage for Soridormi datasets."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Dataset JSONL, prepared manifest, or prepared directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for coverage summary/report artifacts")
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args(argv)

    result = analyze_dataset_coverage(
        args.inputs,
        output_dir=args.output_dir,
        histogram_bins=args.histogram_bins,
        observation_size=args.observation_size,
        action_size=args.action_size,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_coverage_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
