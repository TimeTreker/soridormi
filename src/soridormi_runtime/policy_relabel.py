from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soridormi_runtime.evaluate_policy_profile import _predict_profile
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_ACTION_SIZE,
    DEFAULT_OBSERVATION_SIZE,
    TrainingStepRecord,
    _state_summary,
    load_training_records,
    sha256_file,
)

RELABEL_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("/data/training_datasets/dagger")


@dataclass
class RelabelResult:
    ok: bool
    teacher_profile_name: str
    teacher_profile_path: str
    output_path: str
    manifest_path: str
    source_logs: list[str]
    candidate_sample_count: int
    relabeled_sample_count: int
    skipped_record_count: int
    observation_size: int = DEFAULT_OBSERVATION_SIZE
    action_size: int = DEFAULT_ACTION_SIZE
    dataset_sha256: str | None = None
    mean_abs_teacher_delta: float | None = None
    max_abs_teacher_delta: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MergeDatasetResult:
    ok: bool
    output_path: str
    manifest_path: str
    input_paths: list[str]
    sample_count: int
    skipped_line_count: int = 0
    deduplicated_count: int = 0
    dataset_sha256: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _finite_vector(value: Any, *, size: int) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    out: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _next_state(records: list[TrainingStepRecord], index: int) -> tuple[dict[str, Any] | None, float | None]:
    if index + 1 >= len(records):
        return None, None
    current = records[index]
    nxt = records[index + 1]
    if current.robot_time is not None and nxt.robot_time is not None:
        if float(nxt.robot_time) + 1e-9 < float(current.robot_time):
            return None, None
    return _state_summary(nxt.state), nxt.robot_time


def _candidate_records(
    logs: Iterable[str | Path],
    *,
    observation_size: int,
    max_samples: int | None = None,
) -> tuple[list[tuple[TrainingStepRecord, dict[str, Any] | None, float | None]], int, list[str], list[str]]:
    selected: list[tuple[TrainingStepRecord, dict[str, Any] | None, float | None]] = []
    skipped = 0
    errors: list[str] = []
    warnings: list[str] = []
    for log in logs:
        log_path = Path(log)
        try:
            records = load_training_records(log_path)
        except Exception as exc:
            errors.append(f"{log_path}: failed to read log: {exc!r}")
            continue
        for index, record in enumerate(records):
            observation = _finite_vector(record.observation, size=observation_size)
            if observation is None:
                skipped += 1
                continue
            record.observation = observation
            ns, nrt = _next_state(records, index)
            selected.append((record, ns, nrt))
            if max_samples is not None and max_samples > 0 and len(selected) >= max_samples:
                return selected, skipped, errors, warnings
    if not selected and not errors:
        errors.append("No relabelable observations found in candidate logs")
    return selected, skipped, errors, warnings


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relabel_policy_rollouts_with_teacher(
    logs: Iterable[str | Path],
    *,
    teacher_profile: str | Path | PolicyProfile = "open_duck_forward",
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    max_samples: int | None = None,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
) -> RelabelResult:
    """Create DAgger-style supervised samples from candidate rollout observations.

    Candidate rollout logs contain the states visited by a replacement policy. This
    function keeps those observations but replaces the action target with the
    teacher profile's action. That is the core teacher relabeling data-iteration loop:
    collect candidate states -> ask teacher for labels -> retrain on the expanded
    state distribution.
    """
    profile = teacher_profile if isinstance(teacher_profile, PolicyProfile) else PolicyProfile.load(teacher_profile)
    log_paths = [Path(path) for path in logs]
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_ROOT / f"teacher_relabel_{utc_stamp()}.jsonl"
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(output.suffix + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    records, skipped, errors, warnings = _candidate_records(
        log_paths,
        observation_size=observation_size,
        max_samples=max_samples,
    )
    if errors:
        result = RelabelResult(
            ok=False,
            teacher_profile_name=profile.name,
            teacher_profile_path=str(profile.path),
            output_path=str(output),
            manifest_path=str(manifest),
            source_logs=[str(path) for path in log_paths],
            candidate_sample_count=len(records),
            relabeled_sample_count=0,
            skipped_record_count=skipped,
            observation_size=observation_size,
            action_size=action_size,
            errors=errors,
            warnings=warnings,
        )
        _write_manifest(manifest, {"schema_version": RELABEL_SCHEMA_VERSION, "ok": result.ok, **asdict(result)})
        output.write_text("", encoding="utf-8")
        return result

    observations = np.asarray([record.observation for record, _ns, _nrt in records], dtype=np.float64)
    predictions, predict_errors, predict_warnings, _teacher_sha = _predict_profile(
        profile,
        observations,
        providers=providers,
        require_providers=require_providers,
        prefer_cuda=prefer_cuda,
    )
    errors.extend(predict_errors)
    warnings.extend(predict_warnings)
    if predictions.shape != (len(records), action_size):
        errors.append(f"teacher prediction shape {list(predictions.shape)} != expected {[len(records), action_size]}")
    if errors:
        result = RelabelResult(
            ok=False,
            teacher_profile_name=profile.name,
            teacher_profile_path=str(profile.path),
            output_path=str(output),
            manifest_path=str(manifest),
            source_logs=[str(path) for path in log_paths],
            candidate_sample_count=len(records),
            relabeled_sample_count=0,
            skipped_record_count=skipped,
            observation_size=observation_size,
            action_size=action_size,
            errors=errors,
            warnings=warnings,
        )
        _write_manifest(manifest, {"schema_version": RELABEL_SCHEMA_VERSION, "ok": result.ok, **asdict(result)})
        output.write_text("", encoding="utf-8")
        return result

    deltas: list[float] = []
    max_delta = 0.0
    with output.open("w", encoding="utf-8") as f:
        for index, ((record, next_state, next_robot_time), teacher_action_np) in enumerate(zip(records, predictions)):
            teacher_action = [float(x) for x in teacher_action_np.reshape(-1)]
            source_action = _finite_vector(record.action, size=action_size)
            if source_action is not None:
                diff = np.asarray(teacher_action, dtype=np.float64) - np.asarray(source_action, dtype=np.float64)
                deltas.append(float(np.mean(np.abs(diff))))
                max_delta = max(max_delta, float(np.max(np.abs(diff))))
            sample = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_type": "soridormi.policy_supervision.v1",
                "relabel_type": "soridormi.dagger_teacher_action.v1",
                "teacher_profile": profile.name,
                "source_log": str(record.source_log),
                "source_policy_action": source_action,
                "step_index": int(record.step_index),
                "robot_time": record.robot_time,
                "next_robot_time": next_robot_time,
                "mode": record.mode,
                "backend": record.backend,
                "observation": [float(x) for x in record.observation or []],
                "action": teacher_action,
                "raw_action": record.raw_action,
                "policy_command": record.policy_debug.get("command") if isinstance(record.policy_debug, dict) else None,
                "motor_command": record.command,
                "state": _state_summary(record.state),
                "next_state": next_state,
                "policy_debug": record.policy_debug,
            }
            f.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")

    dataset_sha = sha256_file(output)
    result = RelabelResult(
        ok=True,
        teacher_profile_name=profile.name,
        teacher_profile_path=str(profile.path),
        output_path=str(output),
        manifest_path=str(manifest),
        source_logs=[str(path) for path in log_paths],
        candidate_sample_count=len(records),
        relabeled_sample_count=len(records),
        skipped_record_count=skipped,
        observation_size=observation_size,
        action_size=action_size,
        dataset_sha256=dataset_sha,
        mean_abs_teacher_delta=float(np.mean(deltas)) if deltas else None,
        max_abs_teacher_delta=max_delta if deltas else None,
        errors=errors,
        warnings=warnings,
    )
    manifest_payload = {
        "schema_version": RELABEL_SCHEMA_VERSION,
        "dataset_type": "soridormi.dagger_teacher_relabel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": result.ok,
        **asdict(result),
    }
    _write_manifest(manifest, manifest_payload)
    return result


def _sample_key(sample: dict[str, Any]) -> str:
    observation = sample.get("observation")
    action = sample.get("action")
    payload = {"observation": observation, "action": action}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def merge_supervised_datasets(
    datasets: Iterable[str | Path],
    *,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    deduplicate: bool = True,
) -> MergeDatasetResult:
    inputs = [Path(path) for path in datasets]
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(output.suffix + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    sample_count = 0
    skipped = 0
    deduped = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                errors.append(f"Dataset not found: {path}")
                continue
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        sample = json.loads(text)
                    except json.JSONDecodeError:
                        skipped += 1
                        warnings.append(f"{path}:{line_number}: skipped invalid JSON")
                        continue
                    if not isinstance(sample, dict):
                        skipped += 1
                        continue
                    key = _sample_key(sample)
                    if deduplicate and key in seen:
                        deduped += 1
                        continue
                    seen.add(key)
                    out.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
                    sample_count += 1
    if sample_count == 0 and not errors:
        errors.append("No samples were merged")
    dataset_sha = sha256_file(output) if output.exists() else None
    result = MergeDatasetResult(
        ok=not errors,
        output_path=str(output),
        manifest_path=str(manifest),
        input_paths=[str(path) for path in inputs],
        sample_count=sample_count,
        skipped_line_count=skipped,
        deduplicated_count=deduped,
        dataset_sha256=dataset_sha,
        errors=errors,
        warnings=warnings,
    )
    _write_manifest(
        manifest,
        {
            "schema_version": RELABEL_SCHEMA_VERSION,
            "dataset_type": "soridormi.policy_supervision.merged.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ok": result.ok,
            **asdict(result),
        },
    )
    return result


def print_relabel_summary(result: RelabelResult) -> None:
    print("Soridormi teacher relabel dataset")
    print("=================================")
    print(f"Teacher: {result.teacher_profile_name}")
    print(f"Output: {result.output_path}")
    print(f"Samples: {result.relabeled_sample_count}")
    print(f"Skipped records: {result.skipped_record_count}")
    if result.mean_abs_teacher_delta is not None:
        print(f"Mean abs teacher delta: {result.mean_abs_teacher_delta:.6g}")
    if result.max_abs_teacher_delta is not None:
        print(f"Max abs teacher delta: {result.max_abs_teacher_delta:.6g}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:30]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def print_merge_summary(result: MergeDatasetResult) -> None:
    print("Soridormi supervised dataset merge")
    print("==================================")
    print(f"Output: {result.output_path}")
    print(f"Samples: {result.sample_count}")
    print(f"Deduplicated: {result.deduplicated_count}")
    print(f"Skipped lines: {result.skipped_line_count}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:30]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def _relabel_main(args: argparse.Namespace) -> RelabelResult:
    return relabel_policy_rollouts_with_teacher(
        args.logs,
        teacher_profile=args.teacher_profile,
        output_path=args.output,
        manifest_path=args.manifest,
        max_samples=args.max_samples,
        providers=args.providers,
        require_providers=args.require_provider,
        prefer_cuda=not args.no_prefer_cuda,
    )


def _merge_main(args: argparse.Namespace) -> MergeDatasetResult:
    return merge_supervised_datasets(
        args.datasets,
        output_path=args.output,
        manifest_path=args.manifest,
        deduplicate=not args.no_deduplicate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="teacher relabeling DAgger-style teacher relabel and dataset merge utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    relabel = sub.add_parser("relabel", help="Relabel candidate rollout observations with a teacher policy")
    relabel.add_argument("logs", nargs="+", help="Candidate rollout .jsonl/.mcap logs")
    relabel.add_argument("--teacher-profile", default="open_duck_forward", help="Teacher policy profile")
    relabel.add_argument("--output", type=Path, default=None, help="Output relabeled dataset JSONL")
    relabel.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON")
    relabel.add_argument("--max-samples", type=int, default=None, help="Limit relabeled samples")
    relabel.add_argument("--providers", default=None, help="Comma-separated ONNX Runtime providers for ONNX teacher")
    relabel.add_argument("--require-provider", action="append", default=None, help="Required ONNX Runtime provider; repeatable")
    relabel.add_argument("--no-prefer-cuda", action="store_true")
    relabel.add_argument("--json", action="store_true")

    merge = sub.add_parser("merge", help="Merge supervised JSONL datasets")
    merge.add_argument("datasets", nargs="+", help="Input supervised JSONL datasets")
    merge.add_argument("--output", type=Path, required=True, help="Output merged JSONL")
    merge.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON")
    merge.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate observation/action samples")
    merge.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "relabel":
        result = _relabel_main(args)
        if args.json:
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
        else:
            print_relabel_summary(result)
    else:
        result = _merge_main(args)
        if args.json:
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
        else:
            print_merge_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
