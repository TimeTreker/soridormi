from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASET_SCHEMA_VERSION = 1
DEFAULT_OBSERVATION_SIZE = 101
DEFAULT_ACTION_SIZE = 14


@dataclass
class TrainingStepRecord:
    source_log: Path
    step_index: int
    wall_time_ns: int | None = None
    robot_time: float | None = None
    mode: str | None = None
    backend: str | None = None
    observation: list[float] | None = None
    action: list[float] | None = None
    raw_action: list[float] | None = None
    policy_debug: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    command: dict[str, Any] | None = None


@dataclass
class TrainingDatasetExportResult:
    ok: bool
    output_path: str
    manifest_path: str
    sample_count: int
    source_logs: list[str]
    skipped_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_sha256: str | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_step_index(payload: dict[str, Any]) -> int | None:
    step = _int_or_none(payload.get("step_index"))
    if step is not None:
        return step
    debug = payload.get("debug", payload.get("policy_debug"))
    if isinstance(debug, dict):
        return _int_or_none(debug.get("step_count"))
    return None


def _iter_jsonl_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                continue
            yield str(payload.get("type", "runtime_step")), payload


def _iter_mcap_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError(
            "Cannot export datasets from MCAP because the 'mcap' package is not installed. "
            "Run this inside the runtime container."
        ) from exc

    with path.open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            if channel.message_encoding != "json":
                continue
            try:
                payload = json.loads(message.data.decode("utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                yield channel.topic, payload


def iter_log_payloads(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    log_path = Path(path)
    suffix = log_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_payloads(log_path)
    elif suffix == ".mcap":
        yield from _iter_mcap_payloads(log_path)
    else:
        raise ValueError(f"Unsupported log format: {log_path.suffix}. Expected .jsonl or .mcap")


def _record_for(records: dict[int, TrainingStepRecord], *, source_log: Path, payload: dict[str, Any]) -> TrainingStepRecord | None:
    step = _payload_step_index(payload)
    if step is None:
        return None
    record = records.get(step)
    if record is None:
        record = TrainingStepRecord(source_log=source_log, step_index=step)
        records[step] = record

    wall = _int_or_none(payload.get("time_wall_ns"))
    if wall is not None:
        record.wall_time_ns = wall
    robot_time = _float_or_none(payload.get("robot_time"))
    if robot_time is not None:
        record.robot_time = robot_time
    if payload.get("mode") is not None:
        record.mode = str(payload["mode"])
    if payload.get("backend") is not None:
        record.backend = str(payload["backend"])
    return record


def _ingest_payload(records: dict[int, TrainingStepRecord], *, source_log: Path, topic: str, payload: dict[str, Any]) -> None:
    record = _record_for(records, source_log=source_log, payload=payload)
    if record is None:
        return

    msg_type = str(payload.get("type", ""))
    topic_or_type = {topic, msg_type}

    if "/soridormi/policy_observation" in topic_or_type or "policy_observation" in payload or "observation" in payload:
        observation = _float_list(payload.get("observation", payload.get("policy_observation")))
        if observation is not None:
            record.observation = observation

    if "/soridormi/policy_action" in topic_or_type or "policy_action" in payload or msg_type == "policy_action":
        action = _float_list(payload.get("action", payload.get("policy_action")))
        if action is not None:
            record.action = action

    if "/soridormi/policy_raw_action" in topic_or_type or "policy_raw_action" in payload or msg_type == "policy_raw_action":
        raw_action = _float_list(payload.get("action", payload.get("policy_raw_action")))
        if raw_action is not None:
            record.raw_action = raw_action

    if "/soridormi/policy_debug" in topic_or_type or "policy_debug" in payload or "debug" in payload:
        debug = payload.get("debug", payload.get("policy_debug"))
        if isinstance(debug, dict):
            record.policy_debug = _json_safe(debug)
            if record.robot_time is None:
                debug_robot_time = _float_or_none(debug.get("robot_time"))
                if debug_robot_time is not None:
                    record.robot_time = debug_robot_time

    if "/soridormi/robot_state" in topic_or_type or "state" in payload:
        state = payload.get("state")
        if isinstance(state, dict):
            record.state = _json_safe(state)

    if "/soridormi/motor_command" in topic_or_type or "command" in payload:
        command = payload.get("command")
        if isinstance(command, dict):
            record.command = _json_safe(command)


def load_training_records(path: str | Path) -> list[TrainingStepRecord]:
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    records: dict[int, TrainingStepRecord] = {}
    for topic, payload in iter_log_payloads(log_path):
        _ingest_payload(records, source_log=log_path, topic=topic, payload=payload)
    return [records[key] for key in sorted(records)]


def _state_summary(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    joints = state.get("joints")
    return {
        "joint_names": joints.get("names") if isinstance(joints, dict) else None,
        "joint_positions": joints.get("positions") if isinstance(joints, dict) else None,
        "joint_velocities": joints.get("velocities") if isinstance(joints, dict) else None,
        "base_position_xyz": state.get("base_position_xyz"),
        "base_orientation_xyzw": state.get("base_orientation_xyzw"),
    }


def build_training_sample(
    record: TrainingStepRecord,
    *,
    next_record: TrainingStepRecord | None,
    observation_size: int,
    action_size: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if record.observation is None:
        return None, f"step {record.step_index}: missing policy observation"
    if record.action is None:
        return None, f"step {record.step_index}: missing policy action"
    if len(record.observation) != observation_size:
        return None, (
            f"step {record.step_index}: observation size {len(record.observation)} "
            f"!= expected {observation_size}"
        )
    if len(record.action) != action_size:
        return None, f"step {record.step_index}: action size {len(record.action)} != expected {action_size}"

    policy_command = None
    if isinstance(record.policy_debug, dict):
        command = record.policy_debug.get("command")
        if isinstance(command, list):
            policy_command = command

    next_state = None
    next_robot_time = None
    if next_record is not None and next_record.robot_time is not None and record.robot_time is not None:
        if float(next_record.robot_time) + 1e-9 >= float(record.robot_time):
            next_state = _state_summary(next_record.state)
            next_robot_time = next_record.robot_time

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "sample_type": "soridormi.policy_supervision.v1",
        "source_log": str(record.source_log),
        "step_index": int(record.step_index),
        "robot_time": record.robot_time,
        "next_robot_time": next_robot_time,
        "mode": record.mode,
        "backend": record.backend,
        "observation": record.observation,
        "action": record.action,
        "raw_action": record.raw_action,
        "policy_command": policy_command,
        "motor_command": record.command,
        "state": _state_summary(record.state),
        "next_state": next_state,
        "policy_debug": record.policy_debug,
    }, None


def export_training_dataset(
    logs: Iterable[str | Path],
    *,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    strict: bool = False,
) -> TrainingDatasetExportResult:
    log_paths = [Path(path) for path in logs]
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(output.suffix + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    sample_count = 0
    skipped_records = 0

    with output.open("w", encoding="utf-8") as f:
        for log_path in log_paths:
            try:
                records = load_training_records(log_path)
            except Exception as exc:
                message = f"{log_path}: {exc}"
                if strict:
                    errors.append(message)
                else:
                    warnings.append(message)
                continue

            for index, record in enumerate(records):
                next_record = records[index + 1] if index + 1 < len(records) else None
                sample, skip_reason = build_training_sample(
                    record,
                    next_record=next_record,
                    observation_size=observation_size,
                    action_size=action_size,
                )
                if sample is None:
                    skipped_records += 1
                    if skip_reason is not None:
                        if strict:
                            errors.append(f"{log_path}: {skip_reason}")
                        elif len(warnings) < 20:
                            warnings.append(f"{log_path}: {skip_reason}")
                    continue
                f.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
                sample_count += 1

    if sample_count == 0:
        errors.append("No training samples were exported. Logs must contain policy_observation and policy_action records.")

    dataset_sha = sha256_file(output) if output.exists() else None
    result = TrainingDatasetExportResult(
        ok=not errors,
        output_path=str(output),
        manifest_path=str(manifest),
        sample_count=sample_count,
        source_logs=[str(path) for path in log_paths],
        skipped_records=skipped_records,
        errors=errors,
        warnings=warnings,
        dataset_sha256=dataset_sha,
    )

    manifest_payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": "soridormi.policy_supervision.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_path": str(output),
        "sample_count": sample_count,
        "skipped_records": skipped_records,
        "source_logs": [str(path) for path in log_paths],
        "observation_size": observation_size,
        "action_size": action_size,
        "dataset_sha256": dataset_sha,
        "ok": result.ok,
        "errors": errors,
        "warnings": warnings,
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def print_export_summary(result: TrainingDatasetExportResult) -> None:
    print("Soridormi training dataset export")
    print("=================================")
    print(f"Output: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Samples: {result.sample_count}")
    print(f"Skipped records: {result.skipped_records}")
    if result.dataset_sha256:
        print(f"Dataset SHA256: {result.dataset_sha256}")
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
    parser = argparse.ArgumentParser(description="Export Soridormi runtime logs into a policy-training dataset JSONL.")
    parser.add_argument("logs", nargs="+", help="Runtime .jsonl/.mcap logs to export")
    parser.add_argument("--output", type=Path, default=None, help="Output dataset JSONL path")
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path")
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--strict", action="store_true", help="Treat skipped/malformed records as errors")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path("/data/training_datasets") / f"policy_supervision_{utc_stamp()}.jsonl"

    result = export_training_dataset(
        args.logs,
        output_path=output,
        manifest_path=args.manifest,
        observation_size=args.observation_size,
        action_size=args.action_size,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_export_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
