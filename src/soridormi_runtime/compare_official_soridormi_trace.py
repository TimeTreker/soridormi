from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


OBS_SEGMENTS: list[tuple[str, int, int]] = [
    ("gyro_xyz", 0, 3),
    ("accelerometer_xyz", 3, 6),
    ("command", 6, 13),
    ("joint_offsets", 13, 27),
    ("joint_velocities_scaled", 27, 41),
    ("last_action", 41, 55),
    ("last_last_action", 55, 69),
    ("last_last_last_action", 69, 83),
    ("motor_targets", 83, 97),
    ("feet_contacts", 97, 99),
    ("imitation_phase", 99, 101),
]


@dataclass
class TraceRecord:
    step_index: int
    robot_time: float | None = None
    observation: list[float] | None = None
    action: list[float] | None = None
    raw_action: list[float] | None = None
    motor_targets: list[float] | None = None
    joint_positions: list[float] | None = None
    joint_velocities: list[float] | None = None
    contacts: list[float] | None = None
    phase: list[float] | None = None
    command: list[float] | None = None
    base_position_xyz: list[float] | None = None
    default_actuator: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    try:
        for item in value:
            out.append(float(item))
    except (TypeError, ValueError):
        return None
    return out


def _step(payload: dict[str, Any], default: int) -> int:
    for key in ("step_index", "policy_step"):
        if key in payload:
            try:
                value = int(payload[key])
                return value - 1 if key == "policy_step" and value > 0 else value
            except (TypeError, ValueError):
                pass
    debug = payload.get("debug")
    if isinstance(debug, dict) and "step_count" in debug:
        try:
            return int(debug["step_count"])
        except (TypeError, ValueError):
            pass
    return default


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _read_mcap(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:  # pragma: no cover - exercised in runtime container
        raise RuntimeError(
            "MCAP trace comparison requires the 'mcap' package. Run this inside the runtime container."
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


def _resolve_official_trace(path: Path) -> Path:
    if path.suffix == ".jsonl":
        return path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Official summary is not a JSON object: {path}")
    trace = payload.get("trace_jsonl") or payload.get("latest_trace_jsonl")
    if not trace:
        raise ValueError(f"Official summary has no trace_jsonl field: {path}")
    trace_path = Path(str(trace))
    if not trace_path.exists():
        # Host paths may be mounted differently. Try a path relative to summary file.
        candidate = path.parent / trace_path.name
        if candidate.exists():
            return candidate
    return trace_path


def load_official_trace(path: str | Path) -> list[TraceRecord]:
    trace_path = _resolve_official_trace(Path(path))
    records: list[TraceRecord] = []
    for index, payload in enumerate(_read_jsonl(trace_path)):
        records.append(
            TraceRecord(
                step_index=_step(payload, index),
                robot_time=_safe_float(payload.get("robot_time", payload.get("sim_time"))),
                observation=_float_list(payload.get("observation")),
                action=_float_list(payload.get("action")),
                raw_action=_float_list(payload.get("raw_action")),
                motor_targets=_float_list(payload.get("motor_targets")),
                joint_positions=_float_list(payload.get("joint_positions")),
                joint_velocities=_float_list(payload.get("joint_velocities")),
                contacts=_float_list(payload.get("contacts")),
                phase=_float_list(payload.get("phase", payload.get("imitation_phase"))),
                command=_float_list(payload.get("command")),
                base_position_xyz=_float_list(payload.get("base_position_xyz")),
                default_actuator=_float_list(payload.get("default_actuator")),
                metadata={"source": "official", "path": str(trace_path)},
            )
        )
    return sorted(records, key=lambda item: item.step_index)


def load_soridormi_trace(path: str | Path) -> list[TraceRecord]:
    log_path = Path(path)
    by_step: dict[int, TraceRecord] = {}

    if log_path.suffix.lower() == ".jsonl":
        iterator = ((str(payload.get("type", "runtime_step")), payload) for payload in _read_jsonl(log_path))
    elif log_path.suffix.lower() == ".mcap":
        iterator = _read_mcap(log_path)
    else:
        raise ValueError(f"Unsupported Soridormi log format: {log_path.suffix}")

    for default_index, (topic, payload) in enumerate(iterator):
        step = _step(payload, default_index)
        record = by_step.setdefault(step, TraceRecord(step_index=step, metadata={"source": "soridormi", "path": str(log_path)}))
        record.robot_time = record.robot_time if record.robot_time is not None else _safe_float(payload.get("robot_time"))

        # Direct trace JSONL records produced by simulator-side replay tools use
        # the same top-level fields as official traces instead of MCAP topics.
        # Accept them here so the existing comparison report can isolate whether
        # Soridormi's MuJoCo backend reproduces official dynamics when fed the
        # exact official motor targets.
        if any(
            key in payload
            for key in (
                "base_position_xyz",
                "motor_targets",
                "joint_positions",
                "joint_velocities",
                "contacts",
                "observation",
                "action",
            )
        ):
            record.observation = _float_list(payload.get("observation")) or record.observation
            record.action = _float_list(payload.get("action")) or record.action
            record.raw_action = _float_list(payload.get("raw_action")) or record.raw_action
            record.motor_targets = _float_list(payload.get("motor_targets")) or record.motor_targets
            record.joint_positions = _float_list(payload.get("joint_positions")) or record.joint_positions
            record.joint_velocities = _float_list(payload.get("joint_velocities")) or record.joint_velocities
            record.contacts = _float_list(payload.get("contacts")) or record.contacts
            record.phase = _float_list(payload.get("phase")) or record.phase
            record.command = _float_list(payload.get("command")) or record.command
            record.base_position_xyz = _float_list(payload.get("base_position_xyz")) or record.base_position_xyz
            record.default_actuator = _float_list(payload.get("default_actuator")) or record.default_actuator

        if topic == "/soridormi/robot_state" or "state" in payload:
            state = payload.get("state")
            if isinstance(state, dict):
                record.base_position_xyz = _float_list(state.get("base_position_xyz")) or record.base_position_xyz
                record.contacts = _float_list(state.get("feet_contacts")) or record.contacts
                joints = state.get("joints")
                if isinstance(joints, dict):
                    record.joint_positions = _float_list(joints.get("positions")) or record.joint_positions
                    record.joint_velocities = _float_list(joints.get("velocities")) or record.joint_velocities

        if topic == "/soridormi/motor_command" or "command" in payload:
            command = payload.get("command")
            if isinstance(command, dict):
                record.motor_targets = _float_list(command.get("positions")) or record.motor_targets

        if topic == "/soridormi/policy_raw_action" or "policy_raw_action" in payload:
            record.raw_action = _float_list(payload.get("action", payload.get("policy_raw_action"))) or record.raw_action

        if topic == "/soridormi/policy_action" or "policy_action" in payload:
            record.action = _float_list(payload.get("action", payload.get("policy_action"))) or record.action

        if topic == "/soridormi/policy_observation" or "policy_observation" in payload:
            record.observation = _float_list(payload.get("observation", payload.get("policy_observation"))) or record.observation

        if topic == "/soridormi/policy_debug" or "policy_debug" in payload or "debug" in payload:
            debug = payload.get("debug", payload.get("policy_debug"))
            if isinstance(debug, dict):
                record.phase = _float_list(debug.get("phase")) or record.phase
                record.command = _float_list(debug.get("command")) or record.command
                record.contacts = _float_list(debug.get("feet_contacts")) or record.contacts
                if record.robot_time is None:
                    record.robot_time = _safe_float(debug.get("robot_time"))
                record.metadata.setdefault("debug", debug)

    return [by_step[key] for key in sorted(by_step)]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _paired(a: list[TraceRecord], b: list[TraceRecord], steps: int) -> list[tuple[TraceRecord, TraceRecord]]:
    return list(zip(a[:steps], b[:steps]))


def _mae(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(float(a[i]) - float(b[i])) for i in range(n)) / n


def _max_abs_diff(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    return max(abs(float(a[i]) - float(b[i])) for i in range(n))


def _series_metric(pairs: list[tuple[TraceRecord, TraceRecord]], attr: str) -> dict[str, Any]:
    maes: list[float] = []
    maxes: list[float] = []
    count = 0
    for left, right in pairs:
        a = getattr(left, attr)
        b = getattr(right, attr)
        item_mae = _mae(a, b)
        item_max = _max_abs_diff(a, b)
        if item_mae is not None:
            maes.append(item_mae)
            count += 1
        if item_max is not None:
            maxes.append(item_max)
    return {
        "count": count,
        "mean_mae": sum(maes) / len(maes) if maes else None,
        "max_abs_diff": max(maxes) if maxes else None,
    }


def _observation_segment_metrics(pairs: list[tuple[TraceRecord, TraceRecord]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, start, stop in OBS_SEGMENTS:
        maes: list[float] = []
        maxes: list[float] = []
        for official, soridormi in pairs:
            if official.observation is None or soridormi.observation is None:
                continue
            a = official.observation[start:stop]
            b = soridormi.observation[start:stop]
            item_mae = _mae(a, b)
            item_max = _max_abs_diff(a, b)
            if item_mae is not None:
                maes.append(item_mae)
            if item_max is not None:
                maxes.append(item_max)
        out.append(
            {
                "name": name,
                "range": [start, stop],
                "count": len(maes),
                "mean_mae": sum(maes) / len(maes) if maes else None,
                "max_abs_diff": max(maxes) if maxes else None,
            }
        )
    return out


def _displacement(records: list[TraceRecord]) -> dict[str, Any]:
    values = [record.base_position_xyz for record in records if record.base_position_xyz is not None]
    if len(values) < 2:
        return {"available": False, "delta_xyz": None, "forward_x": None}
    start = values[0]
    end = values[-1]
    assert start is not None and end is not None
    delta = [float(end[i] - start[i]) for i in range(3)]
    return {"available": True, "start_xyz": start, "end_xyz": end, "delta_xyz": delta, "forward_x": delta[0]}


def compare_traces(official: list[TraceRecord], soridormi: list[TraceRecord], *, steps: int = 100) -> dict[str, Any]:
    pairs = _paired(official, soridormi, min(steps, len(official), len(soridormi)))
    segment_metrics = _observation_segment_metrics(pairs)
    sorted_segments = sorted(
        segment_metrics,
        key=lambda item: -1.0 if item["mean_mae"] is None else float(item["mean_mae"]),
        reverse=True,
    )
    result = {
        "steps_compared": len(pairs),
        "official_records": len(official),
        "soridormi_records": len(soridormi),
        "metrics": {
            "observation": _series_metric(pairs, "observation"),
            "action": _series_metric(pairs, "action"),
            "raw_action": _series_metric(pairs, "raw_action"),
            "motor_targets": _series_metric(pairs, "motor_targets"),
            "joint_positions": _series_metric(pairs, "joint_positions"),
            "joint_velocities": _series_metric(pairs, "joint_velocities"),
            "contacts": _series_metric(pairs, "contacts"),
            "phase": _series_metric(pairs, "phase"),
            "command": _series_metric(pairs, "command"),
        },
        "observation_segments": segment_metrics,
        "worst_observation_segments": sorted_segments[:5],
        "official_displacement": _displacement(official[: len(pairs)]),
        "soridormi_displacement": _displacement(soridormi[: len(pairs)]),
        "diagnosis": [],
    }
    result["diagnosis"] = build_diagnosis(result)
    return result


def build_diagnosis(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if summary["steps_compared"] == 0:
        return ["No comparable trace records were found."]

    obs = summary["metrics"]["observation"]
    if obs["count"] == 0:
        findings.append("Soridormi did not log raw policy observations. Re-run after M4.6 logging is applied.")
    else:
        findings.append(
            f"Observation mean MAE over compared steps: {obs['mean_mae']:.6f}; max_abs_diff={obs['max_abs_diff']:.6f}."
        )

    for segment in summary["worst_observation_segments"][:3]:
        if segment["mean_mae"] is not None:
            findings.append(
                f"Worst obs segment: {segment['name']} mean_mae={segment['mean_mae']:.6f} max_abs_diff={segment['max_abs_diff']:.6f}."
            )

    contacts = summary["metrics"]["contacts"]
    if contacts["count"] and contacts["max_abs_diff"] and contacts["max_abs_diff"] > 0.5:
        findings.append("Foot contact traces differ strongly; prioritize contact extraction/ordering.")

    phase = summary["metrics"]["phase"]
    if phase["count"] and phase["mean_mae"] and phase["mean_mae"] > 0.1:
        findings.append("Phase traces differ; prioritize reference-period/step-phase compatibility.")

    motors = summary["metrics"]["motor_targets"]
    if motors["count"] and motors["mean_mae"] and motors["mean_mae"] > 0.05:
        findings.append("Motor target traces differ; inspect default_actuator/bootstrap/action_scale/speed limit.")

    odx = summary.get("official_displacement", {}).get("forward_x")
    sdx = summary.get("soridormi_displacement", {}).get("forward_x")
    if odx is not None and sdx is not None:
        findings.append(f"Forward displacement over compared window: official={odx:.4f} m, soridormi={sdx:.4f} m.")
        if abs(float(sdx)) < 0.1 * max(abs(float(odx)), 1e-6):
            findings.append("Soridormi forward displacement is much smaller than official; port the highest-error obs/control segments first.")

    return findings


def print_comparison(summary: dict[str, Any]) -> None:
    print("Official vs Soridormi trace comparison")
    print("======================================")
    print(f"Steps compared: {summary['steps_compared']}")
    print(f"Official records: {summary['official_records']}")
    print(f"Soridormi records: {summary['soridormi_records']}")
    print("Metrics:")
    for name, metric in summary["metrics"].items():
        print(
            f"  {name}: count={metric['count']} "
            f"mean_mae={_fmt(metric['mean_mae'])} max_abs_diff={_fmt(metric['max_abs_diff'])}"
        )
    print("Worst observation segments:")
    for segment in summary["worst_observation_segments"]:
        print(
            f"  {segment['name']} {segment['range']}: "
            f"mean_mae={_fmt(segment['mean_mae'])} max_abs_diff={_fmt(segment['max_abs_diff'])}"
        )
    print("Displacement:")
    print(f"  official: {summary['official_displacement']}")
    print(f"  soridormi: {summary['soridormi_displacement']}")
    print("Diagnosis:")
    for item in summary["diagnosis"]:
        print(f"  - {item}")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official Open Duck and Soridormi policy traces.")
    parser.add_argument("--official", type=Path, required=True, help="Official trace JSONL or summary JSON")
    parser.add_argument("--soridormi", type=Path, required=True, help="Soridormi runtime .mcap or .jsonl log")
    parser.add_argument("--steps", type=int, default=100, help="Number of initial policy steps to compare")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = compare_traces(
        load_official_trace(args.official),
        load_soridormi_trace(args.soridormi),
        steps=max(1, int(args.steps)),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_comparison(summary)


if __name__ == "__main__":
    main()
