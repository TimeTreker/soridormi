from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


POLICY_ACTION_TOPIC = "/soridormi/policy_action"
POLICY_DEBUG_TOPIC = "/soridormi/policy_debug"
POLICY_OBS_TOPIC = "/soridormi/policy_observation_stats"
ROBOT_STATE_TOPIC = "/soridormi/robot_state"
MOTOR_COMMAND_TOPIC = "/soridormi/motor_command"
RESET_TIME_DROP_SECONDS = 1e-6


@dataclass
class PolicyLogRecord:
    step_index: int
    wall_time_ns: int | None = None
    robot_time: float | None = None
    action: list[float] | None = None
    debug: dict[str, Any] | None = None
    observation_stats: dict[str, Any] | None = None
    motor_positions: list[float] | None = None
    joint_positions: list[float] | None = None
    joint_velocities: list[float] | None = None
    base_position_xyz: list[float] | None = None


@dataclass
class PolicyLogDataset:
    path: Path
    log_format: str
    topic_counts: Counter[str] = field(default_factory=Counter)
    records_by_step: dict[int, PolicyLogRecord] = field(default_factory=dict)
    first_wall_time_ns: int | None = None
    last_wall_time_ns: int | None = None

    @property
    def records(self) -> list[PolicyLogRecord]:
        return [self.records_by_step[key] for key in sorted(self.records_by_step)]


def _coerce_float_list(value: Any) -> list[float] | None:
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


def _step_index(payload: dict[str, Any]) -> int | None:
    raw = payload.get("step_index")
    if raw is None:
        debug = payload.get("debug")
        if isinstance(debug, dict):
            raw = debug.get("step_count")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _record(dataset: PolicyLogDataset, payload: dict[str, Any]) -> PolicyLogRecord | None:
    step = _step_index(payload)
    if step is None:
        return None
    record = dataset.records_by_step.get(step)
    if record is None:
        record = PolicyLogRecord(step_index=step)
        dataset.records_by_step[step] = record

    if "time_wall_ns" in payload:
        try:
            record.wall_time_ns = int(payload["time_wall_ns"])
        except (TypeError, ValueError):
            pass
    if "robot_time" in payload:
        try:
            record.robot_time = float(payload["robot_time"])
        except (TypeError, ValueError):
            pass
    return record


def _update_wall_range(dataset: PolicyLogDataset, payload: dict[str, Any]) -> None:
    if "time_wall_ns" not in payload:
        return
    try:
        wall = int(payload["time_wall_ns"])
    except (TypeError, ValueError):
        return
    if dataset.first_wall_time_ns is None or wall < dataset.first_wall_time_ns:
        dataset.first_wall_time_ns = wall
    if dataset.last_wall_time_ns is None or wall > dataset.last_wall_time_ns:
        dataset.last_wall_time_ns = wall


def _ingest_payload(dataset: PolicyLogDataset, topic: str, payload: dict[str, Any]) -> None:
    dataset.topic_counts[topic] += 1
    _update_wall_range(dataset, payload)

    record = _record(dataset, payload)
    if record is None:
        return

    if topic == POLICY_ACTION_TOPIC or "policy_action" in payload or "action" in payload:
        action = payload.get("action", payload.get("policy_action"))
        record.action = _coerce_float_list(action)

    if topic == POLICY_DEBUG_TOPIC or "policy_debug" in payload or "debug" in payload:
        debug = payload.get("debug", payload.get("policy_debug"))
        if isinstance(debug, dict):
            record.debug = dict(debug)
            if record.robot_time is None and "robot_time" in debug:
                try:
                    record.robot_time = float(debug["robot_time"])
                except (TypeError, ValueError):
                    pass

    if topic == POLICY_OBS_TOPIC or "policy_observation_stats" in payload or "stats" in payload:
        stats = payload.get("stats", payload.get("policy_observation_stats"))
        if isinstance(stats, dict):
            record.observation_stats = dict(stats)

    if topic == MOTOR_COMMAND_TOPIC or "command" in payload:
        command = payload.get("command")
        if isinstance(command, dict):
            record.motor_positions = _coerce_float_list(command.get("positions"))

    if topic == ROBOT_STATE_TOPIC or "state" in payload:
        state = payload.get("state")
        if isinstance(state, dict):
            joints = state.get("joints")
            if isinstance(joints, dict):
                record.joint_positions = _coerce_float_list(joints.get("positions"))
                record.joint_velocities = _coerce_float_list(joints.get("velocities"))
            record.base_position_xyz = _coerce_float_list(state.get("base_position_xyz"))


def _iter_jsonl_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            msg_type = str(payload.get("type", "runtime_step"))
            yield msg_type, payload


def _iter_mcap_payloads(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError(
            "Cannot analyze MCAP logs because the 'mcap' package is not installed. "
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


def load_policy_log(path: str | Path) -> PolicyLogDataset:
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    suffix = log_path.suffix.lower()
    if suffix == ".jsonl":
        log_format = "jsonl"
        iterator = _iter_jsonl_payloads(log_path)
    elif suffix == ".mcap":
        log_format = "mcap"
        iterator = _iter_mcap_payloads(log_path)
    else:
        raise ValueError(f"Unsupported log format: {suffix}. Expected .mcap or .jsonl")

    dataset = PolicyLogDataset(path=log_path, log_format=log_format)
    for topic, payload in iterator:
        _ingest_payload(dataset, topic, payload)
    return dataset


def _finite(values: list[float | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def _basic_stats(values: list[float]) -> dict[str, float | None]:
    values = _finite(values)
    if not values:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "count": len(values),
    }


def _abs_max(values: list[float]) -> float | None:
    values = _finite(values)
    if not values:
        return None
    return max(abs(value) for value in values)


def _all_values(records: list[PolicyLogRecord], attr: str) -> list[float]:
    values: list[float] = []
    for record in records:
        item = getattr(record, attr)
        if isinstance(item, list):
            values.extend(item)
    return values


def _debug_series(records: list[PolicyLogRecord], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if not isinstance(record.debug, dict):
            continue
        value = record.debug.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _last_debug(records: list[PolicyLogRecord]) -> dict[str, Any] | None:
    for record in reversed(records):
        if isinstance(record.debug, dict):
            return record.debug
    return None


def _base_displacement(records: list[PolicyLogRecord]) -> dict[str, Any]:
    base_records = [record for record in records if record.base_position_xyz is not None]
    if len(base_records) < 2:
        return {
            "available": False,
            "start_xyz": None,
            "end_xyz": None,
            "delta_xyz": None,
            "forward_x": None,
            "lateral_y": None,
            "vertical_z": None,
            "horizontal_distance": None,
        }
    start = base_records[0].base_position_xyz or [0.0, 0.0, 0.0]
    end = base_records[-1].base_position_xyz or [0.0, 0.0, 0.0]
    delta = [float(end[i] - start[i]) for i in range(3)]
    horizontal = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    return {
        "available": True,
        "start_xyz": [float(x) for x in start],
        "end_xyz": [float(x) for x in end],
        "delta_xyz": delta,
        "forward_x": delta[0],
        "lateral_y": delta[1],
        "vertical_z": delta[2],
        "horizontal_distance": horizontal,
    }


def detect_reset_cycles(records: list[PolicyLogRecord]) -> list[dict[str, Any]]:
    time_records = [record for record in records if record.robot_time is not None]
    if not time_records:
        return []

    cycles: list[dict[str, Any]] = []
    start = time_records[0]
    previous = time_records[0]

    for record in time_records[1:]:
        assert previous.robot_time is not None
        assert record.robot_time is not None
        if record.robot_time + RESET_TIME_DROP_SECONDS < previous.robot_time:
            cycles.append(_cycle_summary(start, previous))
            start = record
        previous = record

    cycles.append(_cycle_summary(start, previous))
    return cycles


def _cycle_summary(start: PolicyLogRecord, end: PolicyLogRecord) -> dict[str, Any]:
    start_robot_time = float(start.robot_time or 0.0)
    end_robot_time = float(end.robot_time or 0.0)
    wall_duration = None
    if start.wall_time_ns is not None and end.wall_time_ns is not None:
        wall_duration = (end.wall_time_ns - start.wall_time_ns) / 1_000_000_000.0

    return {
        "start_step": int(start.step_index),
        "end_step": int(end.step_index),
        "start_robot_time": start_robot_time,
        "end_robot_time": end_robot_time,
        "duration_robot_time": max(0.0, end_robot_time - start_robot_time),
        "duration_wall_seconds": wall_duration,
    }


def summarize_policy_log(dataset: PolicyLogDataset) -> dict[str, Any]:
    records = dataset.records
    policy_records = [
        record
        for record in records
        if record.action is not None or record.debug is not None or record.observation_stats is not None
    ]

    robot_times = _finite([record.robot_time for record in records])
    cycles = detect_reset_cycles(records)
    cycle_durations = [float(cycle["duration_robot_time"]) for cycle in cycles]
    last_debug = _last_debug(records) or {}

    actions = _all_values(policy_records, "action")
    motor_positions = _all_values(records, "motor_positions")
    joint_positions = _all_values(records, "joint_positions")
    joint_velocities = _all_values(records, "joint_velocities")
    base_displacement = _base_displacement(records)

    obs_min = _debug_series(policy_records, "observation_min")
    obs_max = _debug_series(policy_records, "observation_max")
    obs_l2 = _debug_series(policy_records, "observation_l2_norm")
    # M3.6 stores observation stats on a separate topic. Prefer those when present.
    for record in policy_records:
        if not isinstance(record.observation_stats, dict):
            continue
        for key, target in [("min", obs_min), ("max", obs_max), ("l2_norm", obs_l2)]:
            value = record.observation_stats.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))

    summary = {
        "path": str(dataset.path),
        "format": dataset.log_format,
        "topic_counts": dict(dataset.topic_counts),
        "records": len(records),
        "policy_records": len(policy_records),
        "wall_duration_seconds": (
            (dataset.last_wall_time_ns - dataset.first_wall_time_ns) / 1_000_000_000.0
            if dataset.first_wall_time_ns is not None and dataset.last_wall_time_ns is not None
            else None
        ),
        "robot_time": {
            "min": min(robot_times) if robot_times else None,
            "max": max(robot_times) if robot_times else None,
            "duration": (max(robot_times) - min(robot_times)) if robot_times else None,
        },
        "reset_cycles": {
            "count": max(0, len(cycles) - 1),
            "cycles": cycles[:20],
            "duration_stats": _basic_stats(cycle_durations),
        },
        "action": {
            **_basic_stats(actions),
            "abs_max": _abs_max(actions),
        },
        "motor_positions": {
            **_basic_stats(motor_positions),
            "abs_max": _abs_max(motor_positions),
        },
        "joint_positions": {
            **_basic_stats(joint_positions),
            "abs_max": _abs_max(joint_positions),
        },
        "joint_velocities": {
            **_basic_stats(joint_velocities),
            "abs_max": _abs_max(joint_velocities),
        },
        "observation": {
            "min": _basic_stats(obs_min),
            "max": _basic_stats(obs_max),
            "l2_norm": _basic_stats(obs_l2),
        },
        "base_displacement": base_displacement,
        "latest_command": last_debug.get("command"),
        "latest_phase": last_debug.get("phase"),
        "latest_action_scale": last_debug.get("action_scale"),
        "latest_max_motor_velocity": last_debug.get("max_motor_velocity"),
        "diagnosis": build_diagnosis(
            policy_records=policy_records,
            cycles=cycles,
            actions=actions,
            latest_debug=last_debug,
            base_displacement=base_displacement,
        ),
    }
    return summary


def build_diagnosis(
    *,
    policy_records: list[PolicyLogRecord],
    cycles: list[dict[str, Any]],
    actions: list[float],
    latest_debug: dict[str, Any],
    base_displacement: dict[str, Any] | None = None,
) -> list[str]:
    findings: list[str] = []

    if not policy_records:
        findings.append("No policy debug topics were found. Re-run with the M3.6 logger patch applied.")
        return findings

    reset_count = max(0, len(cycles) - 1)
    if reset_count:
        durations = [float(cycle["duration_robot_time"]) for cycle in cycles if cycle]
        shortest = min(durations) if durations else None
        longest = max(durations) if durations else None
        findings.append(
            f"Detected {reset_count} robot-time reset(s); cycle duration range is "
            f"{shortest:.3f}..{longest:.3f} robot seconds."
        )
    else:
        findings.append("No robot-time reset was detected in this log.")

    action_abs_max = _abs_max(actions)
    if action_abs_max is None:
        findings.append("No policy action vectors were logged.")
    elif action_abs_max < 0.02:
        findings.append("Policy actions are nearly zero; command/phase/observation may still be too neutral.")
    elif action_abs_max > 3.0:
        findings.append("Policy actions are very large; reduce action_scale and inspect observation normalization/order.")
    else:
        findings.append(f"Policy action magnitude looks nonzero: abs_max={action_abs_max:.3f}.")

    command = latest_debug.get("command")
    command_values = _coerce_float_list(command)
    if command_values is not None:
        linear = command_values[:3]
        if max(abs(x) for x in linear) < 1e-6:
            findings.append("Latest command velocity is zero; the policy is being asked to stand/idle.")
        else:
            findings.append(f"Latest command velocity terms are {linear}.")

    phase = _coerce_float_list(latest_debug.get("phase"))
    if phase is not None:
        norm = math.sqrt(sum(x * x for x in phase))
        if norm < 0.5:
            findings.append("Latest phase vector norm is small; verify SORIDORMI_PHASE_FREQUENCY is nonzero.")
        else:
            findings.append(f"Latest phase vector norm is {norm:.3f}.")

    displacement = base_displacement or {}
    if displacement.get("available"):
        forward = float(displacement.get("forward_x") or 0.0)
        lateral = float(displacement.get("lateral_y") or 0.0)
        findings.append(f"Base displacement: forward_x={forward:.3f} m, lateral_y={lateral:.3f} m.")
        if abs(forward) < 0.01:
            findings.append("Base forward displacement is near zero; inspect command sign/frame and action symmetry.")
        elif forward < -0.01:
            findings.append("Base moved backward; command sign or body/world frame convention may be inverted.")
    else:
        findings.append("Base displacement is unavailable; upgrade logs to include RobotState.base_position_xyz.")

    speed_limit = latest_debug.get("speed_limit_enabled")
    if speed_limit is False:
        findings.append("Motor speed limiting is disabled; enable it before aggressive walking tests.")

    return findings


def print_policy_analysis(summary: dict[str, Any]) -> None:
    print(f"Log: {summary['path']}")
    print(f"Format: {summary['format']}")
    print(f"Records: {summary['records']}  policy records: {summary['policy_records']}")

    wall = summary.get("wall_duration_seconds")
    if wall is not None:
        print(f"Wall duration: {wall:.3f} s")

    robot = summary["robot_time"]
    if robot.get("min") is not None and robot.get("max") is not None:
        print(
            f"Robot time: {robot['min']:.3f} .. {robot['max']:.3f} s  "
            f"duration={robot['duration']:.3f} s"
        )

    print("Topics/types:")
    for topic, count in sorted(summary["topic_counts"].items()):
        print(f"  {topic}: {count}")

    resets = summary["reset_cycles"]
    print(f"Reset cycles detected: {resets['count']}")
    if resets["cycles"]:
        print("First cycles:")
        for cycle in resets["cycles"][:5]:
            print(
                f"  steps {cycle['start_step']}..{cycle['end_step']}  "
                f"robot_duration={cycle['duration_robot_time']:.3f}s"
            )

    action = summary["action"]
    print(
        "Action stats: "
        f"count={action['count']} min={_fmt(action['min'])} max={_fmt(action['max'])} "
        f"mean={_fmt(action['mean'])} abs_max={_fmt(action['abs_max'])}"
    )

    motors = summary["motor_positions"]
    print(
        "Motor target stats: "
        f"count={motors['count']} min={_fmt(motors['min'])} max={_fmt(motors['max'])} "
        f"abs_max={_fmt(motors['abs_max'])}"
    )

    joints = summary["joint_positions"]
    print(
        "Joint position stats: "
        f"count={joints['count']} min={_fmt(joints['min'])} max={_fmt(joints['max'])} "
        f"abs_max={_fmt(joints['abs_max'])}"
    )

    displacement = summary.get("base_displacement", {})
    if displacement.get("available"):
        print(
            "Base displacement: "
            f"start={displacement['start_xyz']} end={displacement['end_xyz']} "
            f"delta={displacement['delta_xyz']} forward_x={_fmt(displacement['forward_x'])}m"
        )

    obs = summary["observation"]
    if obs["l2_norm"]["count"]:
        print(
            "Observation l2_norm: "
            f"min={_fmt(obs['l2_norm']['min'])} max={_fmt(obs['l2_norm']['max'])} "
            f"mean={_fmt(obs['l2_norm']['mean'])}"
        )

    if summary.get("latest_command") is not None:
        print(f"Latest command: {summary['latest_command']}")
    if summary.get("latest_phase") is not None:
        print(f"Latest phase: {summary['latest_phase']}")
    if summary.get("latest_action_scale") is not None:
        print(f"Latest action_scale: {summary['latest_action_scale']}")
    if summary.get("latest_max_motor_velocity") is not None:
        print(f"Latest max_motor_velocity: {summary['latest_max_motor_velocity']}")

    print("Diagnosis:")
    for item in summary["diagnosis"]:
        print(f"  - {item}")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def analyze_policy_log(path: str | Path) -> dict[str, Any]:
    return summarize_policy_log(load_policy_log(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Soridormi ONNX policy runtime logs.")
    parser.add_argument("log", type=Path, help="Path to a .mcap or .jsonl runtime log")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = analyze_policy_log(args.log)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_policy_analysis(summary)


if __name__ == "__main__":
    main()
