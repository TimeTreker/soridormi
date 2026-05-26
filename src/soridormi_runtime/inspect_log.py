from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


POLICY_TOPICS = {
    "/soridormi/policy_action",
    "/soridormi/policy_debug",
    "/soridormi/policy_observation_stats",
}


def _compact_policy_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep inspect output small while preserving high-value debug fields."""
    if "debug" in payload and isinstance(payload["debug"], dict):
        debug = payload["debug"]
        keys = [
            "step_count",
            "robot_time",
            "command",
            "phase",
            "action_min",
            "action_max",
            "action_mean",
            "action_std",
            "motor_target_min",
            "motor_target_max",
            "motor_target_mean",
            "action_scale",
            "max_motor_velocity",
            "speed_limit_enabled",
        ]
        return {key: debug.get(key) for key in keys if key in debug}

    if "stats" in payload and isinstance(payload["stats"], dict):
        stats = payload["stats"]
        keys = ["shape", "dtype", "min", "max", "mean", "std", "l2_norm"]
        return {key: stats.get(key) for key in keys if key in stats}

    if "action" in payload:
        action = payload["action"]
        if isinstance(action, list):
            return {
                "shape": [len(action)],
                "min": min(action) if action else None,
                "max": max(action) if action else None,
                "first_values": action[:5],
            }

    return payload


def summarize_jsonl(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    robot_times: list[float] = []
    steps = 0
    latest_policy_debug: dict[str, Any] | None = None
    latest_policy_action: dict[str, Any] | None = None
    latest_policy_observation_stats: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            msg_type = str(payload.get("type", "unknown"))
            counts[msg_type] += 1
            steps += 1
            if "robot_time" in payload:
                robot_times.append(float(payload["robot_time"]))

            if "policy_debug" in payload:
                counts["policy_debug"] += 1
                latest_policy_debug = dict(payload["policy_debug"])
            if "policy_action" in payload:
                counts["policy_action"] += 1
                latest_policy_action = _compact_policy_snapshot({"action": payload["policy_action"]})
            if "policy_observation_stats" in payload:
                counts["policy_observation_stats"] += 1
                latest_policy_observation_stats = dict(payload["policy_observation_stats"])

    return {
        "format": "jsonl",
        "path": str(path),
        "messages": steps,
        "topics": dict(counts),
        "min_robot_time": min(robot_times) if robot_times else None,
        "max_robot_time": max(robot_times) if robot_times else None,
        "latest_policy_debug": latest_policy_debug,
        "latest_policy_action": latest_policy_action,
        "latest_policy_observation_stats": latest_policy_observation_stats,
    }


def summarize_mcap(path: Path) -> dict[str, Any]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError(
            "Cannot inspect MCAP logs because the 'mcap' package is not installed. "
            "Run inside the rebuilt runtime container."
        ) from exc

    counts: Counter[str] = Counter()
    message_count = 0
    first_log_time: int | None = None
    last_log_time: int | None = None
    robot_times: list[float] = []
    latest_policy_debug: dict[str, Any] | None = None
    latest_policy_action: dict[str, Any] | None = None
    latest_policy_observation_stats: dict[str, Any] | None = None

    with path.open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            message_count += 1
            counts[channel.topic] += 1
            if first_log_time is None:
                first_log_time = int(message.log_time)
            last_log_time = int(message.log_time)

            if channel.message_encoding == "json":
                try:
                    payload = json.loads(message.data.decode("utf-8"))
                except Exception:
                    continue
                if "robot_time" in payload:
                    robot_times.append(float(payload["robot_time"]))
                if channel.topic == "/soridormi/policy_debug":
                    latest_policy_debug = _compact_policy_snapshot(payload)
                elif channel.topic == "/soridormi/policy_action":
                    latest_policy_action = _compact_policy_snapshot(payload)
                elif channel.topic == "/soridormi/policy_observation_stats":
                    latest_policy_observation_stats = _compact_policy_snapshot(payload)

    return {
        "format": "mcap",
        "path": str(path),
        "messages": message_count,
        "topics": dict(counts),
        "duration_wall_seconds": (
            (last_log_time - first_log_time) / 1_000_000_000.0
            if first_log_time is not None and last_log_time is not None
            else None
        ),
        "min_robot_time": min(robot_times) if robot_times else None,
        "max_robot_time": max(robot_times) if robot_times else None,
        "latest_policy_debug": latest_policy_debug,
        "latest_policy_action": latest_policy_action,
        "latest_policy_observation_stats": latest_policy_observation_stats,
    }


def summarize_log(path: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    suffix = log_path.suffix.lower()
    if suffix == ".jsonl":
        return summarize_jsonl(log_path)
    if suffix == ".mcap":
        return summarize_mcap(log_path)
    raise ValueError(f"Unsupported log format: {log_path.suffix}. Expected .mcap or .jsonl")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Log: {summary['path']}")
    print(f"Format: {summary['format']}")
    print(f"Messages: {summary['messages']}")

    if summary.get("duration_wall_seconds") is not None:
        print(f"Wall duration: {summary['duration_wall_seconds']:.3f} s")

    if summary.get("min_robot_time") is not None and summary.get("max_robot_time") is not None:
        robot_duration = summary["max_robot_time"] - summary["min_robot_time"]
        print(
            f"Robot time: {summary['min_robot_time']:.3f} .. "
            f"{summary['max_robot_time']:.3f} s  duration={robot_duration:.3f} s"
        )

    print("Topics/types:")
    for topic, count in sorted(summary["topics"].items()):
        print(f"  {topic}: {count}")

    if summary.get("latest_policy_debug") is not None:
        print("Latest policy debug:")
        print(json.dumps(summary["latest_policy_debug"], indent=2, sort_keys=True))

    if summary.get("latest_policy_action") is not None:
        print("Latest policy action:")
        print(json.dumps(summary["latest_policy_action"], indent=2, sort_keys=True))

    if summary.get("latest_policy_observation_stats") is not None:
        print("Latest policy observation stats:")
        print(json.dumps(summary["latest_policy_observation_stats"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Soridormi runtime logs.")
    parser.add_argument("log", type=Path, help="Path to a .mcap or .jsonl log file")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    args = parser.parse_args()

    summary = summarize_log(args.log)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
