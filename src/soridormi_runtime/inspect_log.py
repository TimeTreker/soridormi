from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_jsonl(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    robot_times: list[float] = []
    steps = 0

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

    return {
        "format": "jsonl",
        "path": str(path),
        "messages": steps,
        "topics": dict(counts),
        "min_robot_time": min(robot_times) if robot_times else None,
        "max_robot_time": max(robot_times) if robot_times else None,
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
