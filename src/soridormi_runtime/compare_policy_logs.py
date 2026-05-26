from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.analyze_policy_log import analyze_policy_log


@dataclass(frozen=True)
class PolicyLogComparisonRow:
    path: str
    profile: str
    policy_records: int
    reset_count: int
    best_cycle_seconds: float | None
    mean_cycle_seconds: float | None
    robot_duration_seconds: float | None
    action_abs_max: float | None
    action_mean: float | None
    latest_action_scale: float | None
    latest_max_motor_velocity: float | None
    latest_command: list[float] | None

    def score_tuple(self) -> tuple[float, float, int]:
        mean_cycle = self.mean_cycle_seconds or 0.0
        best_cycle = self.best_cycle_seconds or 0.0
        return (mean_cycle, best_cycle, -self.reset_count)


def expand_log_paths(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        text = str(item)
        matches = [Path(path) for path in glob.glob(text)]
        if not matches:
            matches = [Path(text)]
        for path in matches:
            if path.is_dir():
                paths.extend(sorted(path.glob("*.mcap")))
                paths.extend(sorted(path.glob("*.jsonl")))
            elif path.suffix.lower() in {".mcap", ".jsonl"}:
                paths.append(path)
    # Deduplicate while preserving sorted-ish input order.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def guess_profile_from_path(path: str | Path) -> str:
    name = Path(path).name
    # Examples: runtime_crawl_safe_20260526_052320.mcap, runtime_20260526_052320.mcap
    match = re.match(r"runtime_(?P<profile>[a-zA-Z0-9_]+)_\d{8}_\d{6}\.(mcap|jsonl)$", name)
    if match:
        return match.group("profile")
    return "unknown"


def comparison_row(path: str | Path) -> PolicyLogComparisonRow:
    summary = analyze_policy_log(path)
    cycles = summary["reset_cycles"]["cycles"]
    durations = [float(cycle["duration_robot_time"]) for cycle in cycles]
    duration_stats = summary["reset_cycles"].get("duration_stats", {})

    latest_command = summary.get("latest_command")
    if latest_command is not None:
        latest_command = [float(value) for value in latest_command]

    return PolicyLogComparisonRow(
        path=str(path),
        profile=guess_profile_from_path(path),
        policy_records=int(summary["policy_records"]),
        reset_count=int(summary["reset_cycles"]["count"]),
        best_cycle_seconds=max(durations) if durations else None,
        mean_cycle_seconds=duration_stats.get("mean"),
        robot_duration_seconds=summary["robot_time"].get("duration"),
        action_abs_max=summary["action"].get("abs_max"),
        action_mean=summary["action"].get("mean"),
        latest_action_scale=summary.get("latest_action_scale"),
        latest_max_motor_velocity=summary.get("latest_max_motor_velocity"),
        latest_command=latest_command,
    )


def compare_policy_logs(paths: Iterable[str | Path]) -> list[PolicyLogComparisonRow]:
    rows = [comparison_row(path) for path in expand_log_paths(paths)]
    return sorted(rows, key=lambda row: row.score_tuple(), reverse=True)


def rows_to_json(rows: list[PolicyLogComparisonRow]) -> str:
    return json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True)


def print_comparison(rows: list[PolicyLogComparisonRow]) -> None:
    if not rows:
        print("No .mcap or .jsonl logs found.")
        return

    print("Soridormi policy log comparison")
    print("================================")
    print(
        f"{'rank':>4}  {'profile':<18} {'resets':>6} {'best_s':>8} {'mean_s':>8} "
        f"{'policy':>7} {'abs_act':>8} {'scale':>7} {'max_vel':>7}  log"
    )
    for index, row in enumerate(rows, start=1):
        print(
            f"{index:>4}  {row.profile:<18} {row.reset_count:>6} "
            f"{_fmt(row.best_cycle_seconds):>8} {_fmt(row.mean_cycle_seconds):>8} "
            f"{row.policy_records:>7} {_fmt(row.action_abs_max):>8} "
            f"{_fmt(row.latest_action_scale):>7} {_fmt(row.latest_max_motor_velocity):>7}  "
            f"{row.path}"
        )

    best = rows[0]
    print()
    print("Best current run by mean reset-cycle duration:")
    print(f"  profile: {best.profile}")
    print(f"  log: {best.path}")
    print(f"  mean_cycle_seconds: {_fmt(best.mean_cycle_seconds)}")
    print(f"  best_cycle_seconds: {_fmt(best.best_cycle_seconds)}")
    print(f"  resets: {best.reset_count}")
    if best.latest_command is not None:
        print(f"  latest_command: {best.latest_command}")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Soridormi policy runtime logs.")
    parser.add_argument("logs", nargs="+", help="Log files, directories, or globs")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    rows = compare_policy_logs(args.logs)
    if args.json:
        print(rows_to_json(rows))
    else:
        print_comparison(rows)


if __name__ == "__main__":
    main()
