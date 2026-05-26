from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.compare_policy_logs import compare_policy_logs, guess_profile_from_path


def _record(step: int, robot_time: float, *, action_abs: float, profile_scale: float) -> dict:
    return {
        "type": "runtime_step",
        "step_index": step,
        "time_wall_ns": 1_000_000_000 + step * 20_000_000,
        "robot_time": robot_time,
        "state": {
            "time": robot_time,
            "joints": {
                "names": ["j0"],
                "positions": [robot_time],
                "velocities": [0.0],
                "torques": [0.0],
            },
            "imu": {},
        },
        "command": {
            "names": ["j0"],
            "positions": [0.0],
            "velocities": [0.0],
            "kp": [10.0],
            "kd": [0.5],
            "torques": [0.0],
        },
        "policy_action": [action_abs, -action_abs],
        "policy_debug": {
            "step_count": step,
            "robot_time": robot_time,
            "command": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "phase": [1.0, 0.0],
            "action_scale": profile_scale,
            "max_motor_velocity": 3.0,
            "speed_limit_enabled": True,
        },
        "policy_observation_stats": {"l2_norm": 1.0},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_guess_profile_from_profile_prefixed_log_name() -> None:
    assert guess_profile_from_path("runtime_crawl_safe_20260526_060102.mcap") == "crawl_safe"
    assert guess_profile_from_path("runtime_20260526_060102.mcap") == "unknown"


def test_compare_policy_logs_ranks_longer_cycle_first(tmp_path: Path) -> None:
    short_log = tmp_path / "runtime_crawl_safe_20260526_060102.jsonl"
    long_log = tmp_path / "runtime_crawl_very_safe_20260526_060103.jsonl"

    _write_jsonl(
        short_log,
        [
            _record(0, 0.00, action_abs=0.1, profile_scale=0.1),
            _record(1, 0.02, action_abs=0.1, profile_scale=0.1),
            _record(2, 0.00, action_abs=0.1, profile_scale=0.1),
        ],
    )
    _write_jsonl(
        long_log,
        [
            _record(0, 0.00, action_abs=0.05, profile_scale=0.05),
            _record(1, 0.02, action_abs=0.05, profile_scale=0.05),
            _record(2, 0.04, action_abs=0.05, profile_scale=0.05),
            _record(3, 0.06, action_abs=0.05, profile_scale=0.05),
        ],
    )

    rows = compare_policy_logs([short_log, long_log])

    assert rows[0].profile == "crawl_very_safe"
    assert rows[0].mean_cycle_seconds == 0.06
    assert rows[0].action_abs_max == 0.05
    assert rows[1].profile == "crawl_safe"
