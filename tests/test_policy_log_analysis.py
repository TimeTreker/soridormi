from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.analyze_policy_log import analyze_policy_log


def _runtime_step(step: int, robot_time: float, action: list[float]) -> dict:
    return {
        "type": "runtime_step",
        "step_index": step,
        "time_wall_ns": 1_000_000_000 + step * 20_000_000,
        "robot_time": robot_time,
        "mode": "onnx_policy",
        "backend": "sim",
        "state": {
            "time": robot_time,
            "joints": {
                "names": ["j0", "j1"],
                "positions": [0.1 * step, -0.1 * step],
                "velocities": [0.2, -0.2],
                "torques": [0.0, 0.0],
            },
            "imu": {},
        },
        "command": {
            "names": ["j0", "j1"],
            "positions": [0.01 * step, -0.01 * step],
            "velocities": [0.0, 0.0],
            "kp": [10.0, 10.0],
            "kd": [0.5, 0.5],
            "torques": [0.0, 0.0],
        },
        "policy_action": action,
        "policy_debug": {
            "step_count": step,
            "robot_time": robot_time,
            "command": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "phase": [1.0, 0.0],
            "action_scale": 0.1,
            "max_motor_velocity": 3.0,
            "speed_limit_enabled": True,
        },
        "policy_observation_stats": {
            "shape": [1, 101],
            "dtype": "float32",
            "min": -1.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 0.1,
            "l2_norm": 1.5,
        },
    }


def test_analyze_policy_log_detects_robot_time_reset_in_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "runtime_test.jsonl"
    payloads = [
        _runtime_step(0, 0.00, [0.1, -0.2]),
        _runtime_step(1, 0.02, [0.2, -0.3]),
        _runtime_step(2, 0.04, [0.3, -0.4]),
        _runtime_step(3, 0.00, [0.1, -0.2]),
        _runtime_step(4, 0.02, [0.2, -0.3]),
    ]
    path.write_text("\n".join(json.dumps(payload) for payload in payloads), encoding="utf-8")

    summary = analyze_policy_log(path)

    assert summary["format"] == "jsonl"
    assert summary["records"] == 5
    assert summary["policy_records"] == 5
    assert summary["reset_cycles"]["count"] == 1
    assert summary["action"]["abs_max"] == 0.4
    assert summary["latest_command"] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert summary["latest_action_scale"] == 0.1
    assert any("Detected 1 robot-time reset" in item for item in summary["diagnosis"])


def test_analyze_policy_log_reports_missing_policy_debug(tmp_path: Path) -> None:
    path = tmp_path / "runtime_no_policy.jsonl"
    payload = {
        "type": "runtime_step",
        "step_index": 0,
        "time_wall_ns": 1_000_000_000,
        "robot_time": 0.0,
        "state": {"joints": {"positions": [0.0], "velocities": [0.0]}},
        "command": {"positions": [0.0]},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = analyze_policy_log(path)

    assert summary["policy_records"] == 0
    assert summary["diagnosis"] == [
        "No policy debug topics were found. Re-run with the M3.6 logger patch applied."
    ]
