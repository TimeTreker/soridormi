from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.compare_official_soridormi_trace import (
    compare_traces,
    load_official_trace,
    load_soridormi_trace,
)
from soridormi_runtime.policy_profiles import PolicyProfile


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_load_official_trace_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "official.trace.jsonl"
    write_jsonl(
        path,
        [
            {
                "policy_step": 1,
                "sim_time": 0.02,
                "observation": [0.0] * 101,
                "action": [0.1] * 14,
                "motor_targets": [0.2] * 14,
                "contacts": [1.0, 1.0],
                "phase": [1.0, 0.0],
                "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "base_position_xyz": [0.0, 0.0, 0.2],
            }
        ],
    )

    records = load_official_trace(path)

    assert len(records) == 1
    assert records[0].step_index == 0
    assert records[0].observation == [0.0] * 101
    assert records[0].action == [0.1] * 14


def test_load_soridormi_jsonl_trace_with_policy_observation(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    write_jsonl(
        path,
        [
            {
                "type": "runtime_step",
                "step_index": 0,
                "robot_time": 0.02,
                "policy_observation": [0.01] * 101,
                "policy_action": [0.1] * 14,
                "policy_debug": {
                    "phase": [1.0, 0.0],
                    "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "feet_contacts": [1.0, 1.0],
                },
                "state": {
                    "base_position_xyz": [0.0, 0.0, 0.2],
                    "joints": {"positions": [0.0] * 14, "velocities": [0.0] * 14},
                },
                "command": {"positions": [0.2] * 14},
            }
        ],
    )

    records = load_soridormi_trace(path)

    assert len(records) == 1
    assert records[0].observation == [0.01] * 101
    assert records[0].action == [0.1] * 14
    assert records[0].motor_targets == [0.2] * 14
    assert records[0].contacts == [1.0, 1.0]


def test_compare_traces_reports_worst_observation_segment(tmp_path: Path) -> None:
    official_path = tmp_path / "official.trace.jsonl"
    soridormi_path = tmp_path / "runtime.jsonl"
    official_obs = [0.0] * 101
    soridormi_obs = [0.0] * 101
    # Command segment indexes 6:13 differ.
    for index in range(6, 13):
        soridormi_obs[index] = 1.0

    write_jsonl(
        official_path,
        [
            {
                "policy_step": 1,
                "sim_time": 0.02,
                "observation": official_obs,
                "action": [0.0] * 14,
                "motor_targets": [0.0] * 14,
                "contacts": [1.0, 1.0],
                "phase": [1.0, 0.0],
                "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "base_position_xyz": [0.0, 0.0, 0.2],
            }
        ],
    )
    write_jsonl(
        soridormi_path,
        [
            {
                "type": "runtime_step",
                "step_index": 0,
                "robot_time": 0.02,
                "policy_observation": soridormi_obs,
                "policy_action": [0.0] * 14,
                "policy_debug": {
                    "phase": [1.0, 0.0],
                    "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "feet_contacts": [1.0, 1.0],
                },
                "state": {"base_position_xyz": [0.0, 0.0, 0.2]},
                "command": {"positions": [0.0] * 14},
            }
        ],
    )

    summary = compare_traces(load_official_trace(official_path), load_soridormi_trace(soridormi_path))

    assert summary["steps_compared"] == 1
    assert summary["metrics"]["observation"]["count"] == 1
    assert summary["worst_observation_segments"][0]["name"] == "command"


def test_policy_profile_exports_reset_at_start() -> None:
    profile = PolicyProfile.load("open_duck_forward")

    assert profile.env()["SORIDORMI_RESET_AT_START"] == "1"
