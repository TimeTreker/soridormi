from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.compare_official_soridormi_trace import load_soridormi_trace


def test_compare_loader_accepts_direct_replay_jsonl(tmp_path: Path) -> None:
    trace = tmp_path / "replay.trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "source": "soridormi_official_target_replay",
                "step_index": 0,
                "robot_time": 0.02,
                "motor_targets": [0.1] * 14,
                "joint_positions": [0.0] * 14,
                "joint_velocities": [0.0] * 14,
                "contacts": [1.0, 1.0],
                "base_position_xyz": [0.0, 0.0, 0.15],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_soridormi_trace(trace)

    assert len(records) == 1
    assert records[0].motor_targets == [0.1] * 14
    assert records[0].contacts == [1.0, 1.0]
    assert records[0].base_position_xyz == [0.0, 0.0, 0.15]
