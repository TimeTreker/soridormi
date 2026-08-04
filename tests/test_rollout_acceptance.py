from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.rollout_acceptance import (
    RolloutAcceptanceThresholds,
    evaluate_rollout_acceptance,
    render_rollout_acceptance_report,
    write_rollout_acceptance_outputs,
)


def _write_rollout_jsonl(path: Path, *, steps: int = 5, forward_per_step: float = 0.02) -> None:
    lines = []
    for step in range(steps):
        robot_time = step * 0.02
        base_x = step * forward_per_step
        lines.append(
            {
                "type": "/soridormi/policy_action",
                "step_index": step,
                "time_wall_ns": 1_000_000_000 + step * 20_000_000,
                "robot_time": robot_time,
                "action": [0.1] * 14,
            }
        )
        lines.append(
            {
                "type": "/soridormi/policy_debug",
                "step_index": step,
                "time_wall_ns": 1_000_001_000 + step * 20_000_000,
                "robot_time": robot_time,
                "debug": {
                    "step_count": step,
                    "command": [0.15, 0.0, 0.0],
                    "phase": [1.0, 0.0],
                    "speed_limit_enabled": True,
                },
            }
        )
        lines.append(
            {
                "type": "/soridormi/robot_state",
                "step_index": step,
                "time_wall_ns": 1_000_002_000 + step * 20_000_000,
                "robot_time": robot_time,
                "state": {
                    "joints": {
                        "positions": [0.1, -0.2],
                        "velocities": [0.0, 0.0],
                    },
                    "base_position_xyz": [base_x, 0.001 * step, 0.2],
                },
            }
        )
    path.write_text("\n".join(json.dumps(item) for item in lines) + "\n", encoding="utf-8")


def test_rollout_acceptance_passes_and_writes_artifacts(tmp_path: Path) -> None:
    log = tmp_path / "rollout.jsonl"
    _write_rollout_jsonl(log, steps=6)

    result = evaluate_rollout_acceptance(
        log,
        thresholds=RolloutAcceptanceThresholds(
            min_policy_records=6,
            min_robot_duration=0.08,
            min_forward_x=0.05,
            max_lateral_abs=0.02,
        ),
        profile_name="candidate",
    )

    assert result.ok
    assert result.profile_name == "candidate"
    assert result.summary["policy_records"] == 6
    assert result.summary["base_displacement"]["forward_x"] > 0.05

    output_dir = tmp_path / "acceptance"
    write_rollout_acceptance_outputs(result, output_dir)
    assert (output_dir / "rollout_acceptance.json").exists()
    report = (output_dir / "rollout_acceptance_report.md").read_text(encoding="utf-8")
    assert "Result: PASS" in report
    assert "candidate" in report


def test_rollout_acceptance_fails_thresholds(tmp_path: Path) -> None:
    log = tmp_path / "rollout.jsonl"
    _write_rollout_jsonl(log, steps=3, forward_per_step=0.0)

    result = evaluate_rollout_acceptance(
        log,
        thresholds=RolloutAcceptanceThresholds(
            min_policy_records=5,
            min_robot_duration=1.0,
            min_forward_x=0.01,
            max_action_abs=0.05,
        ),
    )

    assert not result.ok
    joined = "\n".join(result.errors)
    assert "policy_records" in joined
    assert "duration" in joined
    assert "forward_x" in joined
    assert "action abs_max" in joined
    assert "FAIL" in render_rollout_acceptance_report(result)
