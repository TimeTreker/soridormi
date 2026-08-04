from __future__ import annotations

from soridormi_runtime.compare_official_soridormi_trace import TraceRecord
from soridormi_runtime.first_divergence_analyzer import analyze_first_divergence


def _record(
    step: int,
    *,
    observation: list[float] | None = None,
    action: list[float] | None = None,
    motor_targets: list[float] | None = None,
) -> TraceRecord:
    return TraceRecord(
        step_index=step,
        robot_time=step * 0.02,
        observation=observation if observation is not None else [0.0] * 101,
        action=action if action is not None else [0.0] * 14,
        raw_action=action if action is not None else [0.0] * 14,
        motor_targets=motor_targets if motor_targets is not None else [0.0] * 14,
        joint_positions=[0.0] * 14,
        joint_velocities=[0.0] * 14,
        contacts=[1.0, 1.0],
        phase=[1.0, 0.0],
        command=[0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        base_position_xyz=[step * 0.01, 0.0, 0.15],
    )


def test_first_divergence_reports_first_bad_step_and_segment() -> None:
    official = [_record(0), _record(1), _record(2)]
    soridormi = [_record(0), _record(1), _record(2)]
    soridormi[1].observation = list(soridormi[1].observation or [])
    soridormi[1].observation[3] = 0.25  # accelerometer_xyz segment starts at index 3

    summary = analyze_first_divergence(official, soridormi, steps=3, threshold=1e-4)

    assert summary["first_divergence"]["step"] == 1
    assert summary["first_divergence"]["name"] == "accelerometer_xyz"
    assert summary["candidates_at_first_divergent_step"][0]["name"] == "accelerometer_xyz"
    assert any("First threshold crossing" in item for item in summary["diagnosis"])


def test_history_diagnostics_identify_action_history_offset() -> None:
    actions = [[float(i)] * 14 for i in range(5)]
    records: list[TraceRecord] = []
    for i in range(5):
        obs = [0.0] * 101
        if i > 0:
            obs[41:55] = actions[i - 1]
        if i > 1:
            obs[55:69] = actions[i - 2]
        if i > 2:
            obs[69:83] = actions[i - 3]
        records.append(_record(i, observation=obs, action=actions[i]))

    summary = analyze_first_divergence(records, records, steps=5, threshold=1e-4)
    checks = {
        (item["trace"], item["observation_segment"]): item
        for item in summary["history_diagnostics"]
    }

    assert checks[("official", "last_action")]["best_offset"] == -1
    assert checks[("official", "last_last_action")]["best_offset"] == -2
    assert checks[("official", "last_last_last_action")]["best_offset"] == -3
    assert checks[("soridormi", "last_action")]["best_offset"] == -1
