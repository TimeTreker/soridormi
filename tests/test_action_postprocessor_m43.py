from __future__ import annotations

import numpy as np

from soridormi_runtime.action_postprocessor import (
    ActionPostprocessor,
    ActionPostprocessorConfig,
)


JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def test_action_postprocessor_identity_when_disabled() -> None:
    processor = ActionPostprocessor(ActionPostprocessorConfig(enabled=False, leg_gain=2.0, head_gain=0.0))
    action = np.ones(14, dtype=np.float32)

    out = processor.apply(action, JOINTS)

    np.testing.assert_allclose(out, action)
    assert processor.last_joint_gains["left_knee"] == 2.0
    assert processor.last_joint_gains["head_yaw"] == 0.0


def test_action_postprocessor_boosts_legs_and_damps_head() -> None:
    processor = ActionPostprocessor(
        ActionPostprocessorConfig(
            enabled=True,
            leg_gain=2.0,
            head_gain=0.0,
            knee_gain=1.5,
            clip_abs=3.0,
        )
    )
    action = np.ones(14, dtype=np.float32)

    out = processor.apply(action, JOINTS)

    assert out[3] == 3.0  # left_knee: 1 * leg_gain 2 * knee_gain 1.5
    assert out[6] == 0.0  # head_pitch damped
    assert out[12] == 3.0  # right_knee
    assert processor.last_output_stats["abs_max"] == 3.0


def test_action_postprocessor_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_ACTION_POSTPROCESS", "1")
    monkeypatch.setenv("SORIDORMI_LEG_ACTION_GAIN", "1.8")
    monkeypatch.setenv("SORIDORMI_HEAD_ACTION_GAIN", "0.0")
    monkeypatch.setenv("SORIDORMI_ACTION_CLIP_ABS", "2.0")

    processor = ActionPostprocessor.from_env()

    assert processor.config.enabled is True
    assert processor.config.leg_gain == 1.8
    assert processor.config.head_gain == 0.0
    assert processor.config.clip_abs == 2.0
