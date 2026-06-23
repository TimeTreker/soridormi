from __future__ import annotations

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
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


def test_swing_clearance_reflex_adds_side_specific_lift() -> None:
    processor = ActionPostprocessor(
        ActionPostprocessorConfig(
            enabled=True,
            mode="swing_clearance_reflex",
            clearance_reflex_target_m=0.02,
            clearance_reflex_activation_margin_m=0.01,
            clearance_reflex_hip_pitch=0.04,
            clearance_reflex_knee=0.12,
            clearance_reflex_ankle=-0.03,
        )
    )
    action = np.zeros(14, dtype=np.float32)
    state = _state(feet_contacts=[0.0, 1.0], feet_position_xyz=[[0.0, 0.04, 0.01], [0.0, -0.04, 0.0]])

    out = processor.apply(action, JOINTS, state=state)

    assert out[2] == np.float32(0.04)
    assert out[3] == np.float32(0.12)
    assert out[4] == np.float32(-0.03)
    assert out[11] == 0.0
    assert out[12] == 0.0
    assert out[13] == 0.0
    assert processor.last_clearance_reflex["applied"] is True
    assert processor.last_clearance_reflex["feet"][0]["activation"] == 1.0


def test_swing_clearance_reflex_ignores_stance_and_high_swing_feet() -> None:
    processor = ActionPostprocessor(
        ActionPostprocessorConfig(
            enabled=True,
            mode="swing_clearance_reflex",
            clearance_reflex_target_m=0.02,
            clearance_reflex_knee=0.12,
        )
    )
    action = np.zeros(14, dtype=np.float32)
    state = _state(feet_contacts=[1.0, 0.0], feet_position_xyz=[[0.0, 0.04, 0.0], [0.0, -0.04, 0.03]])

    out = processor.apply(action, JOINTS, state=state)

    np.testing.assert_allclose(out, action)
    assert processor.last_clearance_reflex["applied"] is False


def test_swing_clearance_reflex_can_gain_existing_sagittal_action() -> None:
    processor = ActionPostprocessor(
        ActionPostprocessorConfig(
            enabled=True,
            mode="swing_clearance_reflex",
            clearance_reflex_target_m=0.02,
            clearance_reflex_activation_margin_m=0.01,
            clearance_reflex_sagittal_gain=1.5,
        )
    )
    action = np.zeros(14, dtype=np.float32)
    action[2] = -0.2
    action[3] = 0.4
    action[4] = -0.1
    action[11] = 0.2
    action[12] = 0.4
    action[13] = -0.1
    state = _state(
        feet_contacts=[0.0, 1.0],
        feet_position_xyz=[[0.0, 0.04, 0.01], [0.0, -0.04, 0.0]],
    )

    out = processor.apply(action, JOINTS, state=state)

    assert out[2] == np.float32(-0.3)
    assert out[3] == np.float32(0.6)
    assert out[4] == np.float32(-0.15)
    assert out[11] == action[11]
    assert out[12] == action[12]
    assert out[13] == action[13]


def _state(
    *,
    feet_contacts: list[float],
    feet_position_xyz: list[list[float]],
) -> RobotState:
    return RobotState(
        time=0.0,
        joints=JointState(
            names=JOINTS,
            positions=[0.0] * len(JOINTS),
            velocities=[0.0] * len(JOINTS),
            torques=[0.0] * len(JOINTS),
        ),
        imu=IMUState(),
        feet_contacts=feet_contacts,
        feet_position_xyz=feet_position_xyz,
    )
