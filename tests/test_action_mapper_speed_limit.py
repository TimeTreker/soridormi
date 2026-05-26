from __future__ import annotations

import numpy as np

from soridormi_runtime.action_mapper import ActionMapperConfig, PolicyActionMapper


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


def test_action_mapper_applies_motor_speed_limit() -> None:
    mapper = PolicyActionMapper(
        ActionMapperConfig(
            joint_names=JOINTS,
            default_positions_by_name={name: 0.0 for name in JOINTS},
            action_scale=1.0,
            max_motor_velocity=1.0,
            speed_limit_enabled=True,
            clip_to_limits=False,
        )
    )

    action = np.ones(14, dtype=np.float32)
    targets = mapper.action_to_targets(action, dt=0.1)

    assert all(abs(value - 0.1) < 1e-6 for value in targets.values())

    targets = mapper.action_to_targets(action, dt=0.1)
    assert all(abs(value - 0.2) < 1e-6 for value in targets.values())


def test_action_mapper_can_disable_speed_limit() -> None:
    mapper = PolicyActionMapper(
        ActionMapperConfig(
            joint_names=JOINTS,
            default_positions_by_name={name: 0.0 for name in JOINTS},
            action_scale=1.0,
            max_motor_velocity=1.0,
            speed_limit_enabled=False,
            clip_to_limits=False,
        )
    )

    targets = mapper.action_to_targets(np.ones(14, dtype=np.float32), dt=0.1)
    assert all(abs(value - 1.0) < 1e-6 for value in targets.values())
