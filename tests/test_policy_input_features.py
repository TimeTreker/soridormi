from __future__ import annotations

import numpy as np
import pytest

from soridormi_runtime.policy_input_features import (
    INPUT_MODE_CONTEXT_COMMAND_V1,
    INPUT_MODE_OBSERVATION,
    build_policy_input_batch,
    input_size_for,
    command_context_from_policy_command,
)


def test_policy_input_size_for_command_context_mode() -> None:
    assert input_size_for(INPUT_MODE_OBSERVATION, robot_observation_size=101) == 101
    assert input_size_for(INPUT_MODE_CONTEXT_COMMAND_V1, robot_observation_size=101) == 104


def test_command_context_policy_input_appends_vx_vy_yaw() -> None:
    observation = np.zeros((1, 101), dtype=np.float32)
    observation[0, 0] = 2.0

    policy_input = build_policy_input_batch(
        observation,
        input_mode=INPUT_MODE_CONTEXT_COMMAND_V1,
        command_vector=[0.12, -0.02, 0.05, 1.0, 2.0, 3.0, 4.0],
    )

    assert policy_input.shape == (1, 104)
    assert policy_input[0, 0] == 2.0
    np.testing.assert_allclose(policy_input[0, -3:], [0.12, -0.02, 0.05], atol=1e-7)


def test_command_context_requires_three_command_values() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        command_context_from_policy_command([0.1, 0.2])
