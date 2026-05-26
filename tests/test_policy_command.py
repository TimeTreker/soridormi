from __future__ import annotations

import math

from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand


def test_policy_command_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_COMMAND_X", "0.05")
    monkeypatch.setenv("SORIDORMI_COMMAND_Y", "-0.01")
    monkeypatch.setenv("SORIDORMI_COMMAND_YAW", "0.2")
    monkeypatch.setenv("SORIDORMI_HEAD_YAW", "0.3")

    command = PolicyCommand.from_env()

    assert command.as_list() == [0.05, -0.01, 0.2, 0.0, 0.0, 0.3, 0.0]


def test_gait_phase_generator_vector() -> None:
    gen = GaitPhaseGenerator(frequency_hz=1.0, start_time=10.0)

    v0 = gen.vector(now=10.0)
    v_quarter = gen.vector(now=10.25)

    assert v0.shape == (2,)
    assert math.isclose(float(v0[0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(v0[1]), 0.0, abs_tol=1e-6)
    assert abs(float(v_quarter[0])) < 1e-6
    assert math.isclose(float(v_quarter[1]), 1.0, abs_tol=1e-6)
