from __future__ import annotations

import math
import pickle

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

def test_gait_phase_generator_loads_open_duck_period_from_reference_data(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "polynomial_coefficients.pkl"
    payload = {
        "0.0_0.0_0.0": {
            "period": 2.0,
            "fps": 30,
            "frame_offsets": [],
            "startend_double_support_ratio": 0.0,
            "coefficients": {},
        }
    }
    with reference.open("wb") as f:
        pickle.dump(payload, f)

    monkeypatch.setenv("SORIDORMI_PHASE_MODE", "step")
    monkeypatch.setenv("SORIDORMI_PHASE_PERIOD_STEPS", "auto")
    monkeypatch.setenv("SORIDORMI_PHASE_REFERENCE_DATA", str(reference))

    gen = GaitPhaseGenerator.from_env()

    assert gen.period_steps == 60
    assert gen.period_source == "reference_data"


def test_step_phase_matches_official_advance_before_observation() -> None:
    gen = GaitPhaseGenerator(mode="step", period_steps=4, step_increment=1.0)

    first = gen.advance_and_as_list()

    assert abs(first[0]) < 1e-6
    assert math.isclose(first[1], 1.0, abs_tol=1e-6)

