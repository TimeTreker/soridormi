from __future__ import annotations

from soridormi_sim.mujoco_viewer import env_flag


def test_env_flag_defaults_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("SORIDORMI_TEST_FLAG", raising=False)

    assert env_flag("SORIDORMI_TEST_FLAG", default=True) is True
    assert env_flag("SORIDORMI_TEST_FLAG", default=False) is False


def test_env_flag_true_values(monkeypatch) -> None:
    for value in ["1", "true", "TRUE", "yes", "on", "y"]:
        monkeypatch.setenv("SORIDORMI_TEST_FLAG", value)
        assert env_flag("SORIDORMI_TEST_FLAG") is True


def test_env_flag_false_values(monkeypatch) -> None:
    for value in ["0", "false", "no", "off", ""]:
        monkeypatch.setenv("SORIDORMI_TEST_FLAG", value)
        assert env_flag("SORIDORMI_TEST_FLAG") is False
