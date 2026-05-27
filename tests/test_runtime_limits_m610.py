from __future__ import annotations

from soridormi_runtime.runtime_limits import RuntimeLimits, runtime_limit_reached, runtime_limits_from_env


def test_runtime_limits_from_env_parses_disabled_values(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_MAX_STEPS", "0")
    monkeypatch.setenv("SORIDORMI_MAX_SECONDS", "")

    limits = runtime_limits_from_env()

    assert limits.max_steps is None
    assert limits.max_seconds is None


def test_runtime_limits_from_env_parses_positive_values(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_MAX_STEPS", "25")
    monkeypatch.setenv("SORIDORMI_MAX_SECONDS", "3.5")

    limits = runtime_limits_from_env()

    assert limits.max_steps == 25
    assert limits.max_seconds == 3.5


def test_runtime_limit_reached_by_steps() -> None:
    limits = RuntimeLimits(max_steps=3)

    assert not runtime_limit_reached(completed_steps=2, started_at=10.0, now=10.1, limits=limits)
    assert runtime_limit_reached(completed_steps=3, started_at=10.0, now=10.1, limits=limits)


def test_runtime_limit_reached_by_seconds() -> None:
    limits = RuntimeLimits(max_seconds=1.5)

    assert not runtime_limit_reached(completed_steps=100, started_at=10.0, now=11.49, limits=limits)
    assert runtime_limit_reached(completed_steps=100, started_at=10.0, now=11.5, limits=limits)
