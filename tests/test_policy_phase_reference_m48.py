from __future__ import annotations

from pathlib import Path

import pytest

from soridormi_runtime.policy_command import _resolve_phase_period_steps
from soridormi_runtime.policy_profiles import PolicyProfile


def test_phase_resolver_fails_when_reference_required(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError):
        _resolve_phase_period_steps("auto", missing, require_reference_data=True)


def test_phase_resolver_can_fallback_when_reference_not_required(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pkl"
    period, source = _resolve_phase_period_steps("auto", missing, require_reference_data=False)
    assert period == 50
    assert source == "fallback_50"


def test_open_duck_forward_profile_requires_phase_reference_data() -> None:
    profile = PolicyProfile.load("configs/policies/open_duck_forward.yaml")
    env = profile.env()
    assert env["SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA"] == "1"
