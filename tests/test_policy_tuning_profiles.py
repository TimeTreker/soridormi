from __future__ import annotations

import pytest

from soridormi_runtime.policy_tuning_profiles import get_profile, list_profiles


def test_policy_tuning_profiles_include_safe_profile() -> None:
    names = [profile.name for profile in list_profiles()]

    assert "crawl_safe" in names
    assert "idle_debug" in names


def test_policy_tuning_profile_env_values_are_strings() -> None:
    profile = get_profile("crawl_safe")
    env = profile.env()

    assert env["SORIDORMI_RUNTIME_MODE"] == "onnx_policy"
    assert env["SORIDORMI_RUNTIME_LOG"] == "1"
    assert env["SORIDORMI_RUNTIME_LOG_PREFIX"] == "runtime_crawl_safe"
    assert env["SORIDORMI_COMMAND_X"] == "0.01"
    assert env["SORIDORMI_ACTION_SCALE"] == "0.1"


def test_unknown_policy_tuning_profile_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("not_a_profile")
