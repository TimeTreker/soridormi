from __future__ import annotations

from pathlib import Path

import pytest

from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_sim.mujoco_backend import env_flag


def test_open_duck_forward_profile_requests_official_sensor_contact_modes() -> None:
    profile = PolicyProfile.load("open_duck_forward")
    env = profile.env()

    assert env["SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE"] == "1"
    assert env["SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE"] == "1"
    assert env["SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE"] == "1"


def _find_compose_file() -> Path | None:
    """Find compose.sim.yaml when tests run from the host checkout.

    The runtime container mounts source/tests but not always the repository root
    compose file, so this test must skip instead of failing when compose is not
    visible inside the container.
    """

    candidates = [Path.cwd(), *Path.cwd().parents]
    for root in candidates:
        path = root / "compose.sim.yaml"
        if path.exists():
            return path
    return None


def test_compose_forwards_official_mujoco_flags() -> None:
    compose_path = _find_compose_file()
    if compose_path is None:
        pytest.skip("compose.sim.yaml is not mounted in this test environment")

    compose = compose_path.read_text(encoding="utf-8")

    assert "SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE" in compose
    assert "SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE" in compose
    assert "SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE" in compose


def test_env_flag_accepts_official_mode_truthy(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE", "1")
    assert env_flag("SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE", default=False) is True
