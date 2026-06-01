from __future__ import annotations

import json
import subprocess
import sys

import pytest

from soridormi_runtime.skill_execution import SkillExecutionError, SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def test_registry_lists_manifest_backed_executable_skill_ids() -> None:
    registry = _registry()
    assert registry.executable_skill_ids() == (
        "curve_walk",
        "sidestep",
        "stand_idle",
        "stop",
        "turn_in_place",
        "walk_velocity",
    )


def test_walk_velocity_dry_run_uses_validated_parameters() -> None:
    plan = _registry().create_plan(
        "walk_velocity",
        {"vx_mps": 0.12, "vy_mps": 0.01, "yaw_radps": -0.1, "duration_s": 3.0},
        profile="open_duck_forward",
    )
    assert plan.dry_run is True
    assert plan.skill_id == "walk_velocity"
    assert plan.profile == "open_duck_forward"
    assert len(plan.commands) == 1
    command = plan.commands[0]
    assert command.vx_mps == pytest.approx(0.12)
    assert command.vy_mps == pytest.approx(0.01)
    assert command.yaw_radps == pytest.approx(-0.1)
    assert command.duration_s == pytest.approx(3.0)


def test_skill_parameter_defaults_are_applied() -> None:
    plan = _registry().create_plan("turn_in_place", {"yaw_radps": 0.15})
    command = plan.commands[0]
    assert command.vx_mps == 0.0
    assert command.vy_mps == 0.0
    assert command.yaw_radps == pytest.approx(0.15)
    assert command.duration_s == pytest.approx(2.0)


def test_rejects_out_of_range_and_unknown_parameters() -> None:
    registry = _registry()
    with pytest.raises(SkillExecutionError, match="above max"):
        registry.create_plan("walk_velocity", {"vx_mps": 99.0})
    with pytest.raises(SkillExecutionError, match="unknown parameters"):
        registry.create_plan("walk_velocity", {"speed": 0.1})


def test_rejects_future_and_unsupported_skills() -> None:
    registry = _registry()
    with pytest.raises(SkillExecutionError, match="not executable yet"):
        registry.create_plan("step_over_obstacle", {})
    with pytest.raises(SkillExecutionError, match="not executable yet"):
        registry.create_plan("wave_hand", {})


def test_skill_dry_run_cli_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.skill_execution",
            "walk_velocity",
            "--args",
            json.dumps({"vx_mps": 0.11, "duration_s": 1.5}),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["plan"]["commands"][0]["vx_mps"] == pytest.approx(0.11)
    assert payload["plan"]["commands"][0]["duration_s"] == pytest.approx(1.5)


def test_run_skill_shell_wrapper_lists_skills() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_skill_dry_run.sh", "--list"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Soridormi executable dry-run skills" in proc.stdout
    assert "walk_velocity" in proc.stdout
