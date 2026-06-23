from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from soridormi_runtime.skill_execution import SkillExecutionRegistry, plan_shell_exports
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


def test_plan_shell_exports_bind_velocity_command_overrides() -> None:
    registry = SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))
    plan = registry.create_plan(
        "walk_velocity",
        {"vx_mps": 0.12, "vy_mps": 0.01, "yaw_radps": -0.05, "duration_s": 2.5},
        profile="open_duck_forward",
    )
    text = plan_shell_exports(plan)
    assert "export SORIDORMI_COMMAND_X_OVERRIDE=0.12" in text
    assert "export SORIDORMI_COMMAND_Y_OVERRIDE=0.01" in text
    assert "export SORIDORMI_COMMAND_YAW_OVERRIDE=-0.05" in text
    assert "export SORIDORMI_SKILL_DURATION_SECONDS=2.5" in text
    assert "export SORIDORMI_SKILL_ID=walk_velocity" in text
    assert "SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE" not in text


def test_skill_execution_cli_shell_env() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.skill_execution",
            "turn_in_place",
            "--args",
            json.dumps({"yaw_radps": 0.15, "duration_s": 1.25}),
            "--shell-env",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "SORIDORMI_COMMAND_X_OVERRIDE=0" in proc.stdout
    assert "SORIDORMI_COMMAND_YAW_OVERRIDE=0.15" in proc.stdout
    assert "SORIDORMI_SKILL_DURATION_SECONDS=1.25" in proc.stdout


def test_run_skill_in_sim_script_help_and_dry_run_only() -> None:
    help_proc = subprocess.run(
        ["bash", "scripts/run_skill_in_sim.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "--dry-run-only" in help_proc.stdout
    assert "--profile PROFILE" in help_proc.stdout
    assert "--control-hz HZ" in help_proc.stdout
    assert "run_sim_server.sh --backend mujoco" in help_proc.stdout

    dry_proc = subprocess.run(
        [
            "bash",
            "scripts/run_skill_in_sim.sh",
            "walk_velocity",
            "--args",
            json.dumps({"vx_mps": 0.1, "duration_s": 1.25}),
            "--control-hz",
            "40",
            "--dry-run-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "Command overrides: x=0.1" in dry_proc.stdout
    assert "Control Hz: 40" in dry_proc.stdout
    assert "Rollout steps: 50 (derived from skill duration)" in dry_proc.stdout
    assert "Wall-clock seconds cutoff: disabled" in dry_proc.stdout
    assert "Dry-run only; not launching runtime." in dry_proc.stdout


def test_run_skill_in_sim_steps_override_and_wall_clock_cutoff_are_explicit() -> None:
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_skill_in_sim.sh",
            "curve_walk",
            "--args",
            json.dumps({"vx_mps": 0.1, "yaw_radps": 0.12, "duration_s": 3.0}),
            "--steps",
            "123",
            "--seconds",
            "30",
            "--dry-run-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert "Rollout steps: 123 (user override)" in proc.stdout
    assert "Wall-clock seconds cutoff: 30 (user override)" in proc.stdout


def test_runtime_command_override_envs_are_forwarded_and_reapplied() -> None:
    compose = Path("compose.sim.yaml")
    if not compose.exists():
        pytest.skip("compose.sim.yaml is not available in this test environment")
    compose_text = compose.read_text(encoding="utf-8")
    assert "SORIDORMI_COMMAND_X_OVERRIDE" in compose_text
    assert "SORIDORMI_COMMAND_Y_OVERRIDE" in compose_text
    assert "SORIDORMI_COMMAND_YAW_OVERRIDE" in compose_text
    assert "SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE" in compose_text

    experiment_text = Path("scripts/run_policy_experiment.sh").read_text(encoding="utf-8")
    assert "COMMAND_X_OVERRIDE" in experiment_text
    assert "export SORIDORMI_COMMAND_X=\"${COMMAND_X_OVERRIDE}\"" in experiment_text
    assert "export SORIDORMI_COMMAND_YAW=\"${COMMAND_YAW_OVERRIDE}\"" in experiment_text
