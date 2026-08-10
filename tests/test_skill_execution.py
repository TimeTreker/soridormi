from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from soridormi_runtime.skill_execution import (
    FORWARD_WALK_SPEED_PRESETS_MPS,
    MIN_FORWARD_WALK_SPEED_MPS,
    SkillExecutionError,
    SkillExecutionRegistry,
)
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def test_registry_lists_manifest_backed_executable_skill_ids() -> None:
    registry = _registry()
    assert registry.executable_skill_ids() == (
        "acquire_and_deliver_resource",
        "blink_eyes",
        "bow",
        "curve_walk",
        "express_attention",
        "look_at_person",
        "look_direction",
        "neutral_head",
        "nod_yes",
        "shake_no",
        "sidestep",
        "stand_idle",
        "stop",
        "turn_in_place",
        "walk_forward",
        "walk_velocity",
    )




def test_resource_acquisition_mock_plan_keeps_semantic_parameters() -> None:
    plan = _registry().create_plan(
        "acquire_and_deliver_resource",
        {
            "resource": {
                "kind": "physical_object",
                "description": "a cup of water",
                "quantity": "one",
                "attributes": {},
            },
            "source": {"status": "unknown", "description": "", "bindings": {}},
            "recipient": {"description": "requester", "referent_id": None},
        },
    )

    assert plan.execution == "composite"
    assert [segment.label for segment in plan.commands] == [
        "resource_mock_approach",
        "resource_mock_acquire",
        "resource_mock_return",
        "resource_mock_handover",
    ]
    assert plan.commands[0].vx_mps > 0.0
    assert plan.commands[2].vx_mps < 0.0
    assert plan.parameters is not None
    assert plan.parameters["resource"]["description"] == "a cup of water"


def test_resource_acquisition_mock_rejects_information_kind() -> None:
    with pytest.raises(SkillExecutionError, match="physical_object only"):
        _registry().create_plan(
            "acquire_and_deliver_resource",
            {
                "resource": {"kind": "information", "description": "weather"},
                "source": {"status": "provider_resolved"},
                "recipient": {"description": "requester"},
            },
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


def test_walk_velocity_applies_minimum_forward_speed_floor() -> None:
    plan = _registry().create_plan(
        "walk_velocity",
        {"vx_mps": 0.02, "vy_mps": 0.0, "yaw_radps": 0.0, "duration_s": 3.0},
    )

    assert plan.commands[0].vx_mps == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)
    assert plan.parameters is not None
    assert plan.parameters["vx_mps"] == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)
    assert plan.parameters["requested_vx_mps"] == pytest.approx(0.02)
    assert plan.parameters["min_forward_speed_mps"] == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)


def test_walk_velocity_minimum_forward_speed_can_be_overridden_by_env() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "src"
    env["SORIDORMI_MIN_FORWARD_WALK_SPEED_MPS"] = "0.12"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.skill_execution",
            "walk_velocity",
            "--args",
            json.dumps({"vx_mps": 0.02, "vy_mps": 0.0, "yaw_radps": 0.0, "duration_s": 3.0}),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(proc.stdout)
    assert payload["plan"]["commands"][0]["vx_mps"] == pytest.approx(0.12)
    assert payload["plan"]["parameters"]["min_forward_speed_mps"] == pytest.approx(0.12)


def test_walk_forward_uses_semantic_speed_presets() -> None:
    plan = _registry().create_plan(
        "walk_forward",
        {"speed": "quick", "duration_s": 3.0},
    )

    command = plan.commands[0]
    assert plan.execution == "skill_wrapper"
    assert command.vx_mps == pytest.approx(FORWARD_WALK_SPEED_PRESETS_MPS["quick"])
    assert command.vy_mps == 0.0
    assert command.yaw_radps == 0.0
    assert command.duration_s == pytest.approx(3.0)
    assert plan.parameters is not None
    assert plan.parameters["speed"] == "quick"
    assert plan.parameters["lowered_to_skill_id"] == "walk_velocity"


def test_walk_forward_slow_matches_minimum_forward_speed() -> None:
    plan = _registry().create_plan("walk_forward", {"speed": "slow"})

    assert plan.commands[0].vx_mps == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)


def test_curve_walk_applies_minimum_forward_speed_floor() -> None:
    plan = _registry().create_plan(
        "curve_walk",
        {"vx_mps": 0.03, "yaw_radps": 0.12, "duration_s": 3.0},
    )

    command = plan.commands[0]
    assert command.vx_mps == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)
    assert command.yaw_radps == pytest.approx(0.12)


def test_skill_parameter_defaults_are_applied() -> None:
    plan = _registry().create_plan("turn_in_place", {"yaw_radps": 0.15})
    command = plan.commands[0]
    assert command.vx_mps == 0.0
    assert command.vy_mps == 0.0
    assert command.yaw_radps == pytest.approx(0.15)
    assert command.duration_s == pytest.approx(2.0)


def test_blink_eyes_dry_run_uses_visual_expression_segments() -> None:
    plan = _registry().create_plan(
        "blink_eyes",
        {"count": 2, "closed_duration_s": 0.10, "open_duration_s": 0.20},
    )

    assert plan.commands == ()
    assert plan.keyframes == ()
    assert [segment.expression for segment in plan.visual_expressions] == [
        "eyes_open",
        "eyes_closed",
        "eyes_open",
        "eyes_closed",
        "eyes_open",
    ]
    assert plan.visual_expressions[1].duration_s == pytest.approx(0.10)
    assert plan.visual_expressions[2].duration_s == pytest.approx(0.20)
    assert plan.total_duration_s == pytest.approx(0.80)


def test_rejects_out_of_range_and_unknown_parameters() -> None:
    registry = _registry()
    with pytest.raises(SkillExecutionError, match="above max"):
        registry.create_plan("walk_velocity", {"vx_mps": 99.0})
    with pytest.raises(SkillExecutionError, match="unknown parameters"):
        registry.create_plan("walk_velocity", {"speed": 0.1})
    with pytest.raises(SkillExecutionError, match="not in enum"):
        registry.create_plan("walk_forward", {"speed": "run"})


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
    assert payload["plan"]["commands"][0]["vx_mps"] == pytest.approx(0.12)
    assert payload["plan"]["parameters"]["requested_vx_mps"] == pytest.approx(0.11)
    assert payload["plan"]["parameters"]["min_forward_speed_mps"] == pytest.approx(0.12)
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
