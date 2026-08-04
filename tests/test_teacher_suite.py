from __future__ import annotations

import json
from pathlib import Path

import yaml

from soridormi_runtime.teacher_suite import build_teacher_suite


def _base_profile(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "description": "base profile",
                "model": {
                    "path": "/models/policy.onnx",
                    "input_name": "obs",
                    "output_name": "continuous_actions",
                    "input_shape": [1, 101],
                    "output_shape": [1, 14],
                    "input_type": "tensor(float)",
                    "output_type": "tensor(float)",
                },
                "runtime": {"mode": "onnx_policy", "backend": "sim", "control_hz": 50, "reset_at_start": True, "sync_step": True},
                "command": {"x": 0.15, "y": 0.0, "yaw": 0.0},
                "logging": {"enabled": True, "format": "mcap", "every_n": 1, "prefix": "base"},
            },
            sort_keys=False,
        )
    )


def test_build_teacher_suite_generates_command_profiles(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    _base_profile(base)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "teacher_test",
                "base_profile": str(base),
                "default_steps": 123,
                "scenarios": [
                    {"name": "turn_left", "command": {"x": 0.0, "y": 0.0, "yaw": 0.35}, "tags": ["turn"]},
                    {"name": "fast_forward", "steps": 456, "command": {"x": 0.2, "yaw": 0.0}, "tags": ["fast"]},
                ],
            },
            sort_keys=False,
        )
    )

    result = build_teacher_suite(suite, output_dir=tmp_path / "out")

    assert result.ok
    assert len(result.scenarios) == 2
    assert result.scenarios[0].steps == 123
    assert result.scenarios[1].steps == 456
    assert result.scenarios[0].command["yaw"] == 0.35

    first_profile = Path(result.scenarios[0].profile_path)
    if str(first_profile).startswith("/data/"):
        first_profile = tmp_path / "out" / "profiles" / Path(result.scenarios[0].profile_path).name
    payload = yaml.safe_load(first_profile.read_text())
    assert payload["command"]["yaw"] == 0.35
    assert payload["logging"]["prefix"].startswith("teacher_teacher_test_turn_left")

    manifest = json.loads((tmp_path / "out" / "teacher_suite_manifest.json").read_text())
    assert manifest["suite_name"] == "teacher_test"
    assert manifest["scenarios"][1]["name"] == "fast_forward"


def test_build_teacher_suite_rejects_unknown_command_key(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    _base_profile(base)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "base_profile": str(base),
                "scenarios": [{"name": "bad", "command": {"torque": 1.0}}],
            }
        )
    )

    try:
        build_teacher_suite(suite, output_dir=tmp_path / "out")
    except ValueError as exc:
        assert "unsupported command" in str(exc)
    else:
        raise AssertionError("expected ValueError")
