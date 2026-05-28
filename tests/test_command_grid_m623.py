from __future__ import annotations

import json
from pathlib import Path

import yaml

from soridormi_runtime.command_grid import (
    build_candidate_command_grid,
    summarize_command_grid_comparisons,
)


def _write_profile(path: Path, *, name: str) -> None:
    payload = {
        "name": name,
        "model": {"path": "/tmp/model.onnx", "input_shape": [1, 101], "output_shape": [1, 14]},
        "command": {"x": 0.15, "y": 0.0, "yaw": 0.0},
        "logging": {"prefix": f"policy_{name}"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_build_candidate_grid_clones_candidate_profile_with_teacher_commands(tmp_path: Path) -> None:
    candidate_profile = tmp_path / "candidate.yaml"
    _write_profile(candidate_profile, name="student")

    teacher_manifest = tmp_path / "teacher_suite_manifest.json"
    teacher_manifest.write_text(
        json.dumps(
            {
                "suite_name": "teacher_suite",
                "scenarios": [
                    {
                        "name": "turn_left",
                        "profile_path": "/data/teacher/turn_left.yaml",
                        "steps": 123,
                        "command": {"x": 0.0, "y": 0.0, "yaw": 0.35},
                        "tags": ["turn", "left"],
                    },
                    {
                        "name": "fast_forward",
                        "profile_path": "/data/teacher/fast_forward.yaml",
                        "steps": 456,
                        "command": {"x": 0.2, "y": 0.0, "yaw": 0.0},
                        "tags": ["walk", "fast"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_candidate_command_grid(
        teacher_manifest,
        candidate_profile,
        output_dir=tmp_path / "grid",
        force=True,
    )

    assert result.ok
    assert len(result.scenarios) == 2
    first = result.scenarios[0]
    assert first.name == "turn_left"
    assert first.command["yaw"] == 0.35
    assert first.steps == 123

    generated_profile = Path(first.candidate_profile_path.replace("/data/", "data/"))
    # The temp output is not under repo data/, so use the actual result path too.
    generated_profile = tmp_path / "grid" / "candidate_profiles" / "student_turn_left.yaml"
    payload = yaml.safe_load(generated_profile.read_text(encoding="utf-8"))
    assert payload["name"] == "student_turn_left"
    assert payload["command"]["yaw"] == 0.35
    assert payload["logging"]["prefix"] == "grid_student_turn_left"
    assert payload["command_grid"]["scenario"] == "turn_left"


def test_summarize_command_grid_comparisons(tmp_path: Path) -> None:
    comparisons = tmp_path / "comparisons"
    good = comparisons / "turn_left"
    bad = comparisons / "fast_forward"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)

    (good / "rollout_comparison.json").write_text(
        json.dumps(
            {
                "ok": True,
                "reference_log": "/data/logs/teacher_turn_left.mcap",
                "candidate_log": "/data/logs/student_turn_left.mcap",
                "candidate": {"reset_count": 0, "lateral_abs": 0.02, "action_abs_max": 1.0},
                "comparison": {"forward_ratio": 0.9, "forward_speed_ratio": 0.9},
                "errors": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (bad / "rollout_comparison.json").write_text(
        json.dumps(
            {
                "ok": False,
                "reference_log": "/data/logs/teacher_fast_forward.mcap",
                "candidate_log": "/data/logs/student_fast_forward.mcap",
                "candidate": {"reset_count": 1, "lateral_abs": 0.5, "action_abs_max": 2.0},
                "comparison": {"forward_ratio": 0.2, "forward_speed_ratio": 0.2},
                "errors": ["candidate reset count 1 exceeds limit 0"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_command_grid_comparisons(comparisons, output_dir=tmp_path / "out")

    assert not summary.ok
    assert summary.scenario_count == 2
    assert summary.passed_count == 1
    assert summary.failed_count == 1
    assert (tmp_path / "out" / "command_grid_summary.json").exists()
    report = (tmp_path / "out" / "command_grid_report.md").read_text(encoding="utf-8")
    assert "Result: FAIL" in report
    assert "fast_forward" in report
