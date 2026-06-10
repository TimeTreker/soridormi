from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from soridormi_runtime.context_bc_dataset_prepare import prepare_context_bc_dataset
from soridormi_runtime.context_bc_prepared_gate import validate_prepared_context_dataset
from soridormi_runtime.context_bc_training_ready import (
    build_training_ready_report,
    write_markdown_report,
)


def _sample(scenario_id: str, rollout_id: str, step: int) -> dict:
    return {
        "sample_type": "soridormi.policy_supervision.context_v1",
        "schema_version": 1,
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "timestep": step,
        "step_index": step,
        "episode_index": int(rollout_id.rsplit("_", 1)[-1]),
        "skill_id": "walk_velocity",
        "robot_state": {"observation": [float(step)] * 101},
        "desired_command": {"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.0},
        "applied_command": {"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.0},
        "task_context": {"skill_id": "walk_velocity", "gait_style": "default_walk"},
        "environment_context": {"terrain_type": "flat", "obstacle_height_m": 0.0},
        "command_ramp_alpha": 1.0,
        "teacher_action": [0.01 * step] * 14,
        "failure_flags": {"fallen": False, "stuck": False, "terminated": False},
        "short_history": {
            "previous_action": [0.0] * 14,
            "previous_command": {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_radps": 0.0},
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _ready_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    context_jsonl = tmp_path / "context.jsonl"
    rows: list[dict] = []
    for rollout in range(6):
        for step in range(2):
            rows.append(
                _sample("flat_walk_varied_speed_v1", f"rollout_{rollout}", rollout * 10 + step)
            )
    _write_jsonl(context_jsonl, rows)

    scenario_gate_json = tmp_path / "scenario_gate" / "dataset_scenario_gate_summary.json"
    scenario_gate_json.parent.mkdir(parents=True, exist_ok=True)
    scenario_gate_json.write_text(
        json.dumps(
            {
                "ok": True,
                "gate_type": "soridormi.policy_supervision.scenario_gate.v1",
                "required_scenarios": ["flat_walk_varied_speed_v1"],
                "valid_sample_count": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    prepare = prepare_context_bc_dataset(
        [context_jsonl],
        output_dir=tmp_path / "prepared",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=3,
    )
    assert prepare.ok
    prepared_manifest = Path(prepare.manifest_path)

    prepared_gate = validate_prepared_context_dataset(
        prepared_manifest,
        require_scenarios=["flat_walk_varied_speed_v1"],
        min_train_samples=1,
        min_val_samples=1,
        min_test_samples=1,
    )
    assert prepared_gate.ok
    prepared_gate_json = tmp_path / "prepared_gate" / "prepared_context_gate_report.json"
    prepared_gate_json.parent.mkdir(parents=True, exist_ok=True)
    prepared_gate_json.write_text(json.dumps(asdict(prepared_gate), indent=2), encoding="utf-8")
    return prepared_manifest, scenario_gate_json, prepared_gate_json


def test_training_ready_report_accepts_gated_dataset(tmp_path: Path) -> None:
    prepared_manifest, scenario_gate_json, prepared_gate_json = _ready_inputs(tmp_path)

    result = build_training_ready_report(
        prepared_manifest,
        scenario_gate_report_path=scenario_gate_json,
        prepared_gate_report_path=prepared_gate_json,
        profile_name="context_stage1_test",
    )

    assert result.ok
    assert result.total_sample_count == 12
    assert result.split_sample_counts["train"] > 0
    assert result.split_sample_counts["val"] > 0
    assert result.split_sample_counts["test"] > 0
    assert result.scenario_gate_ok is True
    assert result.prepared_gate_ok is True
    assert result.file_hashes["prepared_manifest"].exists
    assert result.file_hashes["split_train"].sha256
    assert "--input-mode" in result.recommended_train_commands.neural_bc
    assert "context_stage1_test" in result.recommended_train_commands.neural_bc


def test_training_ready_report_fails_failed_gate(tmp_path: Path) -> None:
    prepared_manifest, scenario_gate_json, prepared_gate_json = _ready_inputs(tmp_path)
    payload = json.loads(scenario_gate_json.read_text(encoding="utf-8"))
    payload["ok"] = False
    scenario_gate_json.write_text(json.dumps(payload), encoding="utf-8")

    result = build_training_ready_report(
        prepared_manifest,
        scenario_gate_report_path=scenario_gate_json,
        prepared_gate_report_path=prepared_gate_json,
    )

    assert not result.ok
    assert any("scenario coverage gate ok is not true" in error for error in result.errors)


def test_training_ready_cli_json_and_markdown(tmp_path: Path) -> None:
    prepared_manifest, scenario_gate_json, prepared_gate_json = _ready_inputs(tmp_path)
    output_dir = tmp_path / "ready"

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.context_bc_training_ready",
            str(prepared_manifest),
            "--scenario-gate",
            str(scenario_gate_json),
            "--prepared-gate",
            str(prepared_gate_json),
            "--output-dir",
            str(output_dir),
            "--profile-name",
            "context_stage1_cli",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["recommended_train_commands"]["neural_bc"]
    assert (output_dir / "training_ready_manifest.json").exists()
    assert (output_dir / "training_ready_report.md").exists()


def test_training_ready_markdown_includes_train_commands(tmp_path: Path) -> None:
    prepared_manifest, scenario_gate_json, prepared_gate_json = _ready_inputs(tmp_path)
    result = build_training_ready_report(
        prepared_manifest,
        scenario_gate_report_path=scenario_gate_json,
        prepared_gate_report_path=prepared_gate_json,
    )
    report = tmp_path / "ready.md"

    write_markdown_report(result, report)

    text = report.read_text(encoding="utf-8")
    assert "./scripts/train_behavior_clone.sh" in text
    assert "./scripts/train_neural_behavior_clone.sh" in text


def test_training_ready_shell_help_mentions_gate_inputs() -> None:
    text = Path("scripts/build_context_bc_training_ready_report.sh").read_text(encoding="utf-8")
    assert "--scenario-gate PATH" in text
    assert "--prepared-gate PATH" in text


def test_context_bc_training_ready_pipeline_help_mentions_required_inputs() -> None:
    text = Path("scripts/run_context_bc_training_ready_pipeline.sh").read_text(encoding="utf-8")
    assert "--scenario-gate PATH" in text
    assert "--require-scenario SCENARIO" in text
    assert "--output-dir DIR" in text
    assert "--prepared-gate-dir DIR" in text
    assert "--training-ready-dir DIR" in text
