from __future__ import annotations

import json
import subprocess
from pathlib import Path

from soridormi_runtime.context_bc_dataset_prepare import prepare_context_bc_dataset
from soridormi_runtime.context_bc_prepared_gate import validate_prepared_context_dataset


def _sample(scenario_id: str, rollout_id: str, step: int) -> dict:
    return {
        "sample_type": "soridormi.policy_supervision.context_v1",
        "schema_version": 1,
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "timestep": step,
        "step_index": step,
        "episode_index": int(rollout_id.rsplit("_", 1)[-1]) if rollout_id.rsplit("_", 1)[-1].isdigit() else 0,
        "skill_id": "walk_velocity",
        "robot_state": {"observation": [float(step)] * 101},
        "desired_command": {"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.02},
        "applied_command": {"vx_mps": 0.09, "vy_mps": 0.0, "yaw_radps": 0.02},
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


def _prepared_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "context.jsonl"
    rows: list[dict] = []
    for rollout in range(6):
        for step in range(2):
            rows.append(_sample("flat_walk_varied_speed_v1", f"rollout_{rollout}", rollout * 10 + step))
    _write_jsonl(dataset, rows)
    result = prepare_context_bc_dataset(
        [dataset],
        output_dir=tmp_path / "prepared",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=3,
    )
    assert result.ok
    return Path(result.manifest_path)


def test_prepared_gate_accepts_valid_manifest(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)

    result = validate_prepared_context_dataset(
        manifest,
        require_scenarios=["flat_walk_varied_speed_v1"],
        min_train_samples=1,
        min_val_samples=1,
        min_test_samples=1,
    )

    assert result.ok
    assert result.total_sample_count == 12
    assert result.scenario_counts == {"flat_walk_varied_speed_v1": 12}
    assert result.split_group_counts["train"] > 0
    assert result.leaked_groups == {}


def test_prepared_gate_fails_empty_split(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    val_path = Path(manifest_payload["splits"]["val"]["path"])
    val_path.write_text("", encoding="utf-8")

    result = validate_prepared_context_dataset(manifest, require_scenarios=["flat_walk_varied_speed_v1"])

    assert not result.ok
    assert result.splits["val"].sample_count == 0
    assert any("split 'val'" in error and "minimum" in error for error in result.errors)
    assert any("sha256" in error for error in result.errors)


def test_prepared_gate_fails_rollout_group_leakage(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train_path = Path(payload["splits"]["train"]["path"])
    val_path = Path(payload["splits"]["val"]["path"])
    first_train_line = train_path.read_text(encoding="utf-8").splitlines()[0]
    with val_path.open("a", encoding="utf-8") as handle:
        handle.write(first_train_line + "\n")

    result = validate_prepared_context_dataset(manifest, require_scenarios=["flat_walk_varied_speed_v1"])

    assert not result.ok
    assert result.leaked_groups
    assert any("multiple splits" in error for error in result.errors)


def test_prepared_gate_fails_missing_required_scenario(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)

    result = validate_prepared_context_dataset(
        manifest,
        require_scenarios=["flat_walk_varied_speed_v1", "curve_turn_walk_v1"],
        min_samples_per_required_scenario=1,
    )

    assert not result.ok
    assert any("curve_turn_walk_v1" in error for error in result.errors)


def test_prepared_gate_cli_json_and_report(tmp_path: Path) -> None:
    manifest = _prepared_dataset(tmp_path)
    report_dir = tmp_path / "gate"

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.context_bc_prepared_gate",
            str(manifest),
            "--require-scenario",
            "flat_walk_varied_speed_v1",
            "--output-dir",
            str(report_dir),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["total_sample_count"] == 12
    assert (report_dir / "prepared_context_gate_report.json").exists()
    assert (report_dir / "prepared_context_gate_report.md").exists()


def test_prepared_gate_shell_help_mentions_leakage() -> None:
    text = Path("scripts/gate_context_bc_prepared_dataset.sh").read_text(encoding="utf-8")
    assert "rollout groups across splits" in text
    assert "--require-ready-locomotion" in text
