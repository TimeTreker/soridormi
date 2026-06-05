from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.dataset_coverage_report import analyze_dataset_coverage


def _sample(
    step: int,
    *,
    scenario_id: str = "flat_walk_varied_speed_v1",
    skill_id: str = "walk_velocity",
    terrain_type: str = "flat",
    vx: float = 0.1,
    vy: float = 0.0,
    yaw: float = 0.0,
    ramp_alpha: float = 1.0,
    stuck: bool = False,
    fall: bool = False,
    terminated: bool = False,
) -> dict:
    action = [0.1] * 14
    return {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.v1",
        "source_log": f"{scenario_id}:episode_0",
        "scenario_id": scenario_id,
        "scenario_status": "mujoco_registry_ready",
        "scenario_family": "locomotion_flat",
        "skill_id": skill_id,
        "scenario_dataset_tags": ["bc_stage_1", "velocity_conditioned"],
        "task_context": {"skill_family": "locomotion"},
        "environment_context": {"terrain_type": terrain_type, "obstacles": []},
        "command_space": {"vx_mps": [-0.03, 0.25], "vy_mps": [-0.03, 0.03], "yaw_radps": [-0.08, 0.08]},
        "desired_command": {"x_velocity": vx, "y_velocity": vy, "yaw_velocity": yaw},
        "applied_command": {"x_velocity": vx * ramp_alpha, "y_velocity": vy, "yaw_velocity": yaw},
        "policy_command": [vx * ramp_alpha, vy, yaw, 0.0, 0.0, 0.0, 0.0],
        "command_ramp_alpha": ramp_alpha,
        "step_index": step,
        "robot_time": step * 0.02,
        "observation": [float(step)] * 101,
        "action": action,
        "raw_action": action,
        "metrics": {"fall": fall, "terminated": terminated, "stuck_ratio": 1.0 if stuck else 0.0},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_analyze_dataset_coverage_reports_scenario_command_ramp_and_failure_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "teacher.jsonl"
    rows = [
        _sample(0, vx=0.0, ramp_alpha=0.25),
        _sample(1, vx=0.2, ramp_alpha=1.0),
        _sample(
            2,
            scenario_id="rough_ground_walk_v1",
            terrain_type="rough_ground",
            vx=0.12,
            ramp_alpha=0.75,
            stuck=True,
        ),
        _sample(
            3,
            scenario_id="rough_ground_walk_v1",
            terrain_type="rough_ground",
            vx=0.08,
            ramp_alpha=1.0,
            fall=True,
            terminated=True,
        ),
    ]
    _write_jsonl(dataset, rows)

    result = analyze_dataset_coverage(dataset, output_dir=tmp_path / "coverage", histogram_bins=4)

    assert result.ok
    assert result.sample_count == 4
    assert result.valid_sample_count == 4
    assert result.scenario_coverage.counts == {"flat_walk_varied_speed_v1": 2, "rough_ground_walk_v1": 2}
    assert result.skill_coverage.counts == {"walk_velocity": 4}
    assert result.terrain_coverage.counts == {"flat": 2, "rough_ground": 2}
    assert result.tag_coverage.counts["bc_stage_1"] == 4
    assert result.command_coverage["applied_command"]["vx_mps"].count == 4
    assert result.command_coverage["applied_command"]["vx_mps"].minimum == 0.0
    assert result.command_coverage["desired_command"]["vx_mps"].maximum == 0.2
    assert result.ramp_alpha_coverage.count == 4
    assert result.failure_coverage.stuck_count == 1
    assert result.failure_coverage.fall_count == 1
    assert result.failure_coverage.terminated_count == 1
    assert result.failure_coverage.failure_count == 2
    assert Path(result.summary_path).exists()
    assert Path(result.report_path).exists()

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary["coverage_type"] == "soridormi.policy_supervision.coverage.v1"
    assert summary["scenario_coverage"]["distinct_count"] == 2


def test_analyze_dataset_coverage_accepts_prepared_manifest_splits(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    train = prepared / "train.jsonl"
    val = prepared / "val.jsonl"
    test = prepared / "test.jsonl"
    _write_jsonl(train, [_sample(0, scenario_id="flat_walk_varied_speed_v1")])
    _write_jsonl(val, [_sample(1, scenario_id="curve_turn_walk_v1", yaw=0.12)])
    _write_jsonl(test, [_sample(2, scenario_id="rough_ground_walk_v1", terrain_type="rough_ground")])
    manifest = {
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"path": str(train), "sample_count": 1},
            "val": {"path": str(val), "sample_count": 1},
            "test": {"path": str(test), "sample_count": 1},
        },
    }
    manifest_path = prepared / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = analyze_dataset_coverage(manifest_path, output_dir=tmp_path / "coverage")

    assert result.ok
    assert result.valid_sample_count == 3
    assert result.split_coverage.counts == {"test": 1, "train": 1, "val": 1}
    assert result.scenario_coverage.distinct_count == 3
    assert result.command_coverage["applied_command"]["yaw_radps"].maximum == 0.12


def test_analyze_dataset_coverage_rejects_invalid_vectors(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    bad = _sample(0)
    bad["observation"] = [0.0] * 100
    _write_jsonl(dataset, [bad])

    result = analyze_dataset_coverage(dataset, output_dir=tmp_path / "coverage")

    assert not result.ok
    assert result.valid_sample_count == 0
    assert result.invalid_sample_count == 1
    assert any("observation size 100" in error for error in result.errors)


def test_report_dataset_coverage_script_documents_mujoco_collection_flow() -> None:
    import subprocess

    proc = subprocess.run(
        ["bash", "scripts/report_dataset_coverage.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "scenario_id" in proc.stdout
    assert "failure/stuck flags" in proc.stdout
    assert "do not" in proc.stdout.lower()
    assert "run_sim_server.sh" in proc.stdout
    assert "./scripts/collect_random_teacher_dataset.sh" in proc.stdout
    assert "--viewer" in proc.stdout
    assert "--scenario flat_walk_varied_speed_v1" in proc.stdout


def test_report_dataset_coverage_script_overrides_cuda_entrypoint_for_json_stdout() -> None:
    script = Path("scripts/report_dataset_coverage.sh").read_text(encoding="utf-8")

    assert "--entrypoint bash" in script
    assert "python -m soridormi_runtime.dataset_coverage_report" in script
