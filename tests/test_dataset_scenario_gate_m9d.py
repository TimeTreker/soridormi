from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from soridormi_runtime.dataset_scenario_gate import evaluate_dataset_scenario_gate


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
    failure: bool = False,
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
        "scenario_dataset_tags": ["bc_stage_1", "velocity_conditioned", "ramp_commands"],
        "task_context": {"skill_family": "locomotion"},
        "environment_context": {"terrain_type": terrain_type, "obstacles": []},
        "command_space": {"vx_mps": [-0.03, 0.25], "vy_mps": [-0.03, 0.03], "yaw_radps": [-0.08, 0.08]},
        "desired_command": {"x_velocity": vx, "y_velocity": vy, "yaw_velocity": yaw},
        "applied_command": {"x_velocity": vx, "y_velocity": vy, "yaw_velocity": yaw},
        "policy_command": [vx, vy, yaw, 0.0, 0.0, 0.0, 0.0],
        "command_ramp_alpha": ramp_alpha,
        "command_ramp_name": "smooth_start" if ramp_alpha < 1.0 else "hold",
        "step_index": step,
        "robot_time": step * 0.02,
        "observation": [float(step)] * 101,
        "action": action,
        "raw_action": action,
        "metrics": {
            "fall": False,
            "terminated": False,
            "stuck_ratio": 0.0,
            "failure": failure,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_dataset_scenario_gate_passes_structured_flat_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "flat.jsonl"
    rows = [
        _sample(0, vx=-0.03, vy=-0.03, yaw=-0.08, ramp_alpha=0.25),
        _sample(1, vx=0.10, vy=0.00, yaw=0.00, ramp_alpha=0.75),
        _sample(2, vx=0.25, vy=0.03, yaw=0.08, ramp_alpha=1.0),
    ]
    _write_jsonl(dataset, rows)

    result = evaluate_dataset_scenario_gate(
        dataset,
        output_dir=tmp_path / "gate",
        required_scenarios=["flat_walk_varied_speed_v1"],
        min_samples_per_scenario=3,
        min_command_range_fraction=0.5,
        max_failure_ratio=0.1,
    )

    assert result.ok
    assert result.valid_sample_count == 3
    assert result.required_scenarios == ["flat_walk_varied_speed_v1"]
    entry = result.scenario_results[0]
    assert entry.ok
    assert entry.sample_count == 3
    assert entry.ramp_alpha_count == 3
    assert entry.task_context_count == 3
    assert entry.environment_context_count == 3
    assert entry.failure_flag_count == 3
    assert entry.command_stats["vx_mps"].covered_fraction == 1.0
    assert Path(result.summary_path).exists()
    assert Path(result.report_path).exists()

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary["gate_type"] == "soridormi.policy_supervision.scenario_gate.v1"
    assert summary["scenario_results"][0]["ok"] is True


def test_dataset_scenario_gate_fails_missing_required_scenario(tmp_path: Path) -> None:
    dataset = tmp_path / "flat.jsonl"
    _write_jsonl(dataset, [_sample(0, vx=0.1), _sample(1, vx=0.2)])

    result = evaluate_dataset_scenario_gate(
        dataset,
        output_dir=tmp_path / "gate",
        required_scenarios=["rough_ground_walk_v1"],
        min_command_range_fraction=0.0,
    )

    assert not result.ok
    rough = next(item for item in result.scenario_results if item.scenario_id == "rough_ground_walk_v1")
    assert rough.required
    assert rough.sample_count == 0
    assert any("min_samples_per_scenario" in error for error in rough.errors)


def test_dataset_scenario_gate_fails_narrow_command_range(tmp_path: Path) -> None:
    dataset = tmp_path / "narrow.jsonl"
    _write_jsonl(dataset, [_sample(0, vx=0.1, vy=0.0, yaw=0.0), _sample(1, vx=0.1, vy=0.0, yaw=0.0)])

    result = evaluate_dataset_scenario_gate(
        dataset,
        output_dir=tmp_path / "gate",
        required_scenarios=["flat_walk_varied_speed_v1"],
        min_samples_per_scenario=2,
        min_command_range_fraction=0.25,
    )

    assert not result.ok
    entry = result.scenario_results[0]
    assert entry.command_stats["vx_mps"].covered_fraction == 0.0
    assert any("range_fraction" in error for error in entry.errors)


def test_dataset_scenario_gate_accepts_prepared_manifest_splits(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    train = prepared / "train.jsonl"
    val = prepared / "val.jsonl"
    _write_jsonl(train, [_sample(0, vx=-0.03, vy=-0.03, yaw=-0.08), _sample(1, vx=0.25, vy=0.03, yaw=0.08)])
    _write_jsonl(val, [_sample(2, vx=0.10, vy=0.0, yaw=0.0)])
    manifest = {
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"path": str(train), "sample_count": 2},
            "val": {"path": str(val), "sample_count": 1},
        },
    }
    manifest_path = prepared / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_dataset_scenario_gate(
        manifest_path,
        output_dir=tmp_path / "gate",
        required_scenarios=["flat_walk_varied_speed_v1"],
        min_samples_per_scenario=3,
        min_command_range_fraction=0.5,
    )

    assert result.ok
    assert result.scenario_results[0].split_counts == {"train": 2, "val": 1}


def test_gate_dataset_scenario_coverage_wrapper_runs_inside_runtime_container(tmp_path: Path) -> None:
    capture_path = tmp_path / "docker_args.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${SORIDORMI_FAKE_DOCKER_CAPTURE}\"\n"
        "exit 0\n"
    )
    fake_docker.chmod(0o755)

    env_path = Path(".env")
    old_env = env_path.read_text() if env_path.exists() else None
    env_path.write_text("UID=1000\nGID=1000\nCONTAINER_USER=chromie\nSIM_HOST=127.0.0.1\nSIM_PORT=5555\n")
    try:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["SORIDORMI_FAKE_DOCKER_CAPTURE"] = str(capture_path)
        subprocess.run(
            [
                "bash",
                "scripts/gate_dataset_scenario_coverage.sh",
                "data/training_datasets/flat.jsonl",
                "--require-scenario",
                "flat_walk_varied_speed_v1",
                "--output-dir",
                "artifacts/dataset_coverage/pre_bc",
                "--json",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    finally:
        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(old_env)

    docker_args = capture_path.read_text().splitlines()
    assert docker_args[:5] == ["compose", "-f", "compose.sim.yaml", "run", "--rm"]
    assert "--entrypoint" in docker_args
    assert "runtime" in docker_args
    assert "/data/training_datasets/flat.jsonl" in docker_args
    assert "/host_repo/artifacts/dataset_coverage/pre_bc" in docker_args
    assert "soridormi_runtime.dataset_scenario_gate" in " ".join(docker_args)


def test_gate_dataset_scenario_coverage_help_documents_mujoco_flow() -> None:
    proc = subprocess.run(
        ["bash", "scripts/gate_dataset_scenario_coverage.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--require-ready-locomotion" in proc.stdout
    assert "--min-command-range-fraction" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera" in proc.stdout
    assert "./scripts/collect_random_teacher_dataset.sh --backend mujoco --scenario flat_walk_varied_speed_v1" in proc.stdout
