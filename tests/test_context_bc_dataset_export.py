from __future__ import annotations

import json
import subprocess
from pathlib import Path

from soridormi_runtime.bc_training_contract import DEFAULT_CONTRACT_PATH, validate_sample_jsonl
from soridormi_runtime.context_bc_dataset_export import export_context_bc_dataset

OBS = [0.0] * 101
ACTION_A = [0.1] * 14
ACTION_B = [0.2] * 14


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def base_row(**overrides):
    row = {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.v1",
        "scenario_id": "flat_walk_varied_speed_v1",
        "rollout_id": "flat_walk_varied_speed_v1:episode_0",
        "step_index": 0,
        "episode_index": 0,
        "episode_step_index": 0,
        "observation": OBS,
        "action": ACTION_A,
        "policy_command": [0.12, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        "desired_command": {"x_velocity": 0.12, "y_velocity": 0.0, "yaw_velocity": 0.05},
        "applied_command": {"x_velocity": 0.11, "y_velocity": 0.0, "yaw_velocity": 0.04},
        "command_ramp_alpha": 0.8,
        "command_ramp_name": "linear_segment_ramp",
        "policy_debug": {"terminated": False},
    }
    row.update(overrides)
    return row


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_converts_scenario_aware_rows_to_context_contract(tmp_path: Path) -> None:
    source = tmp_path / "flat.jsonl"
    output = tmp_path / "context.jsonl"
    write_jsonl(source, [base_row()])

    result = export_context_bc_dataset([source], output_path=output)

    assert result.ok, result.errors
    assert result.converted_count == 1
    rows = read_jsonl(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_type"] == "soridormi.policy_supervision.context_v1"
    assert row["robot_state"]["observation"] == OBS
    assert row["teacher_action"] == ACTION_A
    assert row["desired_command"] == {"vx_mps": 0.12, "vy_mps": 0.0, "yaw_radps": 0.05}
    assert row["applied_command"] == {"vx_mps": 0.11, "vy_mps": 0.0, "yaw_radps": 0.04}
    assert row["task_context"]["skill_id"] == "walk_velocity"
    assert row["environment_context"]["terrain_type"] == "flat"
    assert row["short_history"]["previous_action"] == [0.0] * 14

    contract, _contract_result = __import__("soridormi_runtime.bc_training_contract", fromlist=["load_and_validate_contract"]).load_and_validate_contract(DEFAULT_CONTRACT_PATH)
    assert contract is not None
    validation = validate_sample_jsonl(output, contract)
    assert validation.ok, validation.errors


def test_export_adds_short_history_from_previous_row_in_rollout(tmp_path: Path) -> None:
    source = tmp_path / "flat.jsonl"
    output = tmp_path / "context.jsonl"
    first = base_row(step_index=0, action=ACTION_A, applied_command={"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.0})
    second = base_row(step_index=1, episode_step_index=1, action=ACTION_B, applied_command={"vx_mps": 0.20, "vy_mps": 0.0, "yaw_radps": 0.0})
    write_jsonl(source, [first, second])

    result = export_context_bc_dataset([source], output_path=output)

    assert result.ok, result.errors
    rows = read_jsonl(output)
    assert rows[1]["short_history"]["previous_action"] == ACTION_A
    assert rows[1]["short_history"]["previous_command"] == {"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.0}


def test_export_can_be_strict_about_unknown_scenarios(tmp_path: Path) -> None:
    source = tmp_path / "unknown.jsonl"
    output = tmp_path / "context.jsonl"
    write_jsonl(source, [base_row(scenario_id="missing_scenario")])

    result = export_context_bc_dataset([source], output_path=output, strict_context=True)

    assert not result.ok
    assert result.converted_count == 0
    assert any("not found in scenario manifest" in error for error in result.errors)


def test_cli_writes_json_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "flat.jsonl"
    output = tmp_path / "context.jsonl"
    write_jsonl(source, [base_row()])

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.context_bc_dataset_export",
            str(source),
            "--output",
            str(output),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["converted_count"] == 1
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    assert manifest.exists()


def test_shell_wrapper_maps_paths_inside_runtime_container(tmp_path: Path) -> None:
    capture_path = tmp_path / "docker_args.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${SORIDORMI_FAKE_DOCKER_CAPTURE}\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env_path = Path(".env")
    old_env = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    env_path.write_text("UID=1000\nGID=1000\nCONTAINER_USER=chromie\nSIM_HOST=127.0.0.1\nSIM_PORT=5555\n", encoding="utf-8")
    try:
        env = dict(__import__("os").environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["SORIDORMI_FAKE_DOCKER_CAPTURE"] = str(capture_path)
        subprocess.run(
            [
                "bash",
                "scripts/export_context_bc_dataset.sh",
                "data/input.jsonl",
                "--output",
                "artifacts/context/out.jsonl",
                "--report",
                "artifacts/context/report.md",
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
            env_path.write_text(old_env, encoding="utf-8")

    docker_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert docker_args[:5] == ["compose", "-f", "compose.sim.yaml", "run", "--rm"]
    assert "--entrypoint" in docker_args
    assert "runtime" in docker_args
    assert "/data/input.jsonl" in docker_args
    assert "/host_repo/artifacts/context/out.jsonl" in docker_args
    assert "/host_repo/artifacts/context/report.md" in docker_args


def test_export_does_not_overwrite_existing_output_when_no_rows_convert(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    output = tmp_path / "context.jsonl"
    source.write_text("", encoding="utf-8")
    output.write_text(json.dumps({"existing": True}) + "\n", encoding="utf-8")

    result = export_context_bc_dataset([source], output_path=output)

    assert not result.ok
    assert result.converted_count == 0
    assert result.output_written is False
    assert any("output file was not updated" in error for error in result.errors)
    assert read_jsonl(output) == [{"existing": True}]


def test_export_removes_temporary_output_on_failed_conversion(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    output = tmp_path / "context.jsonl"
    write_jsonl(source, [base_row(action=[0.1] * 13)])

    result = export_context_bc_dataset([source], output_path=output)

    assert not result.ok
    assert result.output_written is False
    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp").exists()
