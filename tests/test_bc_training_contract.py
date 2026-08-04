from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from soridormi_runtime.bc_training_contract import (
    CONTEXT_SAMPLE_TYPE,
    DEFAULT_CONTRACT_PATH,
    LEGACY_SAMPLE_TYPE,
    build_report,
    load_and_validate_contract,
    validate_sample_jsonl,
)


def _context_sample() -> dict:
    return {
        "sample_type": CONTEXT_SAMPLE_TYPE,
        "schema_version": 1,
        "scenario_id": "flat_walk_varied_speed_v1",
        "rollout_id": "rollout_a",
        "timestep": 1,
        "skill_id": "walk_velocity",
        "robot_state": {"observation": [0.0] * 101},
        "desired_command": {"vx_mps": 0.12, "vy_mps": 0.0, "yaw_radps": 0.03},
        "task_context": {"skill_id": "walk_velocity", "gait_style": "default_walk"},
        "environment_context": {"terrain_type": "flat", "friction_estimate": 1.0},
        "teacher_action": [0.0] * 14,
        "failure_flags": {"fallen": False, "stuck": False},
    }


def test_bc_training_contract_config_validates() -> None:
    contract, result = load_and_validate_contract(DEFAULT_CONTRACT_PATH)
    assert contract is not None
    assert result.ok, result.errors
    assert result.contract_id == "open_duck_mini_v2_context_bc_v1"
    assert result.robot_profile == "open_duck_mini_v2"
    assert result.action_size == 14
    assert result.observation_size == 101
    assert result.natural_language_allowed is False
    assert result.input_groups == [
        "robot_state",
        "desired_command",
        "task_context",
        "environment_context",
        "short_history",
    ]


def test_bc_training_contract_validates_context_jsonl(tmp_path: Path) -> None:
    sample_path = tmp_path / "context.jsonl"
    sample_path.write_text(json.dumps(_context_sample()) + "\n", encoding="utf-8")
    contract, contract_result = load_and_validate_contract(DEFAULT_CONTRACT_PATH)
    assert contract is not None
    assert contract_result.ok

    result = validate_sample_jsonl(sample_path, contract)
    assert result.ok, result.errors
    assert result.sample_count == 1
    assert result.valid_count == 1
    assert result.context_sample_count == 1
    assert result.legacy_sample_count == 0


def test_bc_training_contract_rejects_missing_context_fields(tmp_path: Path) -> None:
    bad = _context_sample()
    del bad["environment_context"]
    sample_path = tmp_path / "bad.jsonl"
    sample_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    contract, _result = load_and_validate_contract(DEFAULT_CONTRACT_PATH)
    assert contract is not None

    result = validate_sample_jsonl(sample_path, contract)
    assert not result.ok
    assert result.invalid_count == 1
    assert any("environment_context" in error for error in result.errors)


def test_bc_training_contract_accepts_legacy_only_when_explicit(tmp_path: Path) -> None:
    legacy = {
        "sample_type": LEGACY_SAMPLE_TYPE,
        "schema_version": 1,
        "observation": [0.0] * 101,
        "action": [0.0] * 14,
    }
    sample_path = tmp_path / "legacy.jsonl"
    sample_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    contract, _result = load_and_validate_contract(DEFAULT_CONTRACT_PATH)
    assert contract is not None

    strict_result = validate_sample_jsonl(sample_path, contract)
    assert not strict_result.ok
    assert strict_result.invalid_count == 1

    bridge_result = validate_sample_jsonl(sample_path, contract, allow_legacy=True)
    assert bridge_result.ok, bridge_result.errors
    assert bridge_result.valid_count == 1
    assert bridge_result.legacy_sample_count == 1
    assert bridge_result.warnings


def test_bc_training_contract_writes_markdown_report(tmp_path: Path) -> None:
    sample_path = tmp_path / "context.jsonl"
    sample_path.write_text(json.dumps(_context_sample()) + "\n", encoding="utf-8")
    out = tmp_path / "report.md"

    report = build_report(contract_path=DEFAULT_CONTRACT_PATH, sample_jsonl=sample_path)
    assert report.ok

    completed = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.bc_training_contract",
            "--contract",
            str(DEFAULT_CONTRACT_PATH),
            "--sample-jsonl",
            str(sample_path),
            "--output",
            str(out),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert out.exists()
    assert "BC Training Contract Report" in out.read_text(encoding="utf-8")


def test_validate_bc_training_contract_shell_wrapper_json_is_parseable(tmp_path: Path) -> None:
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
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["SORIDORMI_FAKE_DOCKER_CAPTURE"] = str(capture_path)
        subprocess.run(
            ["bash", "scripts/validate_bc_training_contract.sh", "--json"],
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
    assert "/host_repo" in "\n".join(docker_args)
    assert "soridormi_runtime.bc_training_contract" in "\n".join(docker_args)
