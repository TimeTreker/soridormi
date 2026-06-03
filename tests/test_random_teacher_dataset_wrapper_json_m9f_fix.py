from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _write_env_file() -> str | None:
    env_path = Path(".env")
    old_env = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    env_path.write_text(
        "UID=1000\nGID=1000\nCONTAINER_USER=chromie\nSIM_HOST=127.0.0.1\nSIM_PORT=5555\n",
        encoding="utf-8",
    )
    return old_env


def _restore_env_file(old_env: str | None) -> None:
    env_path = Path(".env")
    if old_env is None:
        env_path.unlink(missing_ok=True)
    else:
        env_path.write_text(old_env, encoding="utf-8")


def test_random_teacher_collector_shell_json_keeps_status_off_stdout(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'compose/container status noise\\n' >&2\n"
        "cat <<'JSON'\n"
        "{\"ok\":true,\"sample_count\":6,\"output_path\":\"/data/training_datasets/demo.jsonl\",\"errors\":[]}\n"
        "JSON\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    old_env = _write_env_file()
    try:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        result = subprocess.run(
            [
                "bash",
                "scripts/collect_random_teacher_dataset.sh",
                "--backend",
                "mujoco",
                "--scenario",
                "flat_walk_varied_speed_v1",
                "--episodes",
                "1",
                "--steps-per-episode",
                "6",
                "--output",
                "/data/training_datasets/demo.jsonl",
                "--json",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    finally:
        _restore_env_file(old_env)

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["sample_count"] == 6
    assert "Soridormi random-command teacher collection" not in result.stdout
    assert "Expected sim backend" not in result.stdout
    assert "Soridormi random-command teacher collection" in result.stderr
    assert "compose/container status noise" in result.stderr


def test_random_teacher_collector_shell_json_reports_pre_json_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'sim connection failed before collector JSON\\n' >&2\n"
        "exit 17\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    old_env = _write_env_file()
    try:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        result = subprocess.run(
            [
                "bash",
                "scripts/collect_random_teacher_dataset.sh",
                "--backend",
                "mujoco",
                "--scenario",
                "flat_walk_varied_speed_v1",
                "--episodes",
                "1",
                "--steps-per-episode",
                "6",
                "--output",
                "/data/training_datasets/demo.jsonl",
                "--json",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    finally:
        _restore_env_file(old_env)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["docker_exit_code"] == 17
    assert "collector stdout was empty" in payload["errors"]
    assert "sim connection failed before collector JSON" in result.stderr
