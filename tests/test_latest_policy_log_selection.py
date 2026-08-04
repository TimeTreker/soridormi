from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _touch(path: Path, timestamp: int) -> None:
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def _find_latest(log_dir: Path) -> str:
    command = (
        "source scripts/lib/latest_policy_log.sh && "
        f"find_latest_policy_log {str(log_dir)!r}"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_latest_policy_log_ignores_newer_scenario_jsonl(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    policy_log = log_dir / "parity_open_duck_forward_20260601_000000.jsonl"
    scenario_log = log_dir / "scenario_flat_walk_varied_speed_v1_20260603_000000.jsonl"
    _touch(policy_log, 100)
    _touch(scenario_log, 200)

    assert _find_latest(log_dir) == str(policy_log)


def test_latest_policy_log_uses_newest_eligible_policy_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    older = log_dir / "parity_open_duck_forward_20260601_000000.jsonl"
    newer = log_dir / "policy_open_duck_forward_20260601_010000.mcap"
    newest_irrelevant = log_dir / "skill_walk_velocity_20260601_020000.jsonl"
    _touch(older, 100)
    _touch(newer, 200)
    _touch(newest_irrelevant, 300)

    assert _find_latest(log_dir) == str(newer)


def test_latest_policy_log_allows_runtime_fallback(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    runtime_log = log_dir / "runtime_20260601_000000.jsonl"
    _touch(runtime_log, 100)

    assert _find_latest(log_dir) == str(runtime_log)
