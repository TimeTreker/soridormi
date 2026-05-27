from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.policy_candidate_leaderboard import build_policy_candidate_leaderboard, find_evaluation_files


def _write_eval(root: Path, profile: str, *, test_mae: float, ok: bool = True) -> Path:
    directory = root / profile
    directory.mkdir(parents=True)
    path = directory / "evaluation.json"
    payload = {
        "ok": ok,
        "profile_name": profile,
        "output_dir": str(directory),
        "model_kind": "linear_behavior_clone",
        "model_path": f"data/models/{profile}.npz",
        "model_sha256": "abc123",
        "splits": {
            "train": {"sample_count": 10, "mae": test_mae / 2, "rmse": test_mae, "max_abs_error": test_mae * 2},
            "val": {"sample_count": 4, "mae": test_mae * 1.5, "rmse": test_mae * 2, "max_abs_error": test_mae * 3},
            "test": {"sample_count": 4, "mae": test_mae, "rmse": test_mae * 1.25, "max_abs_error": test_mae * 2},
        },
        "errors": [],
        "warnings": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_find_evaluation_files_scans_nested_directories(tmp_path: Path) -> None:
    one = _write_eval(tmp_path / "runs", "candidate_a", test_mae=0.2)
    two = _write_eval(tmp_path / "runs" / "nested", "candidate_b", test_mae=0.1)

    found, warnings = find_evaluation_files([tmp_path / "runs", tmp_path / "missing"])

    assert found == sorted([one, two])
    assert any("Search path not found" in warning for warning in warnings)


def test_leaderboard_ranks_promotable_candidates_by_test_mae(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    _write_eval(root, "slow_candidate", test_mae=0.09)
    _write_eval(root, "best_candidate", test_mae=0.03)
    _write_eval(root, "bad_candidate", test_mae=0.2)

    result = build_policy_candidate_leaderboard(
        [root],
        output_dir=tmp_path / "leaderboard",
        max_test_mae=0.1,
        require_promotable=True,
    )

    assert result.ok
    assert result.best_profile == "best_candidate"
    assert result.promotable_count == 2
    assert [candidate.profile_name for candidate in result.candidates[:2]] == ["best_candidate", "slow_candidate"]
    assert Path(result.leaderboard_path).exists()
    assert Path(result.report_path).read_text(encoding="utf-8").startswith("# Soridormi policy candidate leaderboard")


def test_leaderboard_can_fail_when_no_candidate_is_promotable(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    _write_eval(root, "candidate", test_mae=0.5)

    result = build_policy_candidate_leaderboard(
        [root],
        output_dir=tmp_path / "leaderboard",
        max_test_mae=0.1,
        require_promotable=True,
    )

    assert not result.ok
    assert result.promotable_count == 0
    assert "No promotable candidate found" in result.errors
    assert result.candidates[0].errors


def test_rank_policy_candidates_script_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "evaluations"
    _write_eval(root, "candidate", test_mae=0.01)
    out = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.policy_candidate_leaderboard",
            str(root),
            "--output-dir",
            str(out),
            "--max-test-mae",
            "0.02",
            "--require-promotable",
            "--json",
        ],
        cwd=repo,
        env={"PYTHONPATH": str(repo / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["best_profile"] == "candidate"
