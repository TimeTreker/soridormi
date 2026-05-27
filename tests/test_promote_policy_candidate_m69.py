from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from soridormi_runtime.promote_policy_candidate import promote_policy_candidate


def _copy_profile(tmp_path: Path, name: str = "candidate") -> Path:
    repo = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((repo / "configs/policies/open_duck_forward.yaml").read_text(encoding="utf-8"))
    payload["name"] = name
    payload["description"] = "Candidate profile for promotion tests"
    payload.setdefault("model", {})["path"] = "/tmp/candidate.onnx"
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _leaderboard(tmp_path: Path, profile_path: Path, *, promotable: bool = True) -> Path:
    path = tmp_path / "candidate_leaderboard.json"
    payload = {
        "ok": True,
        "schema_version": 1,
        "leaderboard_type": "soridormi.policy_candidate_leaderboard.v1",
        "best_profile": "candidate",
        "candidate_count": 1,
        "promotable_count": 1 if promotable else 0,
        "candidates": [
            {
                "rank": 1,
                "profile_name": "candidate",
                "profile_path": str(profile_path),
                "evaluation_path": str(tmp_path / "evaluation.json"),
                "output_dir": str(tmp_path),
                "ok": promotable,
                "promotable": promotable,
                "model_kind": "onnx",
                "model_path": "/tmp/candidate.onnx",
                "model_sha256": "abc123",
                "test_mae": 0.01,
                "test_rmse": 0.02,
                "test_max_abs_error": 0.03,
                "errors": [] if promotable else ["threshold failed"],
                "warnings": [],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_promote_policy_candidate_writes_profile_and_record(tmp_path: Path) -> None:
    source = _copy_profile(tmp_path)
    leaderboard = _leaderboard(tmp_path, source)

    result = promote_policy_candidate(
        leaderboard,
        target_profile="promoted_candidate",
        output_dir=tmp_path / "profiles",
        records_dir=tmp_path / "records",
        robot_config_path=Path(__file__).resolve().parents[1] / "configs/robots/open_duck_mini_v2.yaml",
    )

    assert result.ok
    target = Path(result.target_profile_path)
    assert target.exists()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["name"] == "promoted_candidate"
    assert payload["metadata"]["promoted_from_profile"] == "candidate"
    assert payload["metadata"]["promotion_model_sha256"] == "abc123"
    assert payload["logging"]["prefix"] == "policy_promoted_candidate"
    assert Path(result.promotion_record_path).exists()
    assert Path(result.promotion_report_path).read_text(encoding="utf-8").startswith("# Soridormi policy candidate promotion")


def test_promote_policy_candidate_rejects_non_promotable_by_default(tmp_path: Path) -> None:
    source = _copy_profile(tmp_path)
    leaderboard = _leaderboard(tmp_path, source, promotable=False)

    result = promote_policy_candidate(
        leaderboard,
        target_profile="unsafe_candidate",
        output_dir=tmp_path / "profiles",
        records_dir=tmp_path / "records",
        robot_config_path=Path(__file__).resolve().parents[1] / "configs/robots/open_duck_mini_v2.yaml",
    )

    assert not result.ok
    assert not Path(result.target_profile_path).exists()
    assert any("not promotable" in error for error in result.errors)


def test_promote_policy_candidate_cli_json_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = _copy_profile(tmp_path)
    leaderboard = _leaderboard(tmp_path, source)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.promote_policy_candidate",
            str(leaderboard),
            "--target-profile",
            "cli_promoted",
            "--output-dir",
            str(tmp_path / "profiles"),
            "--records-dir",
            str(tmp_path / "records"),
            "--robot-config",
            str(repo / "configs/robots/open_duck_mini_v2.yaml"),
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
    assert payload["target_profile"] == "cli_promoted"
