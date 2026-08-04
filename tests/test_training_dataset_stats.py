from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.training_dataset_stats import analyze_prepared_training_dataset


def _sample(step: int, *, obs0: float | None = None, action0: float | None = None) -> dict:
    observation = [float(step)] * 101
    action = [float(step) / 10.0] * 14
    if obs0 is not None:
        observation[0] = obs0
    if action0 is not None:
        action[0] = action0
    return {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.v1",
        "source_log": "runtime.jsonl",
        "step_index": step,
        "robot_time": step * 0.02,
        "observation": observation,
        "action": action,
        "raw_action": action,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepared_dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_sample(0, obs0=1.0, action0=0.25), _sample(1, obs0=3.0, action0=0.75)])
    _write_jsonl(prepared / "val.jsonl", [_sample(2)])
    _write_jsonl(prepared / "test.jsonl", [_sample(3)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 2},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 1},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 1},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prepared


def test_analyze_prepared_training_dataset_writes_stats_normalization_and_report(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)

    result = analyze_prepared_training_dataset(prepared)

    assert result.ok
    assert result.sample_count == 4
    assert result.splits["train"].sample_count == 2
    assert result.splits["val"].sample_count == 1
    assert result.splits["test"].sample_count == 1
    assert Path(result.stats_path).exists()
    assert Path(result.normalization_path).exists()
    assert Path(result.report_path).exists()

    normalization = json.loads(Path(result.normalization_path).read_text(encoding="utf-8"))
    assert normalization["normalization_type"] == "soridormi.policy_supervision.normalization.v1"
    assert normalization["sample_count"] == 2
    assert normalization["observation_mean"][0] == 2.0
    assert normalization["observation_std"][0] == 1.0
    assert normalization["action_mean"][0] == 0.5
    assert normalization["action_std"][0] == 0.25
    assert normalization["observation_std"][1] == 0.5


def test_analyze_prepared_training_dataset_accepts_manifest_path_and_custom_output_dir(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    output_dir = tmp_path / "stats_out"

    result = analyze_prepared_training_dataset(prepared / "prepared_manifest.json", output_dir=output_dir)

    assert result.ok
    assert Path(result.stats_path).parent == output_dir
    assert Path(result.normalization_path).parent == output_dir
    stats = json.loads(Path(result.stats_path).read_text(encoding="utf-8"))
    assert stats["stats_type"] == "soridormi.policy_supervision.stats.v1"
    assert stats["splits"]["train"]["sample_count"] == 2


def test_analyze_prepared_training_dataset_reports_bad_split_vectors(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [{**_sample(0), "observation": [0.0] * 100}])
    _write_jsonl(prepared / "val.jsonl", [_sample(1)])
    _write_jsonl(prepared / "test.jsonl", [_sample(2)])
    manifest = {
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"path": str(prepared / "train.jsonl"), "sample_count": 1},
            "val": {"path": str(prepared / "val.jsonl"), "sample_count": 1},
            "test": {"path": str(prepared / "test.jsonl"), "sample_count": 1},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = analyze_prepared_training_dataset(prepared)

    assert not result.ok
    assert any("observation size 100" in error for error in result.errors)
    assert any("train split has no valid samples" in error for error in result.errors)
