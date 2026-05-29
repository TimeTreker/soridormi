from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.training_dataset_prepare import (
    load_and_validate_dataset,
    split_training_dataset,
    validate_training_sample,
)


def _sample(step: int, *, observation_size: int = 101, action_size: int = 14) -> dict:
    return {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.v1",
        "source_log": "runtime.jsonl",
        "step_index": step,
        "robot_time": step * 0.02,
        "next_robot_time": (step + 1) * 0.02,
        "observation": [float(step)] * observation_size,
        "action": [float(step) / 10.0] * action_size,
        "raw_action": [float(step) / 5.0] * action_size,
        "policy_command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state": {"joint_positions": [0.0]},
        "next_state": {"joint_positions": [0.1]},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_validate_training_sample_accepts_exported_sample() -> None:
    errors, warnings = validate_training_sample(_sample(0))

    assert errors == []
    assert warnings == []


def test_validate_training_sample_rejects_bad_vectors() -> None:
    sample = _sample(0, observation_size=100)
    sample["action"] = [0.0] * 13
    sample["raw_action"] = [float("nan")] * 14

    errors, _warnings = validate_training_sample(sample)

    assert any("observation size 100" in error for error in errors)
    assert any("action size 13" in error for error in errors)
    assert any("raw_action contains" in error for error in errors)


def test_load_and_validate_dataset_reports_invalid_json_and_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(_sample(0)) + "\n" +
        "not json\n" +
        json.dumps(_sample(1, action_size=13)) + "\n",
        encoding="utf-8",
    )

    samples, summary = load_and_validate_dataset(dataset)

    assert len(samples) == 1
    assert not summary.ok
    assert summary.sample_count == 3
    assert summary.valid_sample_count == 1
    assert summary.invalid_sample_count == 2
    assert any("invalid JSON" in error for error in summary.errors)
    assert any("action size 13" in error for error in summary.errors)


def test_split_training_dataset_writes_deterministic_splits_and_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(dataset, [_sample(step) for step in range(10)])

    result = split_training_dataset(dataset, output_dir=output_dir, seed=123)
    result_again = split_training_dataset(dataset, output_dir=tmp_path / "prepared_again", seed=123)

    assert result.ok
    assert result.sample_count == 10
    assert result.valid_sample_count == 10
    assert result.invalid_sample_count == 0
    assert result.train.sample_count == 8
    assert result.val.sample_count == 1
    assert result.test.sample_count == 1
    assert Path(result.train.path).exists()
    assert Path(result.val.path).exists()
    assert Path(result.test.path).exists()
    assert Path(result.manifest_path).exists()
    assert result.train.sha256 == result_again.train.sha256
    assert result.val.sha256 == result_again.val.sha256
    assert result.test.sha256 == result_again.test.sha256

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "soridormi.policy_supervision.prepared.v1"
    assert manifest["splits"]["train"]["sample_count"] == 8


def test_split_training_dataset_can_preserve_input_order(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "prepared"
    _write_jsonl(dataset, [_sample(step) for step in range(5)])

    result = split_training_dataset(
        dataset,
        output_dir=output_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        shuffle=False,
    )

    assert result.ok
    train_rows = [json.loads(line) for line in Path(result.train.path).read_text(encoding="utf-8").splitlines()]
    assert [row["step_index"] for row in train_rows] == [0, 1, 2]


def test_split_training_dataset_rejects_bad_ratios(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, [_sample(0)])

    result = split_training_dataset(dataset, output_dir=tmp_path / "prepared", train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    assert not result.ok
    assert any("sum to 1.0" in error for error in result.errors)


def test_split_training_dataset_can_split_by_rollout_group_without_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for rollout in range(5):
        for step in range(3):
            sample = _sample(rollout * 10 + step)
            sample["source_log"] = f"rollout_{rollout}.jsonl"
            sample["scenario_id"] = f"scenario_{rollout % 2}"
            rows.append(sample)
    _write_jsonl(dataset, rows)

    result = split_training_dataset(
        dataset,
        output_dir=tmp_path / "prepared_grouped",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
        split_group_field="source_log",
    )

    assert result.ok
    assert result.split_group_field == "source_log"
    assert result.split_group_counts == {"train": 3, "val": 1, "test": 1}
    assert result.train.sample_count == 9
    assert result.val.sample_count == 3
    assert result.test.sample_count == 3

    split_sources = {}
    for split in (result.train, result.val, result.test):
        rows = [json.loads(line) for line in Path(split.path).read_text(encoding="utf-8").splitlines()]
        split_sources[split.name] = {row["source_log"] for row in rows}

    assert split_sources["train"].isdisjoint(split_sources["val"])
    assert split_sources["train"].isdisjoint(split_sources["test"])
    assert split_sources["val"].isdisjoint(split_sources["test"])

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["split_group_field"] == "source_log"
    assert manifest["splits"]["train"]["group_count"] == 3
