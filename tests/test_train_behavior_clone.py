from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soridormi_runtime.train_behavior_clone import (
    INPUT_MODE_CONTEXT_COMMAND_V1,
    predict_linear_behavior_clone,
    train_behavior_clone_baseline,
)
from soridormi_runtime.training_dataset_stats import analyze_prepared_training_dataset


def _sample(step: int) -> dict:
    observation = [0.0] * 101
    observation[0] = float(step)
    observation[1] = float(step % 2)
    action = [0.0] * 14
    action[0] = 0.5 * observation[0] - 0.25 * observation[1] + 0.1
    action[1] = -0.2 * observation[0] + 0.05
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


def _context_sample(step: int) -> dict:
    legacy = _sample(step)
    return {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.context_v1",
        "scenario_id": "flat_walk_varied_speed_v1",
        "rollout_id": f"rollout_{step // 4}",
        "skill_id": "walk_velocity",
        "step_index": step,
        "robot_time": step * 0.02,
        "robot_state": {"observation": legacy["observation"]},
        "desired_command": {"vx_mps": 0.1, "vy_mps": 0.0, "yaw_radps": 0.0},
        "task_context": {"skill_family": "locomotion"},
        "environment_context": {"terrain_type": "flat"},
        "short_history": {
            "previous_action": [0.0] * 14,
            "previous_command": {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_radps": 0.0},
        },
        "teacher_action": legacy["action"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepared_dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_sample(step) for step in range(8)])
    _write_jsonl(prepared / "val.jsonl", [_sample(step) for step in range(8, 10)])
    _write_jsonl(prepared / "test.jsonl", [_sample(step) for step in range(10, 12)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 8},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 2},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 2},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stats = analyze_prepared_training_dataset(prepared)
    assert stats.ok
    return prepared


def _context_prepared_dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "context_prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_context_sample(step) for step in range(8)])
    _write_jsonl(prepared / "val.jsonl", [_context_sample(step) for step in range(8, 10)])
    _write_jsonl(prepared / "test.jsonl", [_context_sample(step) for step in range(10, 12)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.context_prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 8},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 2},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 2},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prepared


def test_train_behavior_clone_baseline_writes_model_metrics_and_report(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)

    result = train_behavior_clone_baseline(prepared, ridge_lambda=0.0)

    assert result.ok
    assert result.train_sample_count == 8
    assert result.metrics["train"].mae is not None
    assert result.metrics["train"].mae < 1e-9
    assert Path(result.model_path).exists()
    assert result.normalization_path is not None
    normalization = json.loads(Path(result.normalization_path).read_text(encoding="utf-8"))
    assert normalization["normalization_type"] == "soridormi.policy_supervision.normalization.v1"
    assert len(normalization["observation_mean"]) == 101
    assert len(normalization["action_mean"]) == 14
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()

    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["training_run_type"] == "soridormi.policy_supervision.linear_behavior_clone.v1"
    assert metrics["model_sha256"]

    model = np.load(result.model_path)
    assert model["weights"].shape == (101, 14)
    assert model["bias"].shape == (14,)


def test_trained_linear_model_can_predict_actions(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    result = train_behavior_clone_baseline(prepared, ridge_lambda=0.0)
    assert result.ok

    model = np.load(result.model_path)
    observations = np.asarray([_sample(3)["observation"], _sample(9)["observation"]], dtype=np.float64)
    expected = np.asarray([_sample(3)["action"], _sample(9)["action"]], dtype=np.float64)
    normalization = {
        "observation_mean": model["observation_mean"].astype(np.float64),
        "observation_std": model["observation_std"].astype(np.float64),
        "action_mean": model["action_mean"].astype(np.float64),
        "action_std": model["action_std"].astype(np.float64),
    }

    predicted = predict_linear_behavior_clone(
        observations,
        weights=model["weights"].astype(np.float64),
        bias=model["bias"].astype(np.float64),
        normalization=normalization,
    )

    np.testing.assert_allclose(predicted[:, :2], expected[:, :2], atol=1e-5)


def test_train_behavior_clone_reports_missing_manifest(tmp_path: Path) -> None:
    result = train_behavior_clone_baseline(tmp_path / "missing_prepared")

    assert not result.ok
    assert any("File not found" in error for error in result.errors)
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()


def test_train_behavior_clone_accepts_context_prepared_rows(tmp_path: Path) -> None:
    prepared = _context_prepared_dataset(tmp_path)

    result = train_behavior_clone_baseline(prepared, ridge_lambda=0.0)

    assert result.ok
    assert result.train_sample_count == 8
    assert result.metrics["train"].mae is not None
    assert result.metrics["train"].mae < 1e-9
    assert Path(result.model_path).exists()


def test_train_behavior_clone_command_context_mode_appends_command_features(tmp_path: Path) -> None:
    prepared = _context_prepared_dataset(tmp_path)

    result = train_behavior_clone_baseline(
        prepared,
        output_dir=tmp_path / "run",
        ridge_lambda=0.0,
        input_mode=INPUT_MODE_CONTEXT_COMMAND_V1,
    )

    assert result.ok
    assert result.input_mode == INPUT_MODE_CONTEXT_COMMAND_V1
    assert result.observation_size == 104
    assert result.normalization_path is not None
    assert Path(result.normalization_path).name == "normalization.context_command_v1.json"
    normalization = json.loads(Path(result.normalization_path).read_text(encoding="utf-8"))
    assert normalization["input_mode"] == INPUT_MODE_CONTEXT_COMMAND_V1
    assert len(normalization["observation_mean"]) == 104

    model = np.load(result.model_path)
    assert model["weights"].shape == (104, 14)
    assert str(model["input_mode"][0]) == INPUT_MODE_CONTEXT_COMMAND_V1


def test_train_behavior_clone_command_context_mode_requires_desired_command(tmp_path: Path) -> None:
    prepared = tmp_path / "context_prepared"
    prepared.mkdir()
    bad = _context_sample(0)
    bad.pop("desired_command")
    _write_jsonl(prepared / "train.jsonl", [bad])
    _write_jsonl(prepared / "val.jsonl", [_context_sample(1)])
    _write_jsonl(prepared / "test.jsonl", [_context_sample(2)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.context_prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 1},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 1},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 1},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = train_behavior_clone_baseline(
        prepared,
        output_dir=tmp_path / "run",
        input_mode=INPUT_MODE_CONTEXT_COMMAND_V1,
    )

    assert not result.ok
    assert any("desired_command must be an object" in error for error in result.errors)


def test_train_behavior_clone_wrapper_overrides_cuda_entrypoint_for_json_stdout() -> None:
    script = Path("scripts/train_behavior_clone.sh").read_text(encoding="utf-8")

    assert "--entrypoint bash" in script
    assert "python -m soridormi_runtime.train_behavior_clone" in script
