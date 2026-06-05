from __future__ import annotations

import json
from pathlib import Path

import pytest

from soridormi_runtime.train_neural_behavior_clone import (
    _parse_hidden_sizes,
    train_neural_behavior_clone,
)
from soridormi_runtime.train_behavior_clone import INPUT_MODE_CONTEXT_STAGE1_COMMAND
from soridormi_runtime.training_dataset_stats import analyze_prepared_training_dataset

pytest.importorskip("torch")


def _sample(step: int) -> dict:
    x = float(step) / 10.0
    observation = [0.0] * 101
    observation[0] = x
    observation[1] = x * x
    action = [0.0] * 14
    action[0] = 0.2 * x + 0.1
    action[1] = -0.3 * x * x + 0.05
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
        "rollout_id": f"rollout_{step // 8}",
        "skill_id": "walk_velocity",
        "step_index": step,
        "robot_time": step * 0.02,
        "robot_state": {"observation": legacy["observation"]},
        "desired_command": {
            "vx_mps": float(step % 3) * 0.05,
            "vy_mps": 0.0,
            "yaw_radps": float(step % 2) * 0.02,
        },
        "task_context": {"skill_id": "walk_velocity", "gait_style": "default_walk"},
        "environment_context": {"terrain_type": "flat"},
        "teacher_action": legacy["action"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepared_dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_sample(step) for step in range(24)])
    _write_jsonl(prepared / "val.jsonl", [_sample(step) for step in range(24, 30)])
    _write_jsonl(prepared / "test.jsonl", [_sample(step) for step in range(30, 36)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 24},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 6},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 6},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stats = analyze_prepared_training_dataset(prepared)
    assert stats.ok
    return prepared


def _context_prepared_dataset(tmp_path: Path) -> Path:
    prepared = tmp_path / "context_prepared"
    prepared.mkdir()
    _write_jsonl(prepared / "train.jsonl", [_context_sample(step) for step in range(24)])
    _write_jsonl(prepared / "val.jsonl", [_context_sample(step) for step in range(24, 30)])
    _write_jsonl(prepared / "test.jsonl", [_context_sample(step) for step in range(30, 36)])
    manifest = {
        "schema_version": 1,
        "dataset_type": "soridormi.policy_supervision.context_prepared.v1",
        "splits": {
            "train": {"name": "train", "path": str(prepared / "train.jsonl"), "sample_count": 24},
            "val": {"name": "val", "path": str(prepared / "val.jsonl"), "sample_count": 6},
            "test": {"name": "test", "path": str(prepared / "test.jsonl"), "sample_count": 6},
        },
    }
    (prepared / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prepared


def test_parse_hidden_sizes_accepts_commas_and_x() -> None:
    assert _parse_hidden_sizes("32,64") == [32, 64]
    assert _parse_hidden_sizes("32x64") == [32, 64]
    with pytest.raises(ValueError):
        _parse_hidden_sizes("0,64")


def test_train_neural_behavior_clone_writes_checkpoint_metrics_and_report(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)

    result = train_neural_behavior_clone(
        prepared,
        output_dir=tmp_path / "run",
        hidden_sizes=[32],
        epochs=8,
        batch_size=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=7,
        device="cpu",
        export_onnx=False,
        create_profile=False,
    )

    assert result.ok
    assert result.train_sample_count == 24
    assert Path(result.checkpoint_path).exists()
    assert result.onnx_path is None
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()
    assert result.metrics["train"].mae is not None

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert payload["training_run_type"] == "soridormi.policy_supervision.neural_behavior_clone.v1"
    assert payload["checkpoint_sha256"]
    assert payload["history"]


def test_train_neural_behavior_clone_reports_missing_manifest(tmp_path: Path) -> None:
    result = train_neural_behavior_clone(
        tmp_path / "missing_prepared",
        output_dir=tmp_path / "run",
        export_onnx=False,
        create_profile=False,
        epochs=1,
        device="cpu",
    )

    assert not result.ok
    assert any("File not found" in error for error in result.errors)
    assert Path(result.metrics_path).exists()
    assert Path(result.report_path).exists()


def test_train_neural_behavior_clone_accepts_stage1_context_mode_checkpoint_only(tmp_path: Path) -> None:
    prepared = _context_prepared_dataset(tmp_path)

    result = train_neural_behavior_clone(
        prepared,
        output_dir=tmp_path / "run",
        input_mode=INPUT_MODE_CONTEXT_STAGE1_COMMAND,
        hidden_sizes=[16],
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
        export_onnx=False,
        create_profile=False,
    )

    assert result.ok
    assert result.input_mode == INPUT_MODE_CONTEXT_STAGE1_COMMAND
    assert result.observation_size == 104
    assert result.normalization_path is not None
    assert Path(result.normalization_path).name == "normalization.context_stage1_command.json"
    assert Path(result.checkpoint_path).exists()


def test_train_neural_behavior_clone_rejects_context_mode_onnx_export(tmp_path: Path) -> None:
    prepared = _context_prepared_dataset(tmp_path)

    result = train_neural_behavior_clone(
        prepared,
        output_dir=tmp_path / "run",
        input_mode=INPUT_MODE_CONTEXT_STAGE1_COMMAND,
        export_onnx=True,
        create_profile=False,
        epochs=1,
        device="cpu",
    )

    assert not result.ok
    assert any("--skip-onnx" in error for error in result.errors)


def test_train_neural_behavior_clone_wrapper_overrides_cuda_entrypoint_for_json_stdout() -> None:
    script = Path("scripts/train_neural_behavior_clone.sh").read_text(encoding="utf-8")

    assert "--entrypoint bash" in script
    assert "python -m soridormi_runtime.train_neural_behavior_clone" in script
