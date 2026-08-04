from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.create_linear_bc_profile import create_linear_bc_profile
from soridormi_runtime.evaluate_policy_profile import evaluate_policy_profile
from soridormi_runtime.train_behavior_clone import INPUT_MODE_CONTEXT_COMMAND_V1, train_behavior_clone_baseline
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _context_sample(step: int) -> dict:
    observation = [0.0] * 101
    observation[0] = float(step)
    observation[1] = float(step % 2)
    vx = float(step % 4) * 0.1
    yaw = float(step % 3) * 0.02
    action = [0.0] * 14
    action[0] = 0.5 * observation[0] - 0.25 * observation[1] + 2.0 * vx - yaw + 0.1
    action[1] = -0.2 * observation[0] + 0.5 * yaw + 0.05
    return {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.context_v1",
        "scenario_id": "flat_walk_varied_speed_v1",
        "rollout_id": f"rollout_{step // 8}",
        "skill_id": "walk_velocity",
        "step_index": step,
        "robot_time": step * 0.02,
        "robot_state": {"observation": observation},
        "desired_command": {"vx_mps": vx, "vy_mps": 0.0, "yaw_radps": yaw},
        "task_context": {"skill_id": "walk_velocity"},
        "environment_context": {"terrain_type": "flat"},
        "teacher_action": action,
    }


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


def _linear_profile(tmp_path: Path) -> tuple[Path, Path]:
    prepared = _prepared_dataset(tmp_path)
    trained = train_behavior_clone_baseline(prepared, output_dir=tmp_path / "run", ridge_lambda=0.0)
    assert trained.ok
    profile_path = tmp_path / "linear_bc_eval.yaml"
    create_linear_bc_profile(
        name="linear_bc_eval",
        model=trained.model_path,
        output_path=profile_path,
        robot_config_path="configs/robots/open_duck_mini_v2.yaml",
        description="Linear BC evaluation profile",
    )
    return profile_path, prepared


def _context_linear_profile(tmp_path: Path) -> tuple[Path, Path]:
    prepared = _context_prepared_dataset(tmp_path)
    trained = train_behavior_clone_baseline(
        prepared,
        output_dir=tmp_path / "context_run",
        ridge_lambda=0.0,
        input_mode=INPUT_MODE_CONTEXT_COMMAND_V1,
    )
    assert trained.ok
    assert trained.observation_size == 104
    profile_path = tmp_path / "context_linear_bc_eval.yaml"
    create_linear_bc_profile(
        name="context_linear_bc_eval",
        model=trained.model_path,
        output_path=profile_path,
        robot_config_path="configs/robots/open_duck_mini_v2.yaml",
        description="Context linear BC evaluation profile",
    )
    return profile_path, prepared


def test_evaluate_linear_behavior_clone_profile_writes_report_and_predictions(tmp_path: Path) -> None:
    profile_path, prepared = _linear_profile(tmp_path)

    result = evaluate_policy_profile(
        profile_path,
        prepared,
        output_dir=tmp_path / "evaluation",
        write_predictions=True,
        max_test_mae=1e-4,
        max_test_rmse=1e-4,
    )

    assert result.ok
    assert result.profile_name == "linear_bc_eval"
    assert result.model_kind == "linear_behavior_clone"
    assert result.splits["train"].sample_count == 8
    assert result.splits["test"].mae is not None
    assert result.splits["test"].mae < 1e-5
    assert Path(result.evaluation_path).exists()
    assert Path(result.report_path).exists()
    assert Path(result.prediction_paths["test"]).exists()
    payload = json.loads(Path(result.evaluation_path).read_text(encoding="utf-8"))
    assert payload["evaluation_type"] == "soridormi.offline_policy_evaluation.v1"


def test_evaluate_context_linear_behavior_clone_profile_uses_policy_input_mode(tmp_path: Path) -> None:
    profile_path, prepared = _context_linear_profile(tmp_path)

    result = evaluate_policy_profile(
        profile_path,
        prepared,
        output_dir=tmp_path / "context_evaluation",
        write_predictions=True,
        max_test_mae=1e-4,
        max_test_rmse=1e-4,
    )

    assert result.ok
    assert result.input_mode == INPUT_MODE_CONTEXT_COMMAND_V1
    assert result.policy_input_size == 104
    assert result.splits["test"].mae is not None
    assert result.splits["test"].mae < 1e-5
    payload = json.loads(Path(result.evaluation_path).read_text(encoding="utf-8"))
    assert payload["input_mode"] == INPUT_MODE_CONTEXT_COMMAND_V1
    assert payload["policy_input_size"] == 104
    prediction = json.loads(Path(result.prediction_paths["test"]).read_text(encoding="utf-8").splitlines()[0])
    assert len(prediction["observation"]) == 104


def test_evaluate_policy_profile_threshold_failure_is_reported(tmp_path: Path) -> None:
    profile_path, prepared = _linear_profile(tmp_path)

    result = evaluate_policy_profile(
        profile_path,
        prepared,
        output_dir=tmp_path / "evaluation_fail",
        max_train_mae=0.0,
    )

    assert not result.ok
    assert any("train.mae" in error for error in result.errors)
    assert Path(result.evaluation_path).exists()
    assert Path(result.report_path).exists()


def test_evaluate_policy_profile_reports_missing_prepared_manifest(tmp_path: Path) -> None:
    profile_path, _prepared = _linear_profile(tmp_path)

    result = evaluate_policy_profile(profile_path, tmp_path / "missing_prepared", output_dir=tmp_path / "bad_eval")

    assert not result.ok
    assert any("File not found" in error for error in result.errors)
    assert Path(result.evaluation_path).exists()
