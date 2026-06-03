from __future__ import annotations

import json
import subprocess
from pathlib import Path

from soridormi_runtime.bc_training_contract import load_and_validate_contract, validate_sample_jsonl
from soridormi_runtime.context_bc_dataset_prepare import prepare_context_bc_dataset


def _sample(
    scenario_id: str,
    rollout_id: str,
    step: int,
    *,
    skill_id: str = "walk_velocity",
) -> dict:
    return {
        "sample_type": "soridormi.policy_supervision.context_v1",
        "schema_version": 1,
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "timestep": step,
        "step_index": step,
        "episode_index": int(rollout_id.rsplit("_", 1)[-1]) if rollout_id.rsplit("_", 1)[-1].isdigit() else 0,
        "skill_id": skill_id,
        "robot_state": {"observation": [float(step)] * 101},
        "desired_command": {"vx_mps": 0.10, "vy_mps": 0.0, "yaw_radps": 0.02},
        "applied_command": {"vx_mps": 0.09, "vy_mps": 0.0, "yaw_radps": 0.02},
        "task_context": {"skill_id": skill_id, "gait_style": "default_walk"},
        "environment_context": {"terrain_type": "flat", "obstacle_height_m": 0.0},
        "command_ramp_alpha": 1.0,
        "teacher_action": [0.01 * step] * 14,
        "failure_flags": {"fallen": False, "stuck": False, "terminated": False},
        "short_history": {
            "previous_action": [0.0] * 14,
            "previous_command": {"vx_mps": 0.0, "vy_mps": 0.0, "yaw_radps": 0.0},
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_prepare_context_bc_dataset_splits_by_rollout_without_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    rows: list[dict] = []
    for scenario_id in ("flat_walk_varied_speed_v1", "curve_turn_walk_v1"):
        for rollout in range(4):
            rollout_id = f"{scenario_id}_rollout_{rollout}"
            for step in range(2):
                rows.append(_sample(scenario_id, rollout_id, rollout * 10 + step))
    _write_jsonl(dataset, rows)

    result = prepare_context_bc_dataset(
        [dataset],
        output_dir=tmp_path / "prepared",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    assert result.ok
    assert result.sample_count == 16
    assert result.valid_sample_count == 16
    assert result.split_group_field == "rollout_id"
    assert result.stratify_by_scenario is True
    assert result.split_group_counts == {"train": 4, "val": 2, "test": 2}
    assert result.train.sample_count == 8
    assert result.val.sample_count == 4
    assert result.test.sample_count == 4

    split_rollouts = {}
    for split in (result.train, result.val, result.test):
        split_rollouts[split.name] = {row["rollout_id"] for row in _read_jsonl(Path(split.path))}
    assert split_rollouts["train"].isdisjoint(split_rollouts["val"])
    assert split_rollouts["train"].isdisjoint(split_rollouts["test"])
    assert split_rollouts["val"].isdisjoint(split_rollouts["test"])

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "soridormi.policy_supervision.context_prepared.v1"
    assert manifest["splits"]["train"]["sample_count"] == 8
    assert manifest["splits"]["val"]["scenario_counts"] == {"curve_turn_walk_v1": 2, "flat_walk_varied_speed_v1": 2}


def test_prepare_context_bc_dataset_outputs_validate_against_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    rows = []
    for rollout in range(3):
        for step in range(3):
            rows.append(_sample("flat_walk_varied_speed_v1", f"rollout_{rollout}", rollout * 10 + step))
    _write_jsonl(dataset, rows)

    result = prepare_context_bc_dataset([dataset], output_dir=tmp_path / "prepared", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert result.ok
    contract, contract_result = load_and_validate_contract()
    assert contract_result.ok
    assert contract is not None
    for split in (result.train, result.val, result.test):
        validation = validate_sample_jsonl(split.path, contract)
        assert validation.ok
        assert validation.context_sample_count == split.sample_count


def test_prepare_context_bc_dataset_reports_invalid_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    bad = _sample("flat_walk_varied_speed_v1", "rollout_0", 0)
    bad["teacher_action"] = [0.0] * 13
    _write_jsonl(dataset, [bad])

    result = prepare_context_bc_dataset([dataset], output_dir=tmp_path / "prepared")

    assert not result.ok
    assert result.invalid_sample_count == 1
    assert result.valid_sample_count == 0
    assert any("teacher_action size 13" in error for error in result.errors)


def test_prepare_context_bc_dataset_can_skip_invalid_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    bad = _sample("flat_walk_varied_speed_v1", "rollout_bad", 0)
    bad["teacher_action"] = [0.0] * 13
    good = _sample("flat_walk_varied_speed_v1", "rollout_good", 1)
    _write_jsonl(dataset, [bad, good])

    result = prepare_context_bc_dataset([dataset], output_dir=tmp_path / "prepared", skip_invalid=True)

    assert result.ok
    assert result.valid_sample_count == 1
    assert result.invalid_sample_count == 1
    assert result.skipped_invalid_count == 1
    assert result.train.sample_count == 1


def test_prepare_context_bc_dataset_rejects_bad_ratios(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    _write_jsonl(dataset, [_sample("flat_walk_varied_speed_v1", "rollout_0", 0)])

    result = prepare_context_bc_dataset([dataset], output_dir=tmp_path / "prepared", train_ratio=0.7, val_ratio=0.7, test_ratio=0.0)

    assert not result.ok
    assert any("sum to 1.0" in error for error in result.errors)


def test_prepare_context_bc_dataset_cli_json(tmp_path: Path) -> None:
    dataset = tmp_path / "context.jsonl"
    _write_jsonl(dataset, [_sample("flat_walk_varied_speed_v1", "rollout_0", 0), _sample("flat_walk_varied_speed_v1", "rollout_1", 1)])

    result = subprocess.run(
        [
            "python",
            "-m",
            "soridormi_runtime.context_bc_dataset_prepare",
            str(dataset),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["valid_sample_count"] == 2
    assert Path(payload["manifest_path"]).exists()
