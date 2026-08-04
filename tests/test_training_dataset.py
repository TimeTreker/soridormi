from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.training_dataset import export_training_dataset, load_training_records


def _runtime_step(step: int, *, observation_size: int = 101, action_size: int = 14) -> dict:
    return {
        "type": "runtime_step",
        "step_index": step,
        "time_wall_ns": 1_000_000_000 + step * 20_000_000,
        "robot_time": step * 0.02,
        "mode": "onnx_policy",
        "backend": "sim",
        "state": {
            "time": step * 0.02,
            "joints": {
                "names": ["j0", "j1"],
                "positions": [0.1 + step, -0.1 - step],
                "velocities": [0.2, -0.2],
                "torques": [0.0, 0.0],
            },
            "base_position_xyz": [0.01 * step, 0.0, 0.2],
            "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "command": {
            "names": ["j0", "j1"],
            "positions": [0.01 * step, -0.01 * step],
            "velocities": [0.0, 0.0],
            "kp": [10.0, 10.0],
            "kd": [0.5, 0.5],
            "torques": [0.0, 0.0],
        },
        "policy_observation": [float(step)] * observation_size,
        "policy_action": [0.1 * step] * action_size,
        "policy_raw_action": [0.2 * step] * action_size,
        "policy_debug": {
            "step_count": step,
            "robot_time": step * 0.02,
            "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "phase": [1.0, 0.0],
        },
    }


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n", encoding="utf-8")


def test_load_training_records_reads_policy_vectors_from_jsonl_runtime_steps(tmp_path: Path) -> None:
    log = tmp_path / "runtime.jsonl"
    _write_jsonl(log, [_runtime_step(0), _runtime_step(1)])

    records = load_training_records(log)

    assert len(records) == 2
    assert records[0].observation == [0.0] * 101
    assert records[1].action == [0.1] * 14
    assert records[1].raw_action == [0.2] * 14
    assert records[0].state is not None
    assert records[0].command is not None


def test_export_training_dataset_writes_samples_and_manifest(tmp_path: Path) -> None:
    log = tmp_path / "runtime.jsonl"
    output = tmp_path / "dataset.jsonl"
    manifest = tmp_path / "dataset.manifest.json"
    _write_jsonl(log, [_runtime_step(0), _runtime_step(1)])

    result = export_training_dataset([log], output_path=output, manifest_path=manifest)

    assert result.ok
    assert result.sample_count == 2
    assert result.skipped_records == 0
    assert output.exists()
    assert manifest.exists()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == 1
    assert rows[0]["sample_type"] == "soridormi.policy_supervision.v1"
    assert rows[0]["observation"] == [0.0] * 101
    assert rows[0]["action"] == [0.0] * 14
    assert rows[0]["policy_command"] == [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rows[0]["next_state"]["joint_positions"] == [1.1, -1.1]

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["sample_count"] == 2
    assert manifest_payload["dataset_sha256"] == result.dataset_sha256


def test_export_training_dataset_rejects_wrong_observation_size(tmp_path: Path) -> None:
    log = tmp_path / "runtime_bad.jsonl"
    output = tmp_path / "dataset.jsonl"
    _write_jsonl(log, [_runtime_step(0, observation_size=100)])

    result = export_training_dataset([log], output_path=output, strict=True)

    assert not result.ok
    assert result.sample_count == 0
    assert any("observation size 100" in error for error in result.errors)


def test_export_training_dataset_skips_records_without_policy_vectors(tmp_path: Path) -> None:
    log = tmp_path / "runtime_no_policy.jsonl"
    output = tmp_path / "dataset.jsonl"
    _write_jsonl(log, [{"type": "runtime_step", "step_index": 0, "robot_time": 0.0}])

    result = export_training_dataset([log], output_path=output)

    assert not result.ok
    assert result.sample_count == 0
    assert result.skipped_records == 1
    assert any("No training samples" in error for error in result.errors)
