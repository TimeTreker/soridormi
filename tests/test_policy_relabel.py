from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from soridormi_runtime import policy_relabel
from soridormi_runtime.policy_relabel import merge_supervised_datasets, relabel_policy_rollouts_with_teacher
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile


def _profile(tmp_path: Path) -> PolicyProfile:
    path = tmp_path / "teacher.yaml"
    return PolicyProfile(
        name="teacher",
        description="test teacher",
        path=path,
        payload={},
        model=PolicyModelSpec(path=str(tmp_path / "teacher.onnx")),
    )


def _write_candidate_log(path: Path) -> None:
    obs0 = [0.1] * 101
    obs1 = [0.2] * 101
    action0 = [0.0] * 14
    action1 = [0.1] * 14
    rows = [
        {"type": "policy_observation", "step_index": 0, "robot_time": 0.0, "observation": obs0},
        {"type": "policy_action", "step_index": 0, "robot_time": 0.0, "action": action0},
        {"type": "robot_state", "step_index": 0, "robot_time": 0.0, "state": {"base_position_xyz": [0, 0, 0]}},
        {"type": "policy_observation", "step_index": 1, "robot_time": 0.02, "observation": obs1},
        {"type": "policy_action", "step_index": 1, "robot_time": 0.02, "action": action1},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_relabel_rollout_uses_teacher_actions(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "candidate.jsonl"
    _write_candidate_log(log)
    teacher = _profile(tmp_path)

    def fake_predict(profile, observations, **kwargs):
        assert profile.name == "teacher"
        assert observations.shape == (2, 101)
        return np.asarray([[0.5] * 14, [0.6] * 14], dtype=np.float64), [], [], "sha"

    monkeypatch.setattr(policy_relabel, "_predict_profile", fake_predict)

    output = tmp_path / "relabel.jsonl"
    result = relabel_policy_rollouts_with_teacher([log], teacher_profile=teacher, output_path=output)

    assert result.ok
    assert result.relabeled_sample_count == 2
    samples = [json.loads(line) for line in output.read_text().splitlines()]
    assert samples[0]["action"] == [0.5] * 14
    assert samples[0]["source_policy_action"] == [0.0] * 14
    assert samples[1]["action"] == [0.6] * 14
    assert result.mean_abs_teacher_delta is not None


def test_merge_supervised_datasets_deduplicates(tmp_path: Path) -> None:
    sample = {
        "schema_version": 1,
        "sample_type": "soridormi.policy_supervision.v1",
        "observation": [0.0] * 101,
        "action": [0.0] * 14,
    }
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    b.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    output = tmp_path / "merged.jsonl"
    result = merge_supervised_datasets([a, b], output_path=output)

    assert result.ok
    assert result.sample_count == 1
    assert result.deduplicated_count == 1
    assert len(output.read_text().splitlines()) == 1
