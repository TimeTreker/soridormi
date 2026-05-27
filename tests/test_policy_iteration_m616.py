from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml

from soridormi_runtime import policy_iteration
from soridormi_runtime.policy_iteration import promote_trained_profile, run_policy_iteration_from_rollouts
from soridormi_runtime.policy_profiles import PolicyModelSpec, PolicyProfile


def test_promote_trained_profile_copies_and_renames(tmp_path: Path) -> None:
    source = tmp_path / "candidate.yaml"
    source.write_text(
        yaml.safe_dump({"name": "candidate", "description": "candidate", "model": {"path": "/data/model.onnx"}}),
        encoding="utf-8",
    )
    target = promote_trained_profile(source, target_profile_name="promoted", output_dir=tmp_path / "profiles")

    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["name"] == "promoted"
    assert payload["model"]["path"] == "/data/model.onnx"
    assert "Promoted from iteration profile candidate" in payload["description"]


def test_policy_iteration_orchestrates_relabel_retrain_evaluate_promote(tmp_path: Path, monkeypatch) -> None:
    teacher = PolicyProfile(
        name="teacher",
        description="teacher",
        path=tmp_path / "teacher.yaml",
        payload={},
        model=PolicyModelSpec(path="teacher.onnx"),
    )
    relabel_out = tmp_path / "out" / "relabel" / "teacher_relabel.jsonl"
    merged_out = tmp_path / "out" / "dataset" / "combined_supervised.jsonl"
    prepared_manifest = tmp_path / "out" / "prepared" / "prepared_manifest.json"
    stats_path = tmp_path / "out" / "prepared" / "dataset_stats.json"
    norm_path = tmp_path / "out" / "prepared" / "normalization.json"
    train_dir = tmp_path / "out" / "train"
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump({"name": "iter_candidate", "description": "candidate", "model": {"path": "/data/model.onnx"}}),
        encoding="utf-8",
    )
    eval_path = tmp_path / "out" / "evaluation" / "evaluation.json"

    monkeypatch.setattr(policy_iteration.PolicyProfile, "load", staticmethod(lambda value: teacher))
    monkeypatch.setattr(
        policy_iteration,
        "relabel_policy_rollouts_with_teacher",
        lambda *args, **kwargs: SimpleNamespace(ok=True, output_path=str(relabel_out), errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        policy_iteration,
        "merge_supervised_datasets",
        lambda *args, **kwargs: SimpleNamespace(ok=True, output_path=str(merged_out), errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        policy_iteration,
        "split_training_dataset",
        lambda *args, **kwargs: SimpleNamespace(ok=True, manifest_path=str(prepared_manifest), errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        policy_iteration,
        "analyze_prepared_training_dataset",
        lambda *args, **kwargs: SimpleNamespace(ok=True, stats_path=str(stats_path), normalization_path=str(norm_path), errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        policy_iteration,
        "train_neural_behavior_clone",
        lambda *args, **kwargs: SimpleNamespace(ok=True, output_dir=str(train_dir), profile_path=str(profile_path), errors=[], warnings=[]),
    )
    monkeypatch.setattr(
        policy_iteration,
        "evaluate_policy_profile",
        lambda *args, **kwargs: SimpleNamespace(ok=True, evaluation_path=str(eval_path), errors=[], warnings=[]),
    )

    result = run_policy_iteration_from_rollouts(
        iteration_name="iter_candidate",
        candidate_logs=[tmp_path / "candidate.mcap"],
        base_datasets=[tmp_path / "base.jsonl"],
        teacher_profile="teacher",
        output_root=tmp_path / "out",
        epochs=1,
        promote_to="promoted",
        force_promote=True,
        promote_output_dir=tmp_path / "promoted_profiles",
    )

    assert result.ok
    assert result.relabel_dataset_path == str(relabel_out)
    assert result.promoted_profile_path is not None
    promoted = yaml.safe_load(Path(result.promoted_profile_path).read_text(encoding="utf-8"))
    assert promoted["name"] == "promoted"
