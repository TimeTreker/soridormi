from pathlib import Path

from soridormi_runtime.training_pipeline import build_training_pipeline_plan, run_training_pipeline


def test_training_pipeline_plan_contains_end_to_end_steps() -> None:
    plan = build_training_pipeline_plan(
        candidate_profile="linear_bc_candidate",
        logs=["data/logs/policy_open_duck_forward.mcap"],
        output_root="data/training_pipelines/linear_bc_candidate",
        max_test_mae=0.05,
        include_model=True,
        check_model=True,
        require_model=True,
        force_profile=True,
    )

    assert [step.name for step in plan.steps] == [
        "export_dataset",
        "prepare_dataset",
        "summarize_dataset",
        "train_linear_bc",
        "create_profile",
        "evaluate_profile",
        "accept_profile",
        "package_profile",
    ]
    assert plan.dataset_jsonl.endswith("/dataset/supervised.jsonl")
    assert any("--max-test-mae" in step.command for step in plan.steps if step.name == "evaluate_profile")
    assert any("--include-model" in step.command for step in plan.steps if step.name == "package_profile")
    assert any("--force" in step.command for step in plan.steps if step.name == "create_profile")


def test_training_pipeline_dry_run_writes_plan(tmp_path: Path) -> None:
    plan = build_training_pipeline_plan(
        candidate_profile="candidate",
        logs=["data/logs/source.mcap"],
        output_root=tmp_path / "candidate_pipeline",
    )

    result = run_training_pipeline(plan, dry_run=True)

    assert result.ok
    assert result.completed_steps == []
    assert result.plan_path is not None
    assert result.report_path is not None
    plan_text = Path(result.plan_path).read_text(encoding="utf-8")
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert '"pipeline_kind": "linear_behavior_clone"' in plan_text
    assert "./scripts/train_behavior_clone.sh" in report_text
