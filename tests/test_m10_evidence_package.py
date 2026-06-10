from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.m10_evidence_package import (
    build_m10_evidence_package,
    build_visual_review_template,
    render_markdown,
)
from soridormi_runtime.m10_clearance_readiness import DEFAULT_REQUIRED_SCENARIOS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _readiness_payload(tmp_path: Path, *, ok: bool = True) -> dict:
    scenarios = []
    for scenario_id in DEFAULT_REQUIRED_SCENARIOS:
        report_path = tmp_path / "rollouts" / scenario_id / "scenario_rollout_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"scenario_id": scenario_id, "ok": True}), encoding="utf-8")
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "report_path": str(report_path),
                "status": "PASS" if ok else "FAIL_CLEARANCE_GATE",
                "clearance_ok": ok,
                "scenario_acceptance_ok": ok,
            }
        )
    return {
        "ok": ok,
        "gate_status": "READY_FOR_VISUAL_INSPECTION" if ok else "BLOCKED_BY_CLEARANCE_GATE",
        "profile": "candidate",
        "scenarios": scenarios,
        "summary_metrics": {},
        "blockers": [] if ok else ["clearance failed"],
    }


def _visual_plan_payload(*, ok: bool = True) -> dict:
    return {
        "ok": ok,
        "status": "READY_FOR_VISUAL_INSPECTION_PLAN" if ok else "BLOCKED_BY_CLEARANCE_READINESS",
        "profile": "candidate",
        "scenarios": [{"scenario_id": scenario_id} for scenario_id in DEFAULT_REQUIRED_SCENARIOS],
        "blockers": [] if ok else ["visual blocked"],
    }


def _visual_review_payload(*, passing: bool = True) -> dict:
    value = "PASS" if passing else "FAIL"
    return {
        "profile": "candidate",
        "status": "VISUAL_PASS" if passing else "VISUAL_FAIL",
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "foot_clearance": value,
                "base_stability": "PASS",
                "toe_drag": "PASS",
                "command_transition_quality": "PASS",
                "notes": "",
            }
            for scenario_id in DEFAULT_REQUIRED_SCENARIOS
        ],
    }


def test_evidence_package_ready_for_visual_review_when_quantitative_and_plan_pass(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    visual_plan_path = tmp_path / "visual" / "m10_visual_inspection_plan.json"
    _write_json(readiness_path, _readiness_payload(tmp_path, ok=True))
    _write_json(visual_plan_path, _visual_plan_payload(ok=True))

    package = build_m10_evidence_package(
        profile="candidate",
        output_dir=tmp_path / "evidence",
        readiness_report=readiness_path,
        visual_plan=visual_plan_path,
    )

    assert package.ok
    assert package.status == "READY_FOR_VISUAL_REVIEW"
    assert package.readiness_ok
    assert package.visual_plan_ok
    assert package.visual_review_ok is None
    assert package.next_steps
    assert package.commands["analyze_clearance_readiness"][0] == "./scripts/analyze_m10_clearance_readiness.sh"


def test_evidence_package_ready_for_teacher_comparison_after_visual_pass(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    visual_plan_path = tmp_path / "visual" / "m10_visual_inspection_plan.json"
    visual_review_path = tmp_path / "evidence" / "m10_visual_review.json"
    _write_json(readiness_path, _readiness_payload(tmp_path, ok=True))
    _write_json(visual_plan_path, _visual_plan_payload(ok=True))
    _write_json(visual_review_path, _visual_review_payload(passing=True))

    package = build_m10_evidence_package(
        profile="candidate",
        output_dir=tmp_path / "evidence",
        readiness_report=readiness_path,
        visual_plan=visual_plan_path,
        visual_review=visual_review_path,
        require_visual_pass=True,
    )

    assert package.ok
    assert package.status == "READY_FOR_TEACHER_COMPARISON"
    assert package.visual_review_ok is True
    assert package.blockers == []


def test_evidence_package_blocks_failed_visual_review_when_required(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    visual_plan_path = tmp_path / "visual" / "m10_visual_inspection_plan.json"
    visual_review_path = tmp_path / "evidence" / "m10_visual_review.json"
    _write_json(readiness_path, _readiness_payload(tmp_path, ok=True))
    _write_json(visual_plan_path, _visual_plan_payload(ok=True))
    _write_json(visual_review_path, _visual_review_payload(passing=False))

    package = build_m10_evidence_package(
        profile="candidate",
        output_dir=tmp_path / "evidence",
        readiness_report=readiness_path,
        visual_plan=visual_plan_path,
        visual_review=visual_review_path,
        require_visual_pass=True,
    )

    assert not package.ok
    assert package.status == "BLOCKED_BY_VISUAL_REVIEW"
    assert any("foot_clearance" in blocker for blocker in package.blockers)


def test_visual_review_template_and_markdown_cover_required_scenarios() -> None:
    template = build_visual_review_template(profile="candidate", scenarios=DEFAULT_REQUIRED_SCENARIOS)
    rendered = render_markdown(
        build_m10_evidence_package(
            profile="candidate",
            output_dir="artifacts/tmp",
            require_clearance_ready=False,
            require_visual_plan=False,
        )
    )

    assert [item["scenario_id"] for item in template["scenarios"]] == list(DEFAULT_REQUIRED_SCENARIOS)
    assert "foot_clearance" in template["scenarios"][0]
    assert "Soridormi M10 evidence package" in rendered
    assert "m10_visual_review_template.json" in rendered


def test_m10_evidence_package_cli_functionally_writes_manifest_and_templates(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    visual_plan_path = tmp_path / "visual" / "m10_visual_inspection_plan.json"
    output_dir = tmp_path / "evidence"
    _write_json(readiness_path, _readiness_payload(tmp_path, ok=True))
    _write_json(visual_plan_path, _visual_plan_payload(ok=True))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.m10_evidence_package",
            "--profile-name",
            "candidate",
            "--output-dir",
            str(output_dir),
            "--readiness-report",
            str(readiness_path),
            "--visual-plan",
            str(visual_plan_path),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "READY_FOR_VISUAL_REVIEW"
    assert (output_dir / "m10_evidence_package.json").exists()
    assert (output_dir / "m10_evidence_package.md").exists()
    assert (output_dir / "m10_visual_review_template.json").exists()
    assert "Promotion rule" in (output_dir / "m10_visual_review_template.md").read_text()


def test_m10_evidence_package_script_functionally_blocks_missing_readiness(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"

    result = subprocess.run(
        [
            "bash",
            "scripts/build_m10_evidence_package.sh",
            "--profile-name",
            "candidate",
            "--output-dir",
            str(output_dir),
            "--readiness-report",
            str(tmp_path / "missing_readiness.json"),
            "--visual-plan",
            str(tmp_path / "missing_visual_plan.json"),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCKED_BY_CLEARANCE_READINESS"
    assert payload["blockers"]
    assert (output_dir / "m10_evidence_package.json").exists()
    assert (output_dir / "m10_visual_review_template.json").exists()
