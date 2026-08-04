import json

from soridormi_runtime.scripted_social_acceptance import run_acceptance
from soridormi_runtime.scripted_social_readiness import (
    build_readiness_report,
    render_markdown,
    write_report_outputs,
)
from soridormi_runtime.skill_manifest import load_skill_manifest


def test_dry_run_readiness_requires_live_before_promotion():
    manifest = load_skill_manifest()
    dry = run_acceptance(skill_ids=["shake_no"], execute=False, control_hz=20.0)

    report = build_readiness_report(manifest=manifest, dry_acceptance=dry)

    assert report.ok
    assert report.ready_count == 0
    row = report.skills[0]
    assert row.skill_id == "shake_no"
    assert row.dry_run_ok is True
    assert row.live_ok is None
    assert row.recommendation == "dry_run_ready_requires_live_acceptance"
    assert any("live MuJoCo acceptance" in warning for warning in row.warnings)


def test_require_live_blocks_when_live_json_missing():
    manifest = load_skill_manifest()
    dry = run_acceptance(skill_ids=["shake_no"], execute=False, control_hz=20.0)

    report = build_readiness_report(manifest=manifest, dry_acceptance=dry, require_live=True)

    assert not report.ok
    row = report.skills[0]
    assert row.recommendation == "keep_available_sim_experimental"
    assert "live MuJoCo acceptance JSON is required but missing" in row.blockers


def test_live_pass_marks_candidate_for_available_sim():
    manifest = load_skill_manifest()
    dry = run_acceptance(skill_ids=["shake_no"], execute=False, control_hz=20.0)
    live = dry.to_dict()
    live["executed"] = True
    live["results"][0]["executed"] = True
    live["results"][0]["ok"] = True
    live["results"][0]["fallen"] = False
    live["results"][0]["warnings"] = []
    live["results"][0]["errors"] = []

    report = build_readiness_report(
        manifest=manifest,
        dry_acceptance=dry,
        live_acceptance=live,
        require_live=True,
    )

    assert report.ok
    assert report.ready_count == 1
    row = report.skills[0]
    assert row.recommendation == "candidate_for_available_sim"
    assert row.blockers == []


def test_live_fall_blocks_promotion():
    manifest = load_skill_manifest()
    dry = run_acceptance(skill_ids=["shake_no"], execute=False, control_hz=20.0)
    live = dry.to_dict()
    live["executed"] = True
    live["results"][0]["executed"] = True
    live["results"][0]["ok"] = True
    live["results"][0]["fallen"] = True

    report = build_readiness_report(
        manifest=manifest,
        dry_acceptance=dry,
        live_acceptance=live,
        require_live=True,
    )

    assert not report.ok
    row = report.skills[0]
    assert row.recommendation == "keep_available_sim_experimental"
    assert "live MuJoCo acceptance reported a fall" in row.blockers


def test_report_outputs_json_and_markdown(tmp_path):
    manifest = load_skill_manifest()
    dry = run_acceptance(skill_ids=["neutral_head"], execute=False, control_hz=20.0)
    report = build_readiness_report(manifest=manifest, dry_acceptance=dry)

    json_path, md_path = write_report_outputs(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    markdown = md_path.read_text(encoding="utf-8")
    assert "Soridormi scripted social readiness report" in markdown
    assert "neutral_head" in markdown
    assert render_markdown(report).startswith("# Soridormi scripted social readiness report")
