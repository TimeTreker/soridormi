from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.m10_visual_inspection import (
    build_m10_visual_inspection_plan,
    render_markdown,
)


def _write_readiness(path: Path, *, ok: bool, gate_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": ok,
                "gate_status": gate_status,
                "profile": "candidate",
                "scenarios": [],
                "summary_metrics": {},
                "blockers": [] if ok else ["clearance failed"],
            }
        ),
        encoding="utf-8",
    )


def test_visual_inspection_plan_includes_follow_camera_and_required_scenarios(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    _write_readiness(readiness_path, ok=True, gate_status="READY_FOR_VISUAL_INSPECTION")

    plan = build_m10_visual_inspection_plan(
        profile="candidate",
        output_dir=tmp_path / "visual",
        readiness_report=readiness_path,
        require_clearance_ready=True,
    )

    assert plan.ok
    assert plan.status == "READY_FOR_VISUAL_INSPECTION_PLAN"
    assert "--follow-camera" in plan.sim_server_command
    assert "--viewer" in plan.sim_server_command
    assert plan.camera["distance"] == 1.4
    assert [item["scenario_id"] for item in plan.scenarios] == [
        "flat_walk_varied_speed_v1",
        "start_stop_velocity_ramp_v1",
        "curve_turn_walk_v1",
    ]
    for item in plan.scenarios:
        assert "./scripts/evaluate_scenario_rollout.sh" == item["rollout_command"][0]
        assert "--json" in item["rollout_command"]
        assert item["visual_checklist"]


def test_visual_inspection_readiness_command_targets_explicit_report_path(tmp_path: Path) -> None:
    readiness_path = tmp_path / "clearance_readiness" / "custom_readiness.json"

    plan = build_m10_visual_inspection_plan(
        profile="candidate",
        output_dir=tmp_path / "visual",
        readiness_report=readiness_path,
    )

    assert plan.readiness_report == str(readiness_path)
    assert "--output-dir" in plan.readiness_command
    assert plan.readiness_command[plan.readiness_command.index("--output-dir") + 1] == str(readiness_path.parent)
    assert "--json-output" in plan.readiness_command
    assert plan.readiness_command[plan.readiness_command.index("--json-output") + 1] == str(readiness_path)

def test_visual_inspection_plan_blocks_when_readiness_required_and_missing(tmp_path: Path) -> None:
    plan = build_m10_visual_inspection_plan(
        profile="candidate",
        output_dir=tmp_path / "visual",
        readiness_report=tmp_path / "missing.json",
        require_clearance_ready=True,
    )

    assert not plan.ok
    assert plan.status == "BLOCKED_BY_CLEARANCE_READINESS"
    assert any("missing" in blocker for blocker in plan.blockers)


def test_visual_inspection_markdown_contains_commands_and_checklist(tmp_path: Path) -> None:
    plan = build_m10_visual_inspection_plan(
        profile="candidate",
        output_dir=tmp_path / "visual",
        scenarios=["curve_turn_walk_v1"],
        camera_distance=2.0,
        camera_azimuth=90.0,
        camera_elevation=-15.0,
        duration_s=1.5,
        steps=75,
    )

    rendered = render_markdown(plan)

    assert "Soridormi M10 visual inspection plan" in rendered
    assert "./scripts/run_sim_server.sh" in rendered
    assert "--follow-camera" in rendered
    assert "--camera-distance 2" in rendered
    assert "curve_turn_walk_v1" in rendered
    assert "--duration-s 1.5" in rendered
    assert "--steps 75" in rendered
    assert "Inspect inside and outside feet separately" in rendered


def test_m10_visual_inspection_cli_functionally_writes_plan_artifacts(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    output_dir = tmp_path / "visual_plan"
    _write_readiness(readiness_path, ok=True, gate_status="READY_FOR_VISUAL_INSPECTION")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.m10_visual_inspection",
            "--profile-name",
            "candidate",
            "--output-dir",
            str(output_dir),
            "--readiness-report",
            str(readiness_path),
            "--require-clearance-ready",
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
    assert payload["status"] == "READY_FOR_VISUAL_INSPECTION_PLAN"
    assert "--follow-camera" in payload["sim_server_command"]
    assert payload["readiness_command"][payload["readiness_command"].index("--output-dir") + 1] == str(readiness_path.parent)
    assert payload["readiness_command"][payload["readiness_command"].index("--json-output") + 1] == str(readiness_path)
    assert (output_dir / "m10_visual_inspection_plan.json").exists()
    assert "Start MuJoCo follow-camera server" in (output_dir / "m10_visual_inspection_plan.md").read_text()


def test_m10_visual_inspection_script_functionally_blocks_failed_readiness(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness" / "m10_clearance_readiness.json"
    output_dir = tmp_path / "blocked_visual_plan"
    _write_readiness(readiness_path, ok=False, gate_status="BLOCKED_BY_CLEARANCE_GATE")

    result = subprocess.run(
        [
            "bash",
            "scripts/plan_m10_visual_inspection.sh",
            "--profile-name",
            "candidate",
            "--output-dir",
            str(output_dir),
            "--readiness-report",
            str(readiness_path),
            "--require-clearance-ready",
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
    assert (output_dir / "m10_visual_inspection_plan.json").exists()
    assert (output_dir / "m10_visual_inspection_plan.md").exists()
