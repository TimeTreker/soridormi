from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.pre_wbc_scenario_surface import (
    build_pre_wbc_scenario_surface_report,
    render_markdown,
)
from soridormi_runtime.wbc_clearance_contract import DEFAULT_CONTRACT_PATH


EXPECTED_SURFACE = [
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
    "startup_tail_clearance_v1",
    "s_turn_reversal_v1",
    "turn_stop_settle_v1",
]


def _contract_payload() -> dict:
    return json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_pre_wbc_surface_matches_default_suite_and_contract() -> None:
    report = build_pre_wbc_scenario_surface_report()

    assert report.ok
    assert report.status == "PRE_WBC_SCENARIO_SURFACE_READY"
    assert report.wbc_scenario_ids == EXPECTED_SURFACE
    assert report.default_suite_scenario_ids == EXPECTED_SURFACE
    assert report.clearance_core_scenarios == EXPECTED_SURFACE[:3]
    assert report.enrichment_scenarios == EXPECTED_SURFACE[3:]
    assert {item["role"] for item in report.selected_scenarios} == {
        "clearance_core",
        "wbc_enrichment",
    }
    assert all(item["run_plan"] for item in report.selected_scenarios)
    assert any("WBC runtime backend is not implemented" in warning for warning in report.warnings)


def test_pre_wbc_surface_rejects_contract_without_enrichment(tmp_path: Path) -> None:
    payload = _contract_payload()
    payload["scenario_ids"] = EXPECTED_SURFACE[:3]
    contract_path = tmp_path / "core_only_wbc_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_pre_wbc_scenario_surface_report(contract_path=contract_path)

    assert not report.ok
    assert report.status == "PRE_WBC_SCENARIO_SURFACE_BLOCKED"
    assert any("at least 3 pre-WBC enrichment scenarios" in item for item in report.blockers)
    assert any("Default ready locomotion suite" in item for item in report.blockers)


def test_pre_wbc_surface_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "surface"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.pre_wbc_scenario_surface",
            "--output-dir",
            str(output_dir),
            "--json",
            "--strict",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "PRE_WBC_SCENARIO_SURFACE_READY"
    assert payload["wbc_scenario_ids"] == EXPECTED_SURFACE
    assert (output_dir / "pre_wbc_scenario_surface_report.json").exists()
    assert "Scenario surface" in (
        output_dir / "pre_wbc_scenario_surface_report.md"
    ).read_text(encoding="utf-8")


def test_pre_wbc_surface_markdown_names_enrichment() -> None:
    rendered = render_markdown(build_pre_wbc_scenario_surface_report())

    assert "Pre-WBC enrichment" in rendered
    assert "startup_tail_clearance_v1" in rendered
    assert "s_turn_reversal_v1" in rendered
    assert "turn_stop_settle_v1" in rendered


def test_pre_wbc_surface_wrapper_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "validate"
    result = subprocess.run(
        [
            "bash",
            "scripts/validate_pre_wbc_scenario_surface.sh",
            "--output-dir",
            str(output_dir),
            "--skip-pytest",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Pre-WBC scenario-surface validation: PASS" in result.stdout
    assert (output_dir / "surface" / "pre_wbc_scenario_surface_report.json").exists()


def test_pre_wbc_surface_script_does_not_launch_sim_or_training() -> None:
    source = Path("scripts/validate_pre_wbc_scenario_surface.sh").read_text(encoding="utf-8")

    assert "--dry-run-only" in source
    assert "run_sim_server.sh" not in source
    assert "train_clearance_residual_policy.sh" not in source
    assert "hardware" in source.lower()
