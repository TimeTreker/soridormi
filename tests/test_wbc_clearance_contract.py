from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from soridormi_runtime.wbc_clearance_contract import (
    DEFAULT_CONTRACT_PATH,
    build_wbc_clearance_experiment_plan,
)

WBC_CLEARANCE_SCENARIOS = [
    "flat_walk_varied_speed_v1",
    "start_stop_velocity_ramp_v1",
    "curve_turn_walk_v1",
    "startup_tail_clearance_v1",
    "s_turn_reversal_v1",
    "turn_stop_settle_v1",
]


def _contract_payload() -> dict:
    return json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_default_wbc_clearance_contract_is_sim_only_and_backend_blocked() -> None:
    plan = build_wbc_clearance_experiment_plan()

    assert plan.ok
    assert plan.status == "READY_FOR_WBC_BACKEND_IMPLEMENTATION"
    assert plan.sim_only is True
    assert plan.hardware_allowed is False
    assert plan.raw_action_14d_allowed is False
    assert plan.chromie_raw_control_allowed is False
    assert plan.candidate_count == 4
    assert all(item["status"] == "WAITING_FOR_WBC_RUNTIME_BACKEND" for item in plan.candidates)
    assert all(item["scenario_ids"] == WBC_CLEARANCE_SCENARIOS for item in plan.candidates)
    assert "WBC runtime backend is not implemented yet." in plan.warnings


def test_wbc_clearance_contract_rejects_out_of_bounds_candidate(tmp_path: Path) -> None:
    payload = _contract_payload()
    payload["candidate_sets"][0]["values"]["target_clearance_m"] = 0.10
    contract_path = tmp_path / "bad_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = build_wbc_clearance_experiment_plan(contract_path=contract_path)

    assert not plan.ok
    assert plan.status == "INVALID_CONTRACT"
    assert any("target_clearance_m" in blocker for blocker in plan.blockers)


def test_wbc_clearance_contract_rejects_hardware_or_raw_action_control(tmp_path: Path) -> None:
    payload = _contract_payload()
    payload["safety"]["hardware_allowed"] = True
    payload["safety"]["raw_action_14d_allowed"] = True
    payload["safety"]["chromie_raw_control_allowed"] = True
    contract_path = tmp_path / "unsafe_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = build_wbc_clearance_experiment_plan(contract_path=contract_path)

    assert not plan.ok
    assert any("hardware_allowed" in blocker for blocker in plan.blockers)
    assert any("raw_action_14d_allowed" in blocker for blocker in plan.blockers)
    assert any("chromie_raw_control_allowed" in blocker for blocker in plan.blockers)


def test_wbc_clearance_contract_fills_missing_candidate_values_from_defaults(
    tmp_path: Path,
) -> None:
    payload = _contract_payload()
    candidate = copy.deepcopy(payload["candidate_sets"][0])
    candidate["id"] = "partial"
    candidate["values"] = {"target_clearance_m": 0.023}
    payload["candidate_sets"] = [candidate]
    contract_path = tmp_path / "partial_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = build_wbc_clearance_experiment_plan(contract_path=contract_path)

    assert plan.ok
    values = plan.candidates[0]["parameter_values"]
    assert values["target_clearance_m"] == 0.023
    assert values["step_length_scale"] == payload["parameters"]["step_length_scale"]["default"]


def test_wbc_clearance_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.wbc_clearance_contract",
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

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "READY_FOR_WBC_BACKEND_IMPLEMENTATION"
    assert payload["candidate_count"] == 4
    assert (output_dir / "wbc_clearance_experiment_plan.json").exists()
    assert "Post-implementation commands" in (
        output_dir / "wbc_clearance_experiment_plan.md"
    ).read_text(encoding="utf-8")


def test_wbc_clearance_wrapper_and_validation_script_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "validate"
    result = subprocess.run(
        [
            "bash",
            "scripts/validate_wbc_clearance_contract.sh",
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
    assert "WBC clearance contract validation: PASS" in result.stdout
    assert (output_dir / "plan" / "wbc_clearance_experiment_plan.json").exists()


def test_wbc_clearance_scripts_parse_and_do_not_launch_sim_or_train() -> None:
    repo = Path(__file__).resolve().parents[1]
    for script_name in (
        "plan_wbc_clearance_experiment.sh",
        "validate_wbc_clearance_contract.sh",
    ):
        script = repo / "scripts" / script_name
        source = script.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert "run_sim_server.sh" not in source
        assert "train_clearance_residual_policy.sh" not in source
