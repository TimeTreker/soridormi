from __future__ import annotations

import json
import subprocess
import sys

from soridormi_runtime.mcp.task_contract_demo import build_demo, render_summary


def _cases_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(case["name"]): case
        for case in payload["cases"]  # type: ignore[index]
    }


def test_task_contract_demo_covers_key_brain_body_examples() -> None:
    payload = build_demo()
    cases = _cases_by_name(payload)

    assert payload["schema_version"] == "soridormi.task_contract_demo.v1"
    assert payload["no_motion"] is True
    assert payload["boundary"]["raw_language_to_low_level_policy"] is False
    assert payload["boundary"]["raw_motor_or_action_14d_from_chromie"] is False

    navigation = cases["walk forward to the house previews as blocked navigation"]
    assert navigation["user_command"] == "Walk forward to the house."
    assert navigation["chromie_boundary"]["soridormi_receives_raw_language"] is False
    assert navigation["chromie_boundary"]["structured_task"]["task_type"] == (
        "navigate_to_location"
    )
    assert navigation["accepted"] is False
    assert navigation["reason_code"] == "missing_navigation_pipeline"
    assert navigation["recommended_actions"] == [
        "report_blocked_or_clarify",
        "do_not_lower_to_velocity_recipe",
    ]

    sequence = cases["turn left then nod twice compiles to skill sequence dry-run"]
    assert sequence["status"] == "completed"
    assert sequence["execution_mode"] == "skill_sequence_dry_run"
    assert sequence["no_motion"] is True
    assert sequence["event_types"] == [
        "task_accepted",
        "task_resolving",
        "task_planning",
        "task_executing",
        "task_monitoring",
        "task_completed",
    ]

    delivery = cases["bring me some water fails closed on missing manipulation"]
    assert delivery["reason_code"] == "missing_manipulation_capability"
    assert "manipulation_capability" in delivery["blocked_subsystems"]

    stop = cases["stop now redirects to dedicated safety tools"]
    assert stop["reason_code"] == "use_safety_tool_for_immediate_stop"
    assert stop["recommended_actions"] == [
        "call_dedicated_stop_tool",
        "do_not_resubmit_as_task",
    ]


def test_task_contract_demo_summary_is_human_readable() -> None:
    payload = build_demo()
    summary = render_summary(payload)

    assert "Soridormi task MCP contract demo" in summary
    assert "Boundary:" in summary
    assert "user_command=Walk forward to the house." in summary
    assert "chromie_structured_task=navigate_to_location" in summary
    assert "no_motion=true" in summary
    assert "missing_navigation_pipeline" in summary
    assert "skill_sequence=turn_in_place, nod_yes" in summary
    assert "call_dedicated_stop_tool" in summary


def test_task_contract_demo_cli_outputs_summary_by_default() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.mcp.task_contract_demo",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout.startswith("Soridormi task MCP contract demo\n")
    assert "walk forward to the house" in completed.stdout
    assert "no_motion=true" in completed.stdout


def test_task_contract_demo_cli_outputs_json_when_requested() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.mcp.task_contract_demo",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "soridormi.task_contract_demo.v1"
    assert len(payload["cases"]) == 4


def test_task_contract_demo_cli_outputs_compact_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.mcp.task_contract_demo",
            "--compact",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "soridormi.task_contract_demo.v1"
    assert "\n  " not in completed.stdout
