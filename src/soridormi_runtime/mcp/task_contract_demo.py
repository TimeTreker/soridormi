from __future__ import annotations

import argparse
import json
from typing import Any

from .local_tools import SoridormiLocalToolService


INTERESTING_TASK_TYPES = [
    "move_velocity",
    "skill_sequence",
    "navigate_to_location",
    "speak_while_moving",
    "stop_now",
    "deliver_object",
]

DEMO_CASES: list[dict[str, Any]] = [
    {
        "name": "walk forward to the house previews as blocked navigation",
        "user_command": "Walk forward to the house.",
        "tool": "soridormi.task.preview",
        "task": {
            "task_type": "navigate_to_location",
            "summary": "walk forward to the house",
            "parameters": {"target_label": "house"},
            "task_context": {
                "source": "chromie",
                "intent": "unresolved_place_navigation",
            },
        },
    },
    {
        "name": "turn left then nod twice compiles to skill sequence dry-run",
        "user_command": "Turn left then nod twice.",
        "tool": "soridormi.task.submit",
        "task": {
            "task_type": "skill_sequence",
            "summary": "turn left then nod twice",
            "parameters": {
                "sequence": [
                    {
                        "skill_id": "turn_in_place",
                        "parameters": {"yaw_radps": 0.12, "duration_s": 2.0},
                    },
                    {
                        "skill_id": "nod_yes",
                        "parameters": {"count": 2, "duration_s": 2.0},
                    },
                ]
            },
            "task_context": {
                "source": "chromie",
                "intent": "turn_then_affirm",
            },
        },
    },
    {
        "name": "bring me some water fails closed on missing manipulation",
        "user_command": "Can you bring me some water?",
        "tool": "soridormi.task.submit",
        "task": {
            "task_type": "deliver_object",
            "summary": "bring water",
            "parameters": {"object_label": "water"},
            "task_context": {
                "source": "chromie",
                "intent": "object_delivery",
            },
        },
    },
    {
        "name": "stop now redirects to dedicated safety tools",
        "user_command": "Stop now.",
        "tool": "soridormi.task.preview",
        "task": {
            "task_type": "stop_now",
            "summary": "stop now",
            "parameters": {"urgency": "immediate"},
            "task_context": {
                "source": "chromie",
                "intent": "immediate_stop",
            },
        },
    },
]


def _action_names(payload: dict[str, Any]) -> list[str]:
    return [
        str(action.get("action"))
        for action in payload.get("recommended_next_actions", [])
        if isinstance(action, dict)
    ]


def _case_summary(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    event_types: list[str] | None = None,
) -> dict[str, Any]:
    structured_task = case["task"]
    result = {
        "name": case["name"],
        "user_command": case["user_command"],
        "chromie_boundary": {
            "role": "Chromie converts user language into a structured Soridormi task.",
            "soridormi_receives_raw_language": False,
            "structured_task": structured_task,
        },
        "tool": case["tool"],
        "task_type": payload.get("task_type"),
        "accepted": payload.get("accepted"),
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "terminal": payload.get("terminal"),
        "no_motion": payload.get("no_motion"),
        "execution_mode": payload.get("execution_mode"),
        "skill_id": payload.get("skill_id"),
        "skill_sequence": payload.get("skill_sequence", []),
        "reason_code": payload.get("reason_code"),
        "blocked_subsystems": payload.get("blocked_subsystems", []),
        "plan_layers": [
            str(step.get("layer"))
            for step in payload.get("plan_steps", [])
            if isinstance(step, dict)
        ],
        "recommended_actions": _action_names(payload),
    }
    if event_types is not None:
        result["event_types"] = event_types
    return result


def _call_demo_case(
    service: SoridormiLocalToolService,
    case: dict[str, Any],
) -> tuple[dict[str, Any], list[str] | None]:
    payload = service.call_tool(case["tool"], case["task"])
    if case["tool"] != "soridormi.task.submit" or "task_id" not in payload:
        return payload, None
    events = service.call_tool("soridormi.task.events", {"task_id": payload["task_id"]})
    return payload, [
        str(event["type"])
        for event in events["events"]
    ]


def _structured_task_type(case_summary: dict[str, Any]) -> str:
    return str(
        case_summary["chromie_boundary"]["structured_task"]["task_type"]
    )


def _boundary_summary() -> dict[str, Any]:
    return {
        "chromie_role": [
            "understand user request",
            "resolve ambiguity and confirmations",
            "create structured task payload",
            "own speech and user-facing reporting",
        ],
        "soridormi_role": [
            "validate structured embodied task",
            "report readiness and missing body subsystems",
            "compile supported tasks to named body skills",
            "refuse unsafe or missing-capability tasks without lowering to raw controls",
        ],
        "raw_language_to_low_level_policy": False,
        "raw_motor_or_action_14d_from_chromie": False,
    }


def build_demo(mode: str = "sim") -> dict[str, Any]:
    service = SoridormiLocalToolService(mode=mode)
    capabilities = service.call_tool("soridormi.task.get_capabilities", {})
    by_type = {
        str(task["task_type"]): task
        for task in capabilities["task_types"]
        if task["task_type"] in INTERESTING_TASK_TYPES
    }

    cases = []
    for case in DEMO_CASES:
        payload, event_types = _call_demo_case(service, case)
        cases.append(_case_summary(case, payload, event_types=event_types))

    return {
        "schema_version": "soridormi.task_contract_demo.v1",
        "mode": mode,
        "backend": "local_tool_dry_run",
        "no_motion": True,
        "boundary": _boundary_summary(),
        "capability_summary": [
            {
                "task_type": task_type,
                "readiness": task["readiness"],
                "execution_modes": task["execution_modes"],
                "persistent_submit_allowed": task["persistent_submit_allowed"],
                "missing_subsystems": task["missing_subsystems"],
                "external_dependencies": task["external_dependencies"],
                "reason_code": task["reason_code"],
            }
            for task_type, task in by_type.items()
        ],
        "cases": cases,
    }


def render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Soridormi task MCP contract demo",
        f"mode={payload['mode']} backend={payload['backend']} no_motion={str(payload['no_motion']).lower()}",
        "",
        "Boundary:",
        "- Chromie: understand user language, ask/confirm, and produce structured task payloads.",
        "- Soridormi: validate structured embodied tasks, compile supported skills, and fail closed when body subsystems are missing.",
        "- Raw language and raw action_14d do not go into the low-level policy.",
        "",
        "Capability readiness:",
    ]
    for task in payload["capability_summary"]:
        missing = ", ".join(task["missing_subsystems"]) or "none"
        external = ", ".join(task["external_dependencies"]) or "none"
        reason = task["reason_code"] or "none"
        lines.append(
            "- "
            f"{task['task_type']}: {task['readiness']} "
            f"modes={','.join(task['execution_modes'])} "
            f"missing={missing} external={external} reason={reason}"
        )

    lines.extend(["", "Demo cases:"])
    for case in payload["cases"]:
        actions = ", ".join(case["recommended_actions"]) or "none"
        blocked = ", ".join(case["blocked_subsystems"]) or "none"
        layers = " -> ".join(case["plan_layers"]) or "none"
        reason = case["reason_code"] or "none"
        lines.append(f"- {case['name']}")
        lines.append(f"  user_command={case['user_command']}")
        lines.append(
            "  "
            f"chromie_structured_task={_structured_task_type(case)}"
        )
        lines.append(
            "  "
            f"{case['tool']} task_type={case['task_type']} "
            f"status={case['status']} phase={case['phase']} "
            f"execution_mode={case['execution_mode']} no_motion={str(case['no_motion']).lower()}"
        )
        lines.append(f"  reason={reason} blocked={blocked}")
        lines.append(f"  plan_layers={layers}")
        lines.append(f"  recommended_actions={actions}")
        if case["skill_sequence"]:
            skill_ids = ", ".join(
                str(step["skill_id"])
                for step in case["skill_sequence"]
            )
            lines.append(f"  skill_sequence={skill_ids}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a no-motion Soridormi task MCP contract demo.",
    )
    parser.add_argument(
        "--mode",
        default="sim",
        choices=["sim", "hardware_shadow", "hardware_dry_run"],
        help="Safe Soridormi mode reported by the local demo service.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json"],
        default="summary",
        help="Output format. Use json for the full machine-readable payload.",
    )
    parser.add_argument("--json", action="store_true", help="Emit full JSON payload.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    payload = build_demo(mode=args.mode)
    if args.json or args.compact or args.format == "json":
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=None if args.compact else 2,
            )
        )
        return
    print(render_summary(payload))


if __name__ == "__main__":
    main()
