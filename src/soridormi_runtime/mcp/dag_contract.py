from __future__ import annotations

from typing import Any


def build_soridormi_dag_contract(*, mode: str = "sim") -> dict[str, Any]:
    """Return Soridormi-side task-graph integration hints.

    Chromie owns the global DAG planner/executor. Soridormi only declares the
    robot-body ordering, safety, and host-tool requirements that Chromie should
    enforce when composing TaskGraphs around Soridormi motion tools.
    """

    return {
        "schema_version": "0.1",
        "source": "soridormi",
        "mode": mode,
        "tool_prefix": "soridormi.",
        "host_required_tools": [
            "chromie.ask_confirmation",
            "chromie.report",
        ],
        "physical_motion_tools": [
            "soridormi.motion.execute_plan",
        ],
        "planning_tools": [
            "soridormi.motion.create_plan",
        ],
        "safety_tools": [
            "soridormi.safety.monitor_motion",
            "soridormi.motion.stop",
            "soridormi.safety.emergency_stop",
        ],
        "default_short_motion_sequence": [
            "soridormi.robot.get_status",
            "soridormi.motion.create_plan",
            "chromie.ask_confirmation",
            "soridormi.safety.monitor_motion during soridormi.motion.execute_plan",
            "soridormi.motion.execute_plan",
            "chromie.report",
        ],
        "rules": [
            "Soridormi tools do not implement Chromie speech, TTS, ASR, or user confirmation.",
            "Chromie must use chromie.ask_confirmation before physical-motion execution unless the action is stop/cancel/emergency_stop.",
            "Chromie must cover soridormi.motion.execute_plan with soridormi.safety.monitor_motion.",
            "Chromie must use soridormi.motion.create_plan before soridormi.motion.execute_plan.",
            "Do not expose raw motor, joint, or torque controls to LLM-generated graphs.",
            "stop and emergency_stop may preempt any running motion task.",
        ],
        "fallback_recommendations": {
            "soridormi.motion.execute_plan": {
                "on_failure": "soridormi.motion.stop then chromie.report",
                "on_timeout": "soridormi.motion.stop then chromie.report",
                "on_safety_event": "soridormi.safety.emergency_stop then chromie.report",
            }
        },
    }
