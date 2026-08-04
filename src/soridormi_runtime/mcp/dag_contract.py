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
            "soridormi.skill.execute_plan",
            "soridormi.activity.execute_plan",
        ],
        "planning_tools": [
            "soridormi.motion.create_plan",
            "soridormi.skill.create_plan",
            "soridormi.activity.get_capabilities",
            "soridormi.activity.create_plan",
            "soridormi.task.get_capabilities",
            "soridormi.task.preview",
            "soridormi.task.submit",
        ],
        "body_activity_tools": [
            "soridormi.activity.get_capabilities",
            "soridormi.activity.create_plan",
            "soridormi.activity.execute_plan",
            "soridormi.activity.status",
            "soridormi.activity.cancel",
        ],
        "embodied_task_tools": [
            "soridormi.task.get_capabilities",
            "soridormi.task.preview",
            "soridormi.task.submit",
            "soridormi.task.status",
            "soridormi.task.events",
            "soridormi.task.cancel",
        ],
        "safety_tools": [
            "soridormi.safety.monitor_motion",
            "soridormi.motion.stop",
            "soridormi.activity.cancel",
            "soridormi.task.cancel",
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
        "chromie_concurrency_model": {
            "cognitive_core": "single_authoritative_mind",
            "lanes": [
                "social_attention_proposal",
                "speaking_execution",
                "activity_execution",
            ],
            "runtime_coordinator": "chromie",
            "speech_owner": "chromie",
            "body_provider": "soridormi",
        },
        "soridormi_concurrency_model": {
            "primary_locomotion_members": "zero_or_one",
            "subtle_expression_members": "zero_or_more_when_resources_do_not_conflict",
            "one_final_motor_command_authority": True,
            "safety_authority": "soridormi",
        },
        "rules": [
            "Soridormi tools do not implement Chromie speech, TTS, ASR, or user confirmation.",
            "The Social-Attention Proposal Lane proposes only; it does not author speech, own Goals, authorize body actions, or operate actuators.",
            "The Speaking Execution Lane and Activity Execution Lane are peer execution lanes beneath one Cognitive Core.",
            "Chromie's authoritative planner selects exact body skills before the Activity Execution Lane calls soridormi.activity.create_plan.",
            "Speech or singing is never a Soridormi activity member; Chromie coordinates peer execution through coordination_id.",
            "Chromie must use chromie.ask_confirmation before physical-motion execution unless the action is stop/cancel/emergency_stop.",
            "Chromie must cover soridormi.motion.execute_plan with soridormi.safety.monitor_motion.",
            "Chromie must use soridormi.motion.create_plan before soridormi.motion.execute_plan.",
            "Chromie must cover soridormi.skill.execute_plan with soridormi.safety.monitor_motion.",
            "Chromie must use soridormi.skill.create_plan before soridormi.skill.execute_plan.",
            "Chromie must use soridormi.activity.create_plan before soridormi.activity.execute_plan.",
            "Chromie must cover soridormi.activity.execute_plan with soridormi.safety.monitor_motion.",
            "Soridormi accepts at most one primary locomotion member and composes compatible head overlays into one final motor command.",
            "Soridormi may reject, constrain, cancel, or recover body activity without waiting for Chromie when physical safety requires it.",
            "Chromie may use soridormi.task.get_capabilities to inspect Soridormi-owned embodied readiness before preview or submission.",
            "Chromie may use soridormi.task.preview before user confirmation or task submission to inspect Soridormi's no-motion embodied interpretation.",
            "Chromie may submit structured embodied goals through soridormi.task.submit, but task submission is contract-only and no-motion until Soridormi adds task execution.",
            "Chromie must use soridormi.task.status or soridormi.task.events to monitor task-level requests.",
            "Do not expose raw motor, joint, or torque controls to LLM-generated graphs.",
            "stop and emergency_stop may preempt any running motion task.",
        ],
        "fallback_recommendations": {
            "soridormi.motion.execute_plan": {
                "on_failure": "soridormi.motion.stop then chromie.report",
                "on_timeout": "soridormi.motion.stop then chromie.report",
                "on_safety_event": "soridormi.safety.emergency_stop then chromie.report",
            },
            "soridormi.skill.execute_plan": {
                "on_failure": "soridormi.motion.stop then chromie.report",
                "on_timeout": "soridormi.motion.stop then chromie.report",
                "on_safety_event": "soridormi.safety.emergency_stop then chromie.report",
            },
            "soridormi.activity.execute_plan": {
                "on_failure": "soridormi.activity.cancel then chromie.report",
                "on_timeout": "soridormi.activity.cancel then soridormi.motion.stop then chromie.report",
                "on_safety_event": "soridormi.safety.emergency_stop then chromie.report",
            }
        },
    }
