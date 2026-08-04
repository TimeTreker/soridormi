from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from soridormi_runtime.skill_execution import SkillExecutionError, SkillExecutionRegistry
from soridormi_runtime.task_capabilities import (
    TaskCapabilityManifestError,
    load_task_capability_manifest,
    task_capabilities_by_type,
    validate_task_capability_manifest,
)

_TASK_CAPABILITY_MANIFEST = load_task_capability_manifest()
_TASK_CAPABILITY_VALIDATION = validate_task_capability_manifest(_TASK_CAPABILITY_MANIFEST)
if not _TASK_CAPABILITY_VALIDATION.ok:
    raise TaskCapabilityManifestError(
        "Invalid task capability manifest: "
        + "; ".join(_TASK_CAPABILITY_VALIDATION.errors)
    )
_TASK_CAPABILITIES_BY_TYPE = task_capabilities_by_type(_TASK_CAPABILITY_MANIFEST)

KNOWN_TASK_TYPES = tuple(_TASK_CAPABILITIES_BY_TYPE)
TASK_READY_SUBSYSTEMS = tuple(_TASK_CAPABILITY_MANIFEST.get("ready_subsystems", []))
TASK_UNSAFE_TYPES = set(_TASK_CAPABILITY_MANIFEST.get("unsafe_task_types", []))
TASK_TYPE_DESCRIPTIONS = {
    task_type: str(task.get("description") or task_type)
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_READINESS = {
    task_type: str(task.get("readiness") or "unsupported")
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_EXECUTION_MODES = {
    task_type: [str(mode) for mode in task.get("execution_modes", [])]
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_REQUIRED_SUBSYSTEMS = {
    task_type: list(task.get("required_subsystems", []))
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_EXTERNAL_DEPENDENCIES = {
    task_type: list(task.get("external_dependencies", []))
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_DECLARED_MISSING_SUBSYSTEMS = {
    task_type: list(task.get("missing_subsystems", []))
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_RECOMMENDED_ACTIONS = {
    task_type: list(task.get("recommended_actions", []))
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_PERSISTENT_SUBMIT_ALLOWED = {
    task_type: bool(task.get("persistent_submit_allowed", False))
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
TASK_REASON_CODES = {
    task_type: task.get("reason_code")
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
}
_MAX_SEQUENCE_STEPS = 8

FUTURE_BLOCKED_TASKS = {
    task_type: str(task.get("reason_code") or "future_capability_blocked")
    for task_type, task in _TASK_CAPABILITIES_BY_TYPE.items()
    if task.get("readiness") == "future_blocked"
}
FUTURE_BLOCKED_SUBSYSTEMS = {
    task_type: list(TASK_DECLARED_MISSING_SUBSYSTEMS.get(task_type, []))
    for task_type in FUTURE_BLOCKED_TASKS
}
SKILL_BACKED_TASK_TYPES = {
    task_type
    for task_type, modes in TASK_EXECUTION_MODES.items()
    if "skill_dry_run" in modes or "skill_sequence_dry_run" in modes
}
CONTRACT_ONLY_TASK_TYPES = {
    task_type
    for task_type, readiness in TASK_READINESS.items()
    if readiness == "contract_planning_hold"
}
SAFE_IDLE_TASK_TYPES = {
    task_type
    for task_type, readiness in TASK_READINESS.items()
    if readiness == "safe_idle_contract"
}
STOP_REDIRECT_TASK_TYPES = {
    task_type
    for task_type, readiness in TASK_READINESS.items()
    if readiness == "safety_redirect"
}

UNSAFE_TASK_TYPES = TASK_UNSAFE_TYPES

TERMINAL_STATUSES = {"completed", "cancelled", "failed", "refused"}
TASK_PHASES = (
    "accepted",
    "resolving",
    "planning",
    "executing",
    "monitoring",
    "recovering",
    "completed",
    "failed",
    "cancelled",
    "refused",
)
TERMINAL_PHASES = {"completed", "failed", "cancelled", "refused"}
_PHASE_STATUSES = {
    "accepted": "accepted",
    "resolving": "accepted",
    "planning": "accepted",
    "executing": "accepted",
    "monitoring": "accepted",
    "recovering": "accepted",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "refused": "refused",
}
_ALLOWED_PHASE_TRANSITIONS = {
    "accepted": {"resolving", "cancelled", "refused"},
    "resolving": {"planning", "failed", "cancelled", "refused"},
    "planning": {"executing", "failed", "cancelled", "refused", "completed"},
    "executing": {"monitoring", "recovering", "failed", "cancelled", "completed"},
    "monitoring": {"executing", "recovering", "failed", "cancelled", "completed"},
    "recovering": {"monitoring", "failed", "cancelled", "completed"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
    "refused": set(),
}

_MAX_TIMEOUT_S = 300.0
_DEFAULT_TIMEOUT_S = 30.0
_ALLOWED_TOP_LEVEL_FIELDS = {
    "client_task_ref",
    "task_type",
    "summary",
    "parameters",
    "task_context",
    "environment_context",
    "safety_constraints",
    "timeout_s",
    "cancellation_policy",
}
_ALLOWED_CANCELLATION_POLICIES = {
    "best_effort_stop",
    "cancel_before_execution",
    "emergency_stop_on_timeout",
}
_FORBIDDEN_LOW_LEVEL_FIELDS = {
    "action14d",
    "action_14d",
    "actuatorcommand",
    "actuatorcommands",
    "actuatorctrl",
    "actuator_ctrl",
    "controllerarray",
    "controller_array",
    "jointaction",
    "jointactions",
    "jointtarget",
    "jointtargets",
    "joint_targets",
    "motorcommand",
    "motorcommands",
    "motor_command",
    "motor_commands",
    "motortarget",
    "motortargets",
    "motor_target",
    "motor_targets",
    "policyaction",
    "policy_actions",
    "rawaction",
    "raw_action",
    "rawpolicyoutput",
    "raw_policy_output",
    "torque",
    "torques",
    "torquecommand",
    "torquecommands",
    "torque_command",
    "torque_commands",
}
_MAX_CLIENT_TASK_REF_LENGTH = 128
_TIMEOUT_REASON_CODES = {
    "task_timeout",
    "task_timeout_emergency_stop_required",
}


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def reject_low_level_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            normalized = _normalized_key(key_text)
            if key_text.lower() in _FORBIDDEN_LOW_LEVEL_FIELDS or normalized in _FORBIDDEN_LOW_LEVEL_FIELDS:
                raise ValueError(f"low-level robot control field is not allowed at {path}.{key_text}")
            reject_low_level_fields(nested, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_low_level_fields(nested, path=f"{path}[{index}]")


def _effective_safe_idle(*, emergency_stop: bool, safe_idle: bool | None) -> bool:
    if emergency_stop:
        return False
    if safe_idle is None:
        return True
    return bool(safe_idle)


def task_capabilities_payload(
    *,
    mode: str,
    backend: str,
    emergency_stop: bool,
    safe_idle: bool | None = None,
    skill_registry: SkillExecutionRegistry | None = None,
) -> dict[str, Any]:
    executable_skill_ids = (
        sorted(skill_registry.executable_skill_ids())
        if skill_registry is not None
        else []
    )
    task_types = []
    ready_subsystem_set = set(TASK_READY_SUBSYSTEMS)
    for task_type in KNOWN_TASK_TYPES:
        task = _TASK_CAPABILITIES_BY_TYPE[task_type]
        reason_code = TASK_REASON_CODES.get(task_type)
        required_subsystems = list(TASK_REQUIRED_SUBSYSTEMS.get(task_type, []))
        external_dependencies = list(TASK_EXTERNAL_DEPENDENCIES.get(task_type, []))
        missing_subsystems = list(TASK_DECLARED_MISSING_SUBSYSTEMS.get(task_type, []))
        ready_subsystems = [
            subsystem
            for subsystem in required_subsystems
            if subsystem in ready_subsystem_set
            and subsystem not in missing_subsystems
        ]
        readiness = TASK_READINESS[task_type]
        execution_modes = list(TASK_EXECUTION_MODES.get(task_type, ["contract_only"]))
        persistent_submit_allowed = TASK_PERSISTENT_SUBMIT_ALLOWED.get(task_type, False)
        recommended_actions = list(TASK_RECOMMENDED_ACTIONS.get(task_type, []))

        if emergency_stop:
            persistent_submit_allowed = False

        task_types.append(
            {
                "task_type": task_type,
                "description": TASK_TYPE_DESCRIPTIONS.get(task_type, task_type),
                "readiness": readiness,
                "task_api_no_motion": bool(task.get("task_api_no_motion", True)),
                "physical_execution_ready": bool(
                    task.get("physical_execution_ready", False)
                ),
                "preview_allowed": bool(task.get("preview_allowed", True)),
                "persistent_submit_allowed": persistent_submit_allowed,
                "execution_modes": execution_modes,
                "required_subsystems": required_subsystems,
                "ready_subsystems": ready_subsystems,
                "missing_subsystems": missing_subsystems,
                "reason_code": reason_code,
                "recommended_actions": recommended_actions,
                "external_dependencies": external_dependencies,
            }
        )

    return {
        "schema_version": "soridormi.task_capabilities.v1",
        "mode": mode,
        "backend": backend,
        "emergency_stop": emergency_stop,
        "safe_idle": _effective_safe_idle(
            emergency_stop=emergency_stop,
            safe_idle=safe_idle,
        ),
        "readiness_profile": str(_TASK_CAPABILITY_MANIFEST["readiness_profile"]),
        "task_api_no_motion": bool(_TASK_CAPABILITY_MANIFEST["task_api_no_motion"]),
        "physical_execution_note": str(_TASK_CAPABILITY_MANIFEST["physical_execution_note"]),
        "ready_subsystems": list(TASK_READY_SUBSYSTEMS),
        "unsafe_task_types": sorted(UNSAFE_TASK_TYPES),
        "executable_skill_ids": executable_skill_ids,
        "task_types": task_types,
    }


def _require_object(args: dict[str, Any], name: str) -> dict[str, Any]:
    value = args.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _timeout_s(args: dict[str, Any]) -> float:
    value = args.get("timeout_s", _DEFAULT_TIMEOUT_S)
    timeout = float(value)
    if timeout <= 0.0 or timeout > _MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {_MAX_TIMEOUT_S}]")
    return timeout


def _client_task_ref(args: dict[str, Any]) -> str | None:
    if "client_task_ref" not in args or args.get("client_task_ref") is None:
        return None
    value = str(args["client_task_ref"]).strip()
    if not value:
        raise ValueError("client_task_ref must be a non-empty string when provided")
    if len(value) > _MAX_CLIENT_TASK_REF_LENGTH:
        raise ValueError(
            f"client_task_ref must be at most {_MAX_CLIENT_TASK_REF_LENGTH} characters"
        )
    return value


def _task_request_fingerprint(args: dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in args.items()
        if key != "client_task_ref"
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"), default=str)


def _copy_present(
    source: dict[str, Any],
    *names: str,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    aliases = aliases or {}
    result: dict[str, Any] = {}
    for name in names:
        if name in source:
            result[name] = source[name]
    for source_name, target_name in aliases.items():
        if source_name in source and target_name not in result:
            result[target_name] = source[source_name]
    return result


def _gesture_skill_id(parameters: dict[str, Any]) -> str:
    raw = str(parameters.get("skill_id") or parameters.get("gesture") or "").strip().lower()
    aliases = {
        "attention": "express_attention",
        "curious": "express_attention",
        "express_attention": "express_attention",
        "expressive_idle": "express_attention",
        "idle": "express_attention",
        "neutral": "neutral_head",
        "neutral_head": "neutral_head",
        "nod": "nod_yes",
        "nod_yes": "nod_yes",
        "yes": "nod_yes",
        "shake": "shake_no",
        "shake_no": "shake_no",
        "no": "shake_no",
        "bow": "bow",
    }
    skill_id = aliases.get(raw)
    if skill_id is None:
        raise ValueError("perform_gesture requires a supported gesture or skill_id")
    return skill_id


def _skill_request_for_task(record: EmbodiedTaskRecord) -> tuple[str, dict[str, Any]]:
    parameters = dict(record.parameters)
    if record.task_type == "move_forward":
        return (
            "walk_forward",
            _copy_present(
                parameters,
                "speed",
                "duration_s",
                aliases={"pace": "speed", "speed_label": "speed"},
            ),
        )
    if record.task_type == "move_velocity":
        return (
            "walk_velocity",
            _copy_present(
                parameters,
                "vx_mps",
                "vy_mps",
                "yaw_radps",
                "duration_s",
                aliases={"vx": "vx_mps", "vy": "vy_mps", "yaw": "yaw_radps"},
            ),
        )
    if record.task_type == "turn_to_heading":
        skill_parameters = _copy_present(parameters, "yaw_radps", "duration_s")
        if "yaw_radps" not in skill_parameters:
            direction = str(parameters.get("direction", "")).strip().lower()
            if direction == "left":
                skill_parameters["yaw_radps"] = 0.12
            elif direction == "right":
                skill_parameters["yaw_radps"] = -0.12
        return "turn_in_place", skill_parameters
    if record.task_type == "look_at_target":
        if "head_yaw_rad" in parameters or "head_pitch_rad" in parameters:
            return (
                "look_direction",
                _copy_present(parameters, "head_yaw_rad", "head_pitch_rad", "duration_s"),
            )
        skill_parameters = _copy_present(
            parameters,
            "target_yaw_rad",
            "target_pitch_rad",
            "target_ref",
            "duration_s",
            "hold_fraction",
            "end_mode",
        )
        if "target_ref" not in skill_parameters:
            target_label = str(parameters.get("target_label") or "").strip()
            if not target_label:
                raise ValueError(
                    "look_at_target requires target_ref or target_label; "
                    "Soridormi must not invent a target"
                )
            skill_parameters["target_ref"] = target_label
        if "target_yaw_rad" not in skill_parameters and "target_pitch_rad" not in skill_parameters:
            raise ValueError(
                "look_at_target requires target_yaw_rad or target_pitch_rad; "
                "Soridormi must not invent a body direction"
            )
        return "look_at_person", skill_parameters
    if record.task_type == "perform_gesture":
        skill_id = _gesture_skill_id(parameters)
        skill_parameters = {
            key: value
            for key, value in parameters.items()
            if key not in {"gesture", "skill_id"}
        }
        return skill_id, skill_parameters
    raise ValueError(f"task type is not skill-backed: {record.task_type}")


def _skill_sequence_for_task(record: EmbodiedTaskRecord) -> list[tuple[str, dict[str, Any]]]:
    sequence = record.parameters.get("sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("skill_sequence requires a non-empty parameters.sequence list")
    if len(sequence) > _MAX_SEQUENCE_STEPS:
        raise ValueError(f"skill_sequence may contain at most {_MAX_SEQUENCE_STEPS} steps")

    result: list[tuple[str, dict[str, Any]]] = []
    for index, step in enumerate(sequence):
        if not isinstance(step, dict):
            raise ValueError(f"skill_sequence step {index} must be an object")
        skill_id = str(step.get("skill_id") or step.get("skill") or "").strip()
        if not skill_id:
            raise ValueError(f"skill_sequence step {index} requires skill_id")
        parameters = step.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(f"skill_sequence step {index} parameters must be an object")
        result.append((skill_id, dict(parameters)))
    return result


def _plan_step(
    index: int,
    *,
    layer: str,
    kind: str,
    summary: str,
    status: str,
    owner: str = "soridormi",
    no_motion: bool = True,
    skill_id: str | None = None,
    estimated_duration_s: float | None = None,
    requires: list[str] | None = None,
    recommended_tools: list[str] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_index": index,
        "owner": owner,
        "layer": layer,
        "kind": kind,
        "status": status,
        "summary": summary,
        "no_motion": no_motion,
    }
    if skill_id is not None:
        step["skill_id"] = skill_id
    if estimated_duration_s is not None:
        step["estimated_duration_s"] = estimated_duration_s
    if requires:
        step["requires"] = requires
    if recommended_tools:
        step["recommended_tools"] = recommended_tools
    return step


def _blocked_plan_steps_for_task(task_type: str) -> list[dict[str, Any]]:
    if task_type == "navigate_to_location":
        return [
            _plan_step(
                1,
                layer="sensing",
                kind="target_resolution",
                status="blocked",
                summary="Resolve a structured target label into a mapped or observable destination.",
                requires=["target_resolution", "environment_map_or_perception"],
            ),
            _plan_step(
                2,
                layer="localization",
                kind="localize_robot",
                status="blocked",
                summary="Estimate robot pose relative to the resolved destination.",
                requires=["localization"],
            ),
            _plan_step(
                3,
                layer="routing",
                kind="route_planning",
                status="blocked",
                summary="Plan a route through known free space.",
                requires=["route_planner", "obstacle_map"],
            ),
            _plan_step(
                4,
                layer="planning",
                kind="local_trajectory_planning",
                status="blocked",
                summary="Convert route segments into bounded local body goals.",
                requires=["local_motion_planner", "safety_constraints"],
            ),
            _plan_step(
                5,
                layer="control",
                kind="skill_execution",
                status="not_started",
                summary="Execute validated local body goals through Soridormi skills.",
                requires=["validated_route"],
            ),
        ]
    if task_type == "approach_target":
        return [
            _plan_step(
                1,
                layer="sensing",
                kind="target_tracking",
                status="blocked",
                summary="Detect and track the target before moving closer.",
                requires=["perception_pipeline", "target_tracking"],
            ),
            _plan_step(
                2,
                layer="planning",
                kind="approach_policy",
                status="blocked",
                summary="Choose a bounded approach distance, speed, and stop condition.",
                requires=["local_motion_planner", "stop_distance_policy"],
            ),
            _plan_step(
                3,
                layer="control",
                kind="skill_execution",
                status="not_started",
                summary="Execute the bounded approach through Soridormi locomotion skills.",
                requires=["validated_target_track"],
            ),
        ]
    if task_type == "deliver_object":
        return [
            _plan_step(
                1,
                layer="sensing",
                kind="object_resolution",
                status="blocked",
                summary="Resolve and locate the requested object.",
                requires=["object_perception", "object_affordance_model"],
            ),
            _plan_step(
                2,
                layer="routing",
                kind="navigation_to_object",
                status="blocked",
                summary="Navigate to the object through Soridormi navigation.",
                requires=["navigation_pipeline"],
            ),
            _plan_step(
                3,
                layer="manipulation",
                kind="grasp_and_carry",
                status="blocked",
                summary="Pick up and carry the object with a validated manipulation stack.",
                requires=["manipulation_capability", "carry_safety_policy"],
            ),
            _plan_step(
                4,
                layer="interaction",
                kind="handoff",
                status="blocked",
                summary="Hand the object to the user with confirmation and safety checks.",
                requires=["handoff_protocol"],
            ),
        ]
    if task_type == "stop_now":
        return [
            _plan_step(
                1,
                layer="safety",
                kind="immediate_stop_redirect",
                status="refused",
                summary="Immediate stop requests must use the dedicated safety tools.",
                recommended_tools=[
                    "soridormi.motion.stop",
                    "soridormi.motion.cancel",
                    "soridormi.safety.emergency_stop",
                ],
            )
        ]
    if task_type in UNSAFE_TASK_TYPES:
        return [
            _plan_step(
                1,
                layer="safety",
                kind="policy_refusal",
                status="refused",
                summary="Unsafe physical requests are not lowered into body motion.",
                requires=["human_safety_policy"],
            )
        ]
    return []


def _contract_only_plan_steps(record: EmbodiedTaskRecord) -> list[dict[str, Any]]:
    if record.task_type == "speak_while_moving":
        return [
            _plan_step(
                1,
                owner="chromie",
                layer="interaction",
                kind="speech_coordination",
                status="external",
                summary=(
                    "Chromie owns communicative meaning, TTS or singing, playback, "
                    "and interaction cancellation."
                ),
                requires=["chromie_speech_coordination"],
            ),
            _plan_step(
                2,
                owner="chromie",
                layer="planning",
                kind="exact_capability_selection",
                status="planning_hold",
                summary=(
                    "The Cognitive Core and planner must select exact speech and body "
                    "capabilities plus their timing relationship."
                ),
                requires=["authoritative_plan", "coordination_id"],
            ),
            _plan_step(
                3,
                layer="body_activity",
                kind="resource_validated_body_activity",
                status="ready_for_exact_plan",
                summary=(
                    "Soridormi can validate and execute a compatible body-activity "
                    "plan through its resource arbiter and final command composer."
                ),
                requires=[
                    "body_activity_scheduler",
                    "body_command_composer",
                    "physical_resource_arbiter",
                ],
                recommended_tools=[
                    "soridormi.activity.get_capabilities",
                    "soridormi.activity.compile",
                    "soridormi.activity.execute",
                ],
            ),
        ]
    if record.task_type == "recover_safe_idle":
        return [
            _plan_step(
                1,
                layer="safety",
                kind="safe_idle_check",
                status="completed",
                summary="Confirmed contract-level safe idle without issuing motion.",
            )
        ]
    return [
        _plan_step(
            1,
            layer="planning",
            kind="task_planning_hold",
            status="planning_hold",
            summary="Task accepted at the no-motion planning boundary.",
            requires=["task_execution_state_machine"],
        )
    ]


def _recommended_next_actions(
    record: EmbodiedTaskRecord,
    *,
    persistent: bool,
) -> list[dict[str, Any]]:
    if record.reason_code in _TIMEOUT_REASON_CODES:
        actions = [
            {
                "action": "report_task_timeout",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            }
        ]
        if record.reason_code == "task_timeout_emergency_stop_required":
            actions.append(
                {
                    "action": "call_emergency_stop_if_motion_active",
                    "owner": "chromie",
                    "priority": "required",
                    "recommended_tools": ["soridormi.safety.emergency_stop"],
                    "reason_code": record.reason_code,
                }
            )
        return actions
    if record.reason_code == "use_safety_tool_for_immediate_stop":
        return [
            {
                "action": "call_dedicated_stop_tool",
                "owner": "chromie",
                "priority": "immediate",
                "recommended_tools": [
                    "soridormi.motion.stop",
                    "soridormi.motion.cancel",
                    "soridormi.safety.emergency_stop",
                ],
                "reason_code": record.reason_code,
            },
            {
                "action": "do_not_resubmit_as_task",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            },
        ]
    if record.reason_code == "unsafe_task":
        return [
            {
                "action": "report_refusal",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            },
            {
                "action": "do_not_lower_to_motion",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            },
        ]
    if record.reason_code in {
        "missing_navigation_pipeline",
        "missing_perception_pipeline",
        "missing_manipulation_capability",
    }:
        return [
            {
                "action": "report_blocked_or_clarify",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
                "blocked_subsystems": list(record.blocked_subsystems),
            },
            {
                "action": "do_not_lower_to_velocity_recipe",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            },
        ]
    if record.reason_code == "emergency_stop_active":
        return [
            {
                "action": "maintain_stop_and_report",
                "owner": "chromie",
                "priority": "required",
                "recommended_tools": ["soridormi.robot.get_status"],
                "reason_code": record.reason_code,
            }
        ]
    if record.status == "failed":
        return [
            {
                "action": "report_task_failure",
                "owner": "chromie",
                "priority": "required",
                "reason_code": record.reason_code,
            }
        ]
    if record.execution_mode in {"skill_dry_run", "skill_sequence_dry_run"}:
        first_action = {
            "action": "report_contract_dry_run_complete"
            if persistent
            else "submit_task_when_confirmed",
            "owner": "chromie",
            "priority": "normal",
            "recommended_tools": []
            if persistent
            else ["soridormi.task.submit"],
            "reason_code": "no_motion_contract",
        }
        return [
            first_action,
            {
                "action": "use_skill_execution_path_for_physical_motion",
                "owner": "chromie",
                "priority": "conditional",
                "recommended_tools": [
                    "soridormi.skill.create_plan",
                    "soridormi.safety.monitor_motion",
                    "soridormi.skill.execute_plan",
                ],
                "reason_code": "task_api_no_motion",
            },
            {
                "action": "do_not_report_physical_completion",
                "owner": "chromie",
                "priority": "required",
                "reason_code": "no_motion_contract",
            },
        ]
    if record.phase == "planning" and record.accepted:
        if persistent:
            return [
                {
                    "action": "monitor_task_or_cancel",
                    "owner": "chromie",
                    "priority": "normal",
                    "recommended_tools": [
                        "soridormi.task.status",
                        "soridormi.task.events",
                        "soridormi.task.cancel",
                    ],
                    "reason_code": "task_execution_held",
                },
                {
                    "action": "do_not_report_physical_completion",
                    "owner": "chromie",
                    "priority": "required",
                    "reason_code": "no_motion_contract",
                },
            ]
        return [
            {
                "action": "submit_task_when_confirmed",
                "owner": "chromie",
                "priority": "normal",
                "recommended_tools": ["soridormi.task.submit"],
                "reason_code": "preview_only",
            }
        ]
    if record.task_type == "recover_safe_idle" and record.phase == "completed":
        return [
            {
                "action": "report_safe_idle",
                "owner": "chromie",
                "priority": "normal",
                "reason_code": "safe_idle_contract",
            }
        ]
    return [
        {
            "action": "report_task_status",
            "owner": "chromie",
            "priority": "normal",
            "reason_code": record.reason_code,
        }
    ]


def _graph_node_from_plan_step(step: dict[str, Any]) -> dict[str, Any]:
    step_index = int(step.get("step_index", 1))
    status = str(step.get("status") or "not_started")
    node: dict[str, Any] = {
        "node_id": f"step-{step_index}",
        "step_index": step_index,
        "owner": str(step.get("owner") or "soridormi"),
        "layer": str(step.get("layer") or "planning"),
        "kind": str(step.get("kind") or "task_step"),
        "status": status,
        "summary": str(step.get("summary") or ""),
        "no_motion": bool(step.get("no_motion", True)),
        "blocked": status == "blocked",
    }
    for field_name in (
        "skill_id",
        "estimated_duration_s",
        "requires",
        "recommended_tools",
    ):
        if field_name in step:
            node[field_name] = step[field_name]
    return node


def _task_graph_for_record(record: EmbodiedTaskRecord) -> dict[str, Any]:
    nodes = [
        _graph_node_from_plan_step(step)
        for step in record.plan_steps
    ]
    if not nodes:
        nodes = [
            {
                "node_id": "task-state",
                "step_index": 1,
                "owner": "soridormi",
                "layer": "lifecycle",
                "kind": "task_state",
                "status": record.phase,
                "summary": "Task has not expanded body plan steps.",
                "no_motion": True,
                "blocked": record.status == "refused",
            }
        ]
    edges = [
        {
            "from": nodes[index]["node_id"],
            "to": nodes[index + 1]["node_id"],
            "kind": "sequence",
        }
        for index in range(len(nodes) - 1)
    ]
    return {
        "schema_version": "soridormi.task_graph.v1",
        "graph_id": f"{record.task_id}:body",
        "task_ref": record.task_id,
        "task_type": record.task_type,
        "owner": "soridormi",
        "current_phase": record.phase,
        "status": record.status,
        "terminal": record.terminal,
        "accepted": record.accepted,
        "no_motion": record.no_motion,
        "execution_mode": record.execution_mode,
        "physical_execution_ready": False,
        "raw_control_allowed": False,
        "blocked_subsystems": list(record.blocked_subsystems),
        "nodes": nodes,
        "edges": edges,
        "boundary": {
            "chromie_owns_global_task_graph": True,
            "soridormi_owns_body_task_graph": True,
            "raw_control_from_chromie_allowed": False,
        },
    }


def _event_poll_recommendation(
    record: EmbodiedTaskRecord,
    *,
    next_after_sequence: int,
) -> dict[str, Any]:
    reason_code = record.reason_code or (
        "task_terminal" if record.terminal else "task_active"
    )
    if record.terminal:
        return {
            "action": "stop_polling",
            "owner": "chromie",
            "priority": "normal",
            "recommended_after_sequence": next_after_sequence,
            "recommended_tools": [],
            "reason_code": reason_code,
        }
    return {
        "action": "continue_polling_or_cancel",
        "owner": "chromie",
        "priority": "normal",
        "recommended_after_sequence": next_after_sequence,
        "recommended_poll_interval_s": 0.5,
        "recommended_tools": [
            "soridormi.task.events",
            "soridormi.task.status",
            "soridormi.task.cancel",
        ],
        "reason_code": reason_code,
    }


@dataclass
class EmbodiedTaskRecord:
    task_id: str
    task_type: str
    status: str
    accepted: bool
    created_at: float
    updated_at: float
    mode: str
    backend: str
    summary: str
    phase: str
    parameters: dict[str, Any]
    task_context: dict[str, Any]
    environment_context: dict[str, Any]
    safety_constraints: dict[str, Any]
    timeout_s: float
    cancellation_policy: str
    reason_code: str | None = None
    reason: str | None = None
    no_motion: bool = True
    execution_mode: str = "contract_only"
    skill_id: str | None = None
    skill_summary: str | None = None
    skill_sequence: list[dict[str, Any]] = field(default_factory=list)
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    blocked_subsystems: list[str] = field(default_factory=list)
    estimated_duration_s: float | None = None
    client_task_ref: str | None = None
    request_fingerprint: str | None = None
    timeout_elapsed_s: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def append_event(
        self,
        event_type: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.updated_at = time.time()
        event = {
            "sequence": len(self.events) + 1,
            "time": self.updated_at,
            "type": event_type,
            "status": self.status,
            "phase": self.phase,
        }
        if self.skill_id is not None:
            event["skill_id"] = self.skill_id
        if self.reason_code is not None:
            event["reason_code"] = self.reason_code
        if message:
            event["message"] = message
        if details:
            event["details"] = details
        self.events.append(event)

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    @property
    def allowed_next_phases(self) -> list[str]:
        return sorted(_ALLOWED_PHASE_TRANSITIONS[self.phase])

    @property
    def deadline_at(self) -> float:
        return self.created_at + self.timeout_s

    @property
    def expired(self) -> bool:
        return self.reason_code in _TIMEOUT_REASON_CODES

    def transition_to(
        self,
        phase: str,
        *,
        event_type: str | None = None,
        message: str | None = None,
        reason_code: str | None = None,
        reason: str | None = None,
        accepted: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if phase not in TASK_PHASES:
            raise ValueError(f"unsupported task phase: {phase}")
        if phase != self.phase and phase not in _ALLOWED_PHASE_TRANSITIONS[self.phase]:
            raise ValueError(f"invalid task phase transition: {self.phase} -> {phase}")
        self.phase = phase
        self.status = _PHASE_STATUSES[phase]
        if accepted is not None:
            self.accepted = accepted
        if reason_code is not None:
            self.reason_code = reason_code
        if reason is not None:
            self.reason = reason
        self.append_event(
            event_type or f"task_{phase}",
            message=message or reason,
            details=details,
        )

    def status_payload(
        self,
        *,
        safe_idle: bool,
        persistent: bool = True,
        idempotent_replay: bool = False,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "client_task_ref": self.client_task_ref,
            "idempotent_replay": idempotent_replay,
            "task_type": self.task_type,
            "status": self.status,
            "phase": self.phase,
            "terminal": self.terminal,
            "allowed_next_phases": self.allowed_next_phases,
            "accepted": self.accepted,
            "mode": self.mode,
            "backend": self.backend,
            "summary": self.summary,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "safe_idle": safe_idle,
            "no_motion": self.no_motion,
            "execution_mode": self.execution_mode,
            "skill_id": self.skill_id,
            "skill_summary": self.skill_summary,
            "skill_sequence": self.skill_sequence,
            "plan_steps": self.plan_steps,
            "task_graph": _task_graph_for_record(self),
            "blocked_subsystems": self.blocked_subsystems,
            "recommended_next_actions": _recommended_next_actions(
                self,
                persistent=persistent,
            ),
            "estimated_duration_s": self.estimated_duration_s,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline_at": self.deadline_at,
            "timeout_s": self.timeout_s,
            "expired": self.expired,
            "timeout_elapsed_s": self.timeout_elapsed_s,
            "cancellation_policy": self.cancellation_policy,
            "events_count": len(self.events),
        }

    def preview_payload(self, *, safe_idle: bool) -> dict[str, Any]:
        payload = self.status_payload(safe_idle=safe_idle, persistent=False)
        payload["preview_id"] = payload.pop("task_id")
        payload["persistent"] = False
        payload["submit_tool"] = "soridormi.task.submit"
        payload["would_record_task_on_submit"] = True
        return payload


@dataclass
class EmbodiedTaskStore:
    tasks: dict[str, EmbodiedTaskRecord] = field(default_factory=dict)
    client_task_refs: dict[str, str] = field(default_factory=dict)

    def submit_task(
        self,
        args: dict[str, Any],
        *,
        mode: str,
        backend: str,
        emergency_stop: bool,
        safe_idle: bool | None = None,
        skill_registry: SkillExecutionRegistry | None = None,
    ) -> dict[str, Any]:
        body_safe_idle = _effective_safe_idle(
            emergency_stop=emergency_stop,
            safe_idle=safe_idle,
        )
        reject_low_level_fields(args)
        client_ref = _client_task_ref(args)
        request_fingerprint = _task_request_fingerprint(args)
        if client_ref is not None and client_ref in self.client_task_refs:
            existing = self._record(self.client_task_refs[client_ref])
            if existing.request_fingerprint != request_fingerprint:
                raise ValueError(
                    "client_task_ref already exists for a different task payload"
                )
            self._expire_if_timed_out(existing)
            return existing.status_payload(
                safe_idle=body_safe_idle,
                idempotent_replay=True,
            )
        record = self._record_from_args(
            args,
            mode=mode,
            backend=backend,
            emergency_stop=emergency_stop,
            id_prefix="soridormi-task",
            field_context="task submit",
        )
        self._apply_initial_lifecycle(
            record,
            skill_registry=skill_registry,
            safe_idle=body_safe_idle,
        )
        self.tasks[record.task_id] = record
        if record.client_task_ref is not None:
            self.client_task_refs[record.client_task_ref] = record.task_id
        return record.status_payload(safe_idle=body_safe_idle)

    def preview_task(
        self,
        args: dict[str, Any],
        *,
        mode: str,
        backend: str,
        emergency_stop: bool,
        safe_idle: bool | None = None,
        skill_registry: SkillExecutionRegistry | None = None,
    ) -> dict[str, Any]:
        body_safe_idle = _effective_safe_idle(
            emergency_stop=emergency_stop,
            safe_idle=safe_idle,
        )
        record = self._record_from_args(
            args,
            mode=mode,
            backend=backend,
            emergency_stop=emergency_stop,
            id_prefix="soridormi-preview",
            field_context="task preview",
        )
        self._apply_initial_lifecycle(
            record,
            skill_registry=skill_registry,
            safe_idle=body_safe_idle,
        )
        return record.preview_payload(safe_idle=body_safe_idle)

    @staticmethod
    def _record_from_args(
        args: dict[str, Any],
        *,
        mode: str,
        backend: str,
        emergency_stop: bool,
        id_prefix: str,
        field_context: str,
    ) -> EmbodiedTaskRecord:
        reject_low_level_fields(args)
        unknown = set(args) - _ALLOWED_TOP_LEVEL_FIELDS
        if unknown:
            raise ValueError(
                f"unsupported {field_context} field(s): {', '.join(sorted(unknown))}"
            )

        task_type = str(args.get("task_type", "")).strip()
        if not task_type:
            raise ValueError("task_type is required")
        client_ref = _client_task_ref(args)
        request_fingerprint = _task_request_fingerprint(args)
        summary = str(args.get("summary") or task_type)
        parameters = _require_object(args, "parameters")
        task_context = _require_object(args, "task_context")
        environment_context = _require_object(args, "environment_context")
        safety_constraints = _require_object(args, "safety_constraints")
        timeout = _timeout_s(args)
        cancellation_policy = str(args.get("cancellation_policy") or "best_effort_stop")
        if cancellation_policy not in _ALLOWED_CANCELLATION_POLICIES:
            raise ValueError(f"unsupported cancellation_policy: {cancellation_policy}")

        accepted, status, reason_code, reason = EmbodiedTaskStore._classify_task(
            task_type,
            emergency_stop=emergency_stop,
        )
        now = time.time()
        return EmbodiedTaskRecord(
            task_id=f"{id_prefix}-{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            status=status,
            accepted=accepted,
            created_at=now,
            updated_at=now,
            mode=mode,
            backend=backend,
            summary=summary,
            phase="accepted" if accepted else "refused",
            parameters=parameters,
            task_context=task_context,
            environment_context=environment_context,
            safety_constraints=safety_constraints,
            timeout_s=timeout,
            cancellation_policy=cancellation_policy,
            reason_code=reason_code,
            reason=reason,
            client_task_ref=client_ref,
            request_fingerprint=request_fingerprint,
        )

    @staticmethod
    def _apply_initial_lifecycle(
        record: EmbodiedTaskRecord,
        *,
        skill_registry: SkillExecutionRegistry | None,
        safe_idle: bool,
    ) -> None:
        if record.accepted:
            record.append_event("task_accepted")
            EmbodiedTaskStore._prime_lifecycle(
                record,
                skill_registry=skill_registry,
                safe_idle=safe_idle,
            )
        else:
            EmbodiedTaskStore._populate_refusal_plan(record)
            record.append_event("task_refused", message=record.reason)

    def task_status(
        self,
        args: dict[str, Any],
        *,
        emergency_stop: bool,
        safe_idle: bool | None = None,
    ) -> dict[str, Any]:
        record = self._record_for_lookup(args)
        self._expire_if_timed_out(record)
        return record.status_payload(
            safe_idle=_effective_safe_idle(
                emergency_stop=emergency_stop,
                safe_idle=safe_idle,
            )
        )

    def task_events(
        self,
        args: dict[str, Any],
        *,
        emergency_stop: bool,
        safe_idle: bool | None = None,
    ) -> dict[str, Any]:
        record = self._record_for_lookup(args)
        after_sequence = int(args.get("after_sequence", 0) or 0)
        if after_sequence < 0:
            raise ValueError("after_sequence must be >= 0")
        self._expire_if_timed_out(record)
        events = [
            event for event in record.events if int(event["sequence"]) > after_sequence
        ]
        latest_sequence = len(record.events)
        return {
            "schema_version": "soridormi.task_events.v1",
            "task_id": record.task_id,
            "client_task_ref": record.client_task_ref,
            "status": record.status,
            "phase": record.phase,
            "terminal": record.terminal,
            "safe_idle": _effective_safe_idle(
                emergency_stop=emergency_stop,
                safe_idle=safe_idle,
            ),
            "deadline_at": record.deadline_at,
            "expired": record.expired,
            "timeout_elapsed_s": record.timeout_elapsed_s,
            "events": events,
            "returned_count": len(events),
            "latest_sequence": latest_sequence,
            "next_after_sequence": latest_sequence,
            "has_more": False,
            "poll_recommendation": _event_poll_recommendation(
                record,
                next_after_sequence=latest_sequence,
            ),
        }

    def cancel_task(
        self,
        args: dict[str, Any],
        *,
        emergency_stop: bool,
        safe_idle: bool | None = None,
    ) -> dict[str, Any]:
        body_safe_idle = _effective_safe_idle(
            emergency_stop=emergency_stop,
            safe_idle=safe_idle,
        )
        record = self._record_for_lookup(args)
        self._expire_if_timed_out(record)
        if record.status in TERMINAL_STATUSES or record.terminal:
            return {
                "task_id": record.task_id,
                "client_task_ref": record.client_task_ref,
                "cancelled": False,
                "status": record.status,
                "phase": record.phase,
                "terminal": record.terminal,
                "safe_idle": body_safe_idle,
                "reason_code": record.reason_code,
            }
        reason = str(args.get("reason") or "Task cancelled by caller.")
        record.transition_to(
            "cancelled",
            event_type="task_cancelled",
            reason_code="operator_cancelled",
            reason=reason,
            accepted=False,
        )
        return {
            "task_id": record.task_id,
            "client_task_ref": record.client_task_ref,
            "cancelled": True,
            "status": record.status,
            "phase": record.phase,
            "terminal": record.terminal,
            "safe_idle": body_safe_idle,
            "reason_code": record.reason_code,
        }

    def _record(self, task_id: str) -> EmbodiedTaskRecord:
        record = self.tasks.get(task_id)
        if record is None:
            raise KeyError(f"task not found: {task_id}")
        return record

    def _record_for_lookup(self, args: dict[str, Any]) -> EmbodiedTaskRecord:
        task_id = str(args.get("task_id", "") or "").strip()
        client_ref = _client_task_ref(args) if "client_task_ref" in args else None
        if task_id and client_ref:
            raise ValueError("provide either task_id or client_task_ref, not both")
        if client_ref:
            mapped_task_id = self.client_task_refs.get(client_ref)
            if mapped_task_id is None:
                raise KeyError(f"task not found for client_task_ref: {client_ref}")
            return self._record(mapped_task_id)
        if not task_id:
            raise ValueError("task_id or client_task_ref is required")
        return self._record(task_id)

    @staticmethod
    def _expire_if_timed_out(record: EmbodiedTaskRecord) -> None:
        if record.terminal:
            return
        now = time.time()
        if now <= record.deadline_at:
            return
        elapsed = max(0.0, now - record.created_at)
        overrun = max(0.0, now - record.deadline_at)
        if record.cancellation_policy == "emergency_stop_on_timeout":
            reason_code = "task_timeout_emergency_stop_required"
            reason = (
                "Task exceeded timeout_s; caller should use the dedicated "
                "emergency-stop path if physical motion may be active."
            )
        else:
            reason_code = "task_timeout"
            reason = "Task exceeded timeout_s at the no-motion planning boundary."
        record.timeout_elapsed_s = elapsed
        record.transition_to(
            "failed",
            event_type="task_timed_out",
            reason_code=reason_code,
            reason=reason,
            accepted=False,
            details={
                "timeout_s": record.timeout_s,
                "elapsed_s": elapsed,
                "overrun_s": overrun,
                "cancellation_policy": record.cancellation_policy,
                "no_motion": record.no_motion,
            },
        )

    @staticmethod
    def _prime_lifecycle(
        record: EmbodiedTaskRecord,
        *,
        skill_registry: SkillExecutionRegistry | None,
        safe_idle: bool,
    ) -> None:
        record.transition_to(
            "resolving",
            event_type="task_resolving",
            message="Task accepted; resolving structured embodied goal.",
        )
        record.transition_to(
            "planning",
            event_type="task_planning",
            message="Task moved into Soridormi-owned embodied planning skeleton.",
        )
        record.plan_steps = _contract_only_plan_steps(record)
        if record.task_type in SAFE_IDLE_TASK_TYPES:
            if safe_idle:
                record.transition_to(
                    "completed",
                    event_type="task_completed",
                    message="Live body state confirms safe idle; no motion was sent.",
                )
            else:
                record.transition_to(
                    "failed",
                    event_type="task_failed",
                    reason_code="robot_not_safe_idle",
                    reason=(
                        "Live body state does not confirm safe idle; "
                        "Soridormi will not manufacture a safe-idle result."
                    ),
                    accepted=False,
                )
            return
        if record.task_type in SKILL_BACKED_TASK_TYPES and skill_registry is not None:
            EmbodiedTaskStore._complete_skill_dry_run(record, skill_registry)
            return
        record.append_event(
            "task_execution_held",
            message=(
                "Task execution is held at the no-motion planning boundary until "
                "the embodied task executor is implemented."
            ),
        )

    @staticmethod
    def _complete_skill_dry_run(
        record: EmbodiedTaskRecord,
        skill_registry: SkillExecutionRegistry,
    ) -> None:
        if record.task_type == "skill_sequence":
            EmbodiedTaskStore._complete_skill_sequence_dry_run(record, skill_registry)
            return
        try:
            skill_id, parameters = _skill_request_for_task(record)
            plan = skill_registry.create_plan(skill_id, parameters)
        except (SkillExecutionError, ValueError) as exc:
            record.transition_to(
                "failed",
                event_type="task_failed",
                reason_code="skill_planning_failed",
                reason=str(exc),
                accepted=False,
            )
            return

        record.skill_id = plan.skill_id
        record.skill_summary = plan.summary
        record.estimated_duration_s = plan.total_duration_s
        record.execution_mode = "skill_dry_run"
        record.plan_steps = [
            _plan_step(
                1,
                layer="skill",
                kind="skill_execution",
                status="dry_run_completed",
                summary=plan.summary,
                skill_id=plan.skill_id,
                estimated_duration_s=plan.total_duration_s,
            )
        ]
        record.transition_to(
            "executing",
            event_type="task_executing",
            message="Created Soridormi skill-backed dry-run plan; no robot command was sent.",
            details={
                "skill_id": plan.skill_id,
                "estimated_duration_s": plan.total_duration_s,
                "dry_run": True,
            },
        )
        record.transition_to(
            "monitoring",
            event_type="task_monitoring",
            message="Skill-backed dry-run plan passed contract monitoring.",
            details={"no_motion": True},
        )
        record.transition_to(
            "completed",
            event_type="task_completed",
            message="Skill-backed dry-run task completed; no robot command was sent.",
            details={
                "skill_id": plan.skill_id,
                "estimated_duration_s": plan.total_duration_s,
                "no_motion": True,
            },
        )

    @staticmethod
    def _complete_skill_sequence_dry_run(
        record: EmbodiedTaskRecord,
        skill_registry: SkillExecutionRegistry,
    ) -> None:
        try:
            requested_steps = _skill_sequence_for_task(record)
            plans = [
                skill_registry.create_plan(skill_id, parameters)
                for skill_id, parameters in requested_steps
            ]
        except (SkillExecutionError, ValueError) as exc:
            record.transition_to(
                "failed",
                event_type="task_failed",
                reason_code="skill_sequence_planning_failed",
                reason=str(exc),
                accepted=False,
            )
            return

        total_duration = sum(plan.total_duration_s for plan in plans)
        record.execution_mode = "skill_sequence_dry_run"
        record.skill_sequence = [
            {
                "step_index": index,
                "skill_id": plan.skill_id,
                "summary": plan.summary,
                "estimated_duration_s": plan.total_duration_s,
            }
            for index, plan in enumerate(plans, start=1)
        ]
        record.plan_steps = [
            _plan_step(
                index,
                layer="skill",
                kind="skill_execution",
                status="dry_run_completed",
                summary=plan.summary,
                skill_id=plan.skill_id,
                estimated_duration_s=plan.total_duration_s,
            )
            for index, plan in enumerate(plans, start=1)
        ]
        record.estimated_duration_s = total_duration
        record.transition_to(
            "executing",
            event_type="task_executing",
            message="Created Soridormi skill-sequence dry-run plan; no robot command was sent.",
            details={
                "sequence_length": len(plans),
                "skill_ids": [plan.skill_id for plan in plans],
                "estimated_duration_s": total_duration,
                "dry_run": True,
            },
        )
        record.transition_to(
            "monitoring",
            event_type="task_monitoring",
            message="Skill-sequence dry-run plan passed contract monitoring.",
            details={"no_motion": True},
        )
        record.transition_to(
            "completed",
            event_type="task_completed",
            message="Skill-sequence dry-run task completed; no robot command was sent.",
            details={
                "sequence_length": len(plans),
                "skill_ids": [plan.skill_id for plan in plans],
                "estimated_duration_s": total_duration,
                "no_motion": True,
            },
        )

    @staticmethod
    def _classify_task(
        task_type: str,
        *,
        emergency_stop: bool,
    ) -> tuple[bool, str, str | None, str | None]:
        if task_type in UNSAFE_TASK_TYPES:
            return False, "refused", "unsafe_task", "Unsafe physical task refused."
        if task_type not in KNOWN_TASK_TYPES:
            return False, "refused", "unsupported_task_type", f"Unsupported task type: {task_type}."
        if emergency_stop and task_type != "stop_now":
            return False, "refused", "emergency_stop_active", "Emergency stop is active."
        if task_type in FUTURE_BLOCKED_TASKS:
            reason_code = FUTURE_BLOCKED_TASKS[task_type]
            return False, "refused", reason_code, f"{task_type} is declared but not executable yet."
        if task_type in STOP_REDIRECT_TASK_TYPES:
            reason_code = str(TASK_REASON_CODES.get(task_type) or "use_safety_tool_for_immediate_stop")
            return False, "refused", reason_code, (
                "Use soridormi.motion.stop, soridormi.motion.cancel, or "
                "soridormi.safety.emergency_stop for immediate stop behavior."
            )
        if task_type in SAFE_IDLE_TASK_TYPES:
            if emergency_stop:
                return False, "refused", "manual_emergency_recovery_required", (
                    "Emergency-stop recovery requires an explicit operator path."
                )
            return True, "completed", None, None
        if task_type in CONTRACT_ONLY_TASK_TYPES or task_type in SKILL_BACKED_TASK_TYPES:
            return True, "accepted", None, None
        return False, "refused", "unsupported_task_type", f"Unsupported task type: {task_type}."

    @staticmethod
    def _populate_refusal_plan(record: EmbodiedTaskRecord) -> None:
        record.plan_steps = _blocked_plan_steps_for_task(record.task_type)
        record.blocked_subsystems = list(FUTURE_BLOCKED_SUBSYSTEMS.get(record.task_type, []))
        if record.task_type in UNSAFE_TASK_TYPES:
            record.blocked_subsystems = ["human_safety_policy"]
        elif record.task_type in STOP_REDIRECT_TASK_TYPES:
            record.blocked_subsystems = list(
                TASK_DECLARED_MISSING_SUBSYSTEMS.get(
                    record.task_type,
                    ["dedicated_stop_tool_required"],
                )
            )
