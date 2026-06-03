"""Dry-run skill execution registry for Soridormi M7.

This module is intentionally conservative: it resolves manifest-declared skills
into safe, inspectable command plans, but it does not connect to MuJoCo,
hardware, or motor commands.  It is the boundary between the skill vocabulary
and future runtime/MCP execution.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from soridormi_runtime.skill_manifest import (
    AVAILABLE_STATUSES,
    DEFAULT_SKILL_MANIFEST,
    SkillManifestError,
    load_skill_manifest,
    skills_by_id,
    validate_skill_manifest,
)


DEFAULT_SKILL_PROFILE = "open_duck_forward"


class SkillExecutionError(ValueError):
    """Raised when a skill cannot be resolved into a safe command plan."""


@dataclass(frozen=True)
class VelocitySegment:
    """High-level velocity segment emitted by a locomotion skill."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    yaw_radps: float = 0.0
    duration_s: float = 1.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "yaw_radps": self.yaw_radps,
            "duration_s": self.duration_s,
            "label": self.label,
        }


@dataclass(frozen=True)
class JointKeyframeSegment:
    """Scripted joint target segment emitted by a non-locomotion skill."""

    positions_by_name: Mapping[str, float]
    duration_s: float = 1.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions_by_name": {name: float(value) for name, value in self.positions_by_name.items()},
            "duration_s": self.duration_s,
            "label": self.label,
        }


@dataclass(frozen=True)
class SkillPlan:
    """Dry-run plan produced from a manifest-declared skill."""

    skill_id: str
    status: str
    category: str
    execution: str
    profile: str
    dry_run: bool
    summary: str
    commands: tuple[VelocitySegment, ...] = ()
    keyframes: tuple[JointKeyframeSegment, ...] = ()
    safety: Mapping[str, Any] | None = None
    parameters: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "category": self.category,
            "execution": self.execution,
            "profile": self.profile,
            "dry_run": self.dry_run,
            "summary": self.summary,
            "commands": [command.to_dict() for command in self.commands],
            "keyframes": [keyframe.to_dict() for keyframe in self.keyframes],
            "safety": dict(self.safety or {}),
            "parameters": dict(self.parameters or {}),
            "total_duration_s": self.total_duration_s,
        }

    @property
    def total_duration_s(self) -> float:
        return sum(command.duration_s for command in self.commands) + sum(
            keyframe.duration_s for keyframe in self.keyframes
        )


SkillPlanner = Callable[[dict[str, Any], Mapping[str, Any], str], SkillPlan]


def _coerce_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SkillExecutionError(f"parameter {name} must be numeric, got boolean")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SkillExecutionError(f"parameter {name} must be numeric") from exc


def _resolve_parameters(skill: dict[str, Any], provided: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(provided or {})
    spec = skill.get("parameters", {})
    if not isinstance(spec, dict):
        raise SkillExecutionError(f"skill {skill.get('id')}: parameters spec must be an object")

    unknown = set(provided) - set(spec)
    if unknown:
        raise SkillExecutionError(f"skill {skill.get('id')}: unknown parameters {sorted(unknown)}")

    resolved: dict[str, Any] = {}
    for name, rule in spec.items():
        if not isinstance(rule, dict):
            continue
        if name in provided:
            value = provided[name]
        elif "default" in rule:
            value = rule["default"]
        else:
            raise SkillExecutionError(f"skill {skill.get('id')}: missing required parameter {name}")

        param_type = rule.get("type")
        # Existing M7 manifests often declare numeric parameters with min/max/default
        # only. Treat those as number parameters to keep the manifest compact.
        if param_type is None and any(key in rule for key in ("min", "max")):
            param_type = "number"
        if param_type == "number":
            number = _coerce_number(name, value)
            if "min" in rule and number < float(rule["min"]):
                raise SkillExecutionError(
                    f"skill {skill.get('id')}: parameter {name}={number} below min {rule['min']}"
                )
            if "max" in rule and number > float(rule["max"]):
                raise SkillExecutionError(
                    f"skill {skill.get('id')}: parameter {name}={number} above max {rule['max']}"
                )
            resolved[name] = number
        elif param_type == "string":
            text = str(value)
            allowed = rule.get("enum")
            if isinstance(allowed, list) and allowed and text not in {str(item) for item in allowed}:
                raise SkillExecutionError(
                    f"skill {skill.get('id')}: parameter {name}={text!r} not in enum {allowed!r}"
                )
            resolved[name] = text
        elif param_type is None:
            resolved[name] = value
        else:
            raise SkillExecutionError(f"skill {skill.get('id')}: unsupported parameter type {param_type!r}")
    return resolved


def _require_available(skill: dict[str, Any]) -> None:
    status = str(skill.get("status"))
    if status not in AVAILABLE_STATUSES:
        raise SkillExecutionError(
            f"skill {skill.get('id')} is not executable yet: status={status}. "
            "Only available_sim and available_sim_experimental skills can be dry-run planned."
        )
    safety = skill.get("safety", {})
    if not isinstance(safety, dict) or safety.get("hardware_enabled") is not False:
        raise SkillExecutionError(f"skill {skill.get('id')} is not safe for M7 sim-first execution")


def _velocity_skill_plan(
    skill: dict[str, Any],
    parameters: Mapping[str, Any],
    profile: str,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    yaw: float = 0.0,
    duration: float = 1.0,
) -> SkillPlan:
    skill_id = str(skill["id"])
    command = VelocitySegment(vx_mps=vx, vy_mps=vy, yaw_radps=yaw, duration_s=duration, label=skill_id)
    summary = (
        f"Dry-run {skill_id}: vx={vx:.3f} m/s, vy={vy:.3f} m/s, "
        f"yaw={yaw:.3f} rad/s for {duration:.2f}s using profile {profile}."
    )
    return SkillPlan(
        skill_id=skill_id,
        status=str(skill.get("status")),
        category=str(skill.get("category")),
        execution=str(skill.get("execution")),
        profile=profile,
        dry_run=True,
        summary=summary,
        commands=(command,),
        safety=skill.get("safety", {}),
        parameters=parameters,
    )


def _plan_stand_idle(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    duration = float(parameters.get("duration_s", 2.0))
    return _velocity_skill_plan(skill, parameters, profile, duration=duration)


def _plan_stop(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    duration = float(parameters.get("duration_s", 1.0))
    return _velocity_skill_plan(skill, parameters, profile, duration=duration)


def _plan_walk_velocity(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    return _velocity_skill_plan(
        skill,
        parameters,
        profile,
        vx=float(parameters.get("vx_mps", 0.0)),
        vy=float(parameters.get("vy_mps", 0.0)),
        yaw=float(parameters.get("yaw_radps", 0.0)),
        duration=float(parameters.get("duration_s", 2.0)),
    )


def _plan_turn_in_place(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    return _velocity_skill_plan(
        skill,
        parameters,
        profile,
        yaw=float(parameters.get("yaw_radps", 0.0)),
        duration=float(parameters.get("duration_s", 2.0)),
    )


def _plan_curve_walk(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    return _velocity_skill_plan(
        skill,
        parameters,
        profile,
        vx=float(parameters.get("vx_mps", 0.0)),
        yaw=float(parameters.get("yaw_radps", 0.0)),
        duration=float(parameters.get("duration_s", 3.0)),
    )


def _plan_sidestep(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    return _velocity_skill_plan(
        skill,
        parameters,
        profile,
        vy=float(parameters.get("vy_mps", 0.0)),
        duration=float(parameters.get("duration_s", 2.0)),
    )


def _scripted_keyframe_plan(
    skill: dict[str, Any],
    parameters: Mapping[str, Any],
    profile: str,
    *,
    keyframes: Sequence[JointKeyframeSegment],
    summary: str,
) -> SkillPlan:
    return SkillPlan(
        skill_id=str(skill["id"]),
        status=str(skill.get("status")),
        category=str(skill.get("category")),
        execution=str(skill.get("execution")),
        profile=profile,
        dry_run=True,
        summary=summary,
        keyframes=tuple(keyframes),
        safety=skill.get("safety", {}),
        parameters=parameters,
    )


def _head_keyframe(*, head_pitch: float = 0.0, head_yaw: float = 0.0, duration_s: float, label: str) -> JointKeyframeSegment:
    return JointKeyframeSegment(
        positions_by_name={
            "neck_pitch": 0.0,
            "head_pitch": float(head_pitch),
            "head_yaw": float(head_yaw),
            "head_roll": 0.0,
        },
        duration_s=float(duration_s),
        label=label,
    )


def _amplitude_radians(value: Any, *, small: float, medium: float) -> float:
    text = str(value or "small")
    if text == "small":
        return float(small)
    if text == "medium":
        return float(medium)
    raise SkillExecutionError(f"unsupported scripted head amplitude: {text!r}")


def _count_cycles(value: Any, *, minimum: int = 1, maximum: int = 8) -> int:
    count = float(value)
    if not count.is_integer():
        raise SkillExecutionError("scripted social count must be an integer number of cycles")
    result = int(count)
    if result < minimum:
        raise SkillExecutionError(f"scripted social count must be at least {minimum} cycles")
    if result > maximum:
        raise SkillExecutionError(f"scripted social count must be at most {maximum} cycles")
    return result


def _plan_neutral_head(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    duration = float(parameters.get("duration_s", 3.0))
    keyframe = _head_keyframe(duration_s=duration, label=skill_id)
    summary = (
        f"Plan {skill_id}: return head/neck joints to neutral straight-ahead pose "
        f"over {duration:.2f}s using a scripted head trajectory."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=(keyframe,), summary=summary)


def _plan_look_direction(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    head_yaw = float(parameters.get("head_yaw_rad", 0.0))
    head_pitch = float(parameters.get("head_pitch_rad", 0.0))
    duration = float(parameters.get("duration_s", 1.0))
    keyframe = _head_keyframe(head_pitch=head_pitch, head_yaw=head_yaw, duration_s=duration, label=skill_id)
    summary = (
        f"Plan {skill_id}: head_yaw={head_yaw:.3f} rad, "
        f"head_pitch={head_pitch:.3f} rad over {duration:.2f}s using scripted head keyframes."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=(keyframe,), summary=summary)


def _plan_look_at_person(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    target_yaw = float(parameters.get("target_yaw_rad", 0.0))
    target_pitch = float(parameters.get("target_pitch_rad", 0.0))
    target_ref = str(parameters.get("target_ref", "person") or "person")
    duration = float(parameters.get("duration_s", 4.0))
    hold_fraction = float(parameters.get("hold_fraction", 0.50))
    end_mode = str(parameters.get("end_mode", "hold_target") or "hold_target")
    if end_mode not in {"hold_target", "return_neutral"}:
        raise SkillExecutionError("look_at_person end_mode must be 'hold_target' or 'return_neutral'")
    if not 0.0 <= hold_fraction <= 0.8:
        raise SkillExecutionError("look_at_person hold_fraction must be between 0.0 and 0.8")
    settle_duration = duration * 0.15
    hold_duration = duration * hold_fraction
    remaining_duration = duration - settle_duration - hold_duration
    move_duration = remaining_duration / 2.0 if end_mode == "return_neutral" else remaining_duration
    if move_duration <= 0.0:
        raise SkillExecutionError("look_at_person duration_s is too short for the requested hold_fraction")
    keyframes: list[JointKeyframeSegment] = [
        _head_keyframe(duration_s=settle_duration, label=f"{skill_id}_neutral_start"),
        _head_keyframe(
            head_pitch=target_pitch,
            head_yaw=target_yaw,
            duration_s=move_duration,
            label=f"{skill_id}_acquire_target",
        ),
        _head_keyframe(
            head_pitch=target_pitch,
            head_yaw=target_yaw,
            duration_s=hold_duration if hold_duration > 0.0 else 1e-6,
            label=f"{skill_id}_hold_target",
        ),
    ]
    if end_mode == "return_neutral":
        keyframes.append(_head_keyframe(duration_s=move_duration, label=f"{skill_id}_neutral_end"))
    summary = (
        f"Plan {skill_id}: look toward structured target {target_ref!r} "
        f"with target_yaw={target_yaw:.3f} rad and target_pitch={target_pitch:.3f} rad "
        f"over {duration:.2f}s using a scripted head trajectory; "
        f"end_mode={end_mode}; no perception is run."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=tuple(keyframes), summary=summary)


def _plan_nod_yes(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    count = _count_cycles(parameters.get("count", 2), minimum=2, maximum=8)
    down_amplitude = _amplitude_radians(parameters.get("amplitude", "small"), small=0.18, medium=0.26)
    up_amplitude = _amplitude_radians(parameters.get("amplitude", "small"), small=0.12, medium=0.18)
    duration = float(parameters.get("duration_s", 4.0))
    segment_duration = duration / float(max(1, count * 2 + 2))
    keyframes: list[JointKeyframeSegment] = [
        _head_keyframe(duration_s=segment_duration, label=f"{skill_id}_neutral_start")
    ]
    for index in range(count):
        keyframes.append(_head_keyframe(head_pitch=-down_amplitude, duration_s=segment_duration, label=f"{skill_id}_down_{index + 1}"))
        keyframes.append(_head_keyframe(head_pitch=up_amplitude, duration_s=segment_duration, label=f"{skill_id}_up_{index + 1}"))
    keyframes.append(_head_keyframe(duration_s=segment_duration, label=f"{skill_id}_neutral_end"))
    summary = (
        f"Plan {skill_id}: {count} visible nod cycle(s), amplitude={parameters.get('amplitude', 'small')} "
        f"(down={down_amplitude:.3f} rad, up={up_amplitude:.3f} rad pitch offsets) over "
        f"{duration:.2f}s using scripted head keyframes."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=keyframes, summary=summary)


def _plan_shake_no(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    count = _count_cycles(parameters.get("count", 2), minimum=2, maximum=8)
    amplitude = _amplitude_radians(parameters.get("amplitude", "small"), small=0.28, medium=0.40)
    duration = float(parameters.get("duration_s", 4.0))
    segment_duration = duration / float(max(1, count * 2 + 2))
    keyframes: list[JointKeyframeSegment] = [
        _head_keyframe(duration_s=segment_duration, label=f"{skill_id}_neutral_start")
    ]
    for index in range(count):
        keyframes.append(_head_keyframe(head_yaw=amplitude, duration_s=segment_duration, label=f"{skill_id}_right_{index + 1}"))
        keyframes.append(_head_keyframe(head_yaw=-amplitude, duration_s=segment_duration, label=f"{skill_id}_left_{index + 1}"))
    keyframes.append(_head_keyframe(duration_s=segment_duration, label=f"{skill_id}_neutral_end"))
    summary = (
        f"Plan {skill_id}: {count} visible shake cycle(s), amplitude={parameters.get('amplitude', 'small')} "
        f"({amplitude:.3f} rad yaw offsets) over {duration:.2f}s using scripted head keyframes."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=keyframes, summary=summary)


def _plan_bow(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    depth = str(parameters.get("depth", "small") or "small")
    if depth == "small":
        neck_pitch = -0.06
        head_pitch = -0.18
    elif depth == "medium":
        neck_pitch = -0.10
        head_pitch = -0.26
    else:
        raise SkillExecutionError(f"unsupported bow depth: {depth!r}")
    duration = float(parameters.get("duration_s", 5.0))
    hold_fraction = float(parameters.get("hold_fraction", 0.35))
    if not 0.0 <= hold_fraction <= 0.8:
        raise SkillExecutionError("bow hold_fraction must be between 0.0 and 0.8")
    settle_duration = duration * 0.10
    hold_duration = duration * hold_fraction
    move_duration = (duration - settle_duration - hold_duration) / 2.0
    if move_duration <= 0.0:
        raise SkillExecutionError("bow duration_s is too short for the requested hold_fraction")
    keyframes = (
        _head_keyframe(duration_s=settle_duration, label=f"{skill_id}_neutral_start"),
        _head_keyframe(
            head_pitch=head_pitch,
            duration_s=move_duration,
            label=f"{skill_id}_down",
        ),
        _head_keyframe(
            head_pitch=head_pitch,
            duration_s=hold_duration if hold_duration > 0.0 else 1e-6,
            label=f"{skill_id}_hold",
        ),
        _head_keyframe(duration_s=move_duration, label=f"{skill_id}_neutral_end"),
    )
    # Add neck pitch after using the common helper so all non-moving axes remain
    # explicitly neutral. The gesture is head/neck only; no torso/leg bow is
    # attempted until whole-body posture control is validated.
    keyframes = tuple(
        JointKeyframeSegment(
            positions_by_name={
                **keyframe.positions_by_name,
                "neck_pitch": neck_pitch
                if keyframe.label in {f"{skill_id}_down", f"{skill_id}_hold"}
                else 0.0,
            },
            duration_s=keyframe.duration_s,
            label=keyframe.label,
        )
        for keyframe in keyframes
    )
    summary = (
        f"Plan {skill_id}: gentle {depth} head/neck bow over {duration:.2f}s "
        f"(neck_pitch={neck_pitch:.3f} rad, head_pitch={head_pitch:.3f} rad) "
        "using a scripted head trajectory; no torso or leg motion."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=keyframes, summary=summary)


def _plan_express_attention(skill: dict[str, Any], parameters: Mapping[str, Any], profile: str) -> SkillPlan:
    skill_id = str(skill["id"])
    style = str(parameters.get("style", "neutral") or "neutral")
    if style == "neutral":
        head_pitch = -0.07
        head_yaw = 0.0
    elif style == "curious":
        head_pitch = -0.06
        head_yaw = 0.14
    else:
        raise SkillExecutionError(f"unsupported attention style: {style!r}")
    duration = float(parameters.get("duration_s", 4.0))
    hold_fraction = float(parameters.get("hold_fraction", 0.45))
    if not 0.0 <= hold_fraction <= 0.8:
        raise SkillExecutionError("express_attention hold_fraction must be between 0.0 and 0.8")
    settle_duration = duration * 0.15
    hold_duration = duration * hold_fraction
    move_duration = (duration - settle_duration - hold_duration) / 2.0
    if move_duration <= 0.0:
        raise SkillExecutionError("express_attention duration_s is too short for the requested hold_fraction")
    keyframes = (
        _head_keyframe(duration_s=settle_duration, label=f"{skill_id}_neutral_start"),
        _head_keyframe(
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            duration_s=move_duration,
            label=f"{skill_id}_{style}_focus",
        ),
        _head_keyframe(
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            duration_s=hold_duration if hold_duration > 0.0 else 1e-6,
            label=f"{skill_id}_{style}_hold",
        ),
        _head_keyframe(duration_s=move_duration, label=f"{skill_id}_neutral_end"),
    )
    summary = (
        f"Plan {skill_id}: {style} attention/listening cue over {duration:.2f}s "
        f"(head_pitch={head_pitch:.3f} rad, head_yaw={head_yaw:.3f} rad) "
        "using a scripted head-only trajectory."
    )
    return _scripted_keyframe_plan(skill, parameters, profile, keyframes=keyframes, summary=summary)


BUILTIN_SKILL_PLANNERS: dict[str, SkillPlanner] = {
    "stand_idle": _plan_stand_idle,
    "stop": _plan_stop,
    "walk_velocity": _plan_walk_velocity,
    "turn_in_place": _plan_turn_in_place,
    "curve_walk": _plan_curve_walk,
    "sidestep": _plan_sidestep,
    "neutral_head": _plan_neutral_head,
    "look_direction": _plan_look_direction,
    "look_at_person": _plan_look_at_person,
    "nod_yes": _plan_nod_yes,
    "shake_no": _plan_shake_no,
    "bow": _plan_bow,
    "express_attention": _plan_express_attention,
}


class SkillExecutionRegistry:
    """Registry that resolves manifest skills into safe dry-run command plans."""

    def __init__(
        self,
        manifest: dict[str, Any],
        planners: Mapping[str, SkillPlanner] | None = None,
        *,
        default_profile: str = DEFAULT_SKILL_PROFILE,
    ) -> None:
        validation = validate_skill_manifest(manifest)
        if not validation.ok:
            raise SkillManifestError("Invalid skill manifest: " + "; ".join(validation.errors))
        self.manifest = manifest
        self.skills = skills_by_id(manifest)
        self.planners = dict(planners or BUILTIN_SKILL_PLANNERS)
        self.default_profile = default_profile

    @classmethod
    def from_manifest_path(
        cls,
        path: str | Path = DEFAULT_SKILL_MANIFEST,
        *,
        default_profile: str = DEFAULT_SKILL_PROFILE,
    ) -> "SkillExecutionRegistry":
        return cls(load_skill_manifest(path), default_profile=default_profile)

    def executable_skill_ids(self) -> tuple[str, ...]:
        ids = []
        for skill_id, skill in sorted(self.skills.items()):
            if skill.get("status") in AVAILABLE_STATUSES and skill_id in self.planners:
                ids.append(skill_id)
        return tuple(ids)

    def create_plan(
        self,
        skill_id: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        profile: str | None = None,
    ) -> SkillPlan:
        skill = self.skills.get(skill_id)
        if skill is None:
            raise SkillExecutionError(f"unknown skill: {skill_id}")
        _require_available(skill)
        planner = self.planners.get(skill_id)
        if planner is None:
            raise SkillExecutionError(f"skill {skill_id} has no registered planner yet")
        resolved_parameters = _resolve_parameters(skill, parameters)
        return planner(skill, resolved_parameters, profile or self.default_profile)


def _load_json_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SkillExecutionError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SkillExecutionError("--args must decode to a JSON object")
    return value




def plan_shell_exports(plan: SkillPlan) -> str:
    """Return shell exports that bind a one-segment skill plan to runtime command overrides.

    M7D intentionally supports only single-segment locomotion skill execution.
    Multi-segment choreography should be added through a later skill scheduler so
    each segment can be logged, interrupted, and safety-gated separately.
    """

    if len(plan.commands) != 1:
        raise SkillExecutionError(
            f"skill {plan.skill_id} produced {len(plan.commands)} command segments; "
            "M7D sim execution supports exactly one segment"
        )
    command = plan.commands[0]
    exports = {
        "SORIDORMI_SKILL_ID": plan.skill_id,
        "SORIDORMI_SKILL_PROFILE": plan.profile,
        "SORIDORMI_SKILL_DURATION_SECONDS": f"{command.duration_s:.10g}",
        "SORIDORMI_COMMAND_X_OVERRIDE": f"{command.vx_mps:.10g}",
        "SORIDORMI_COMMAND_Y_OVERRIDE": f"{command.vy_mps:.10g}",
        "SORIDORMI_COMMAND_YAW_OVERRIDE": f"{command.yaw_radps:.10g}",
        "SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE": "0",
        "SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE": f"skill_{plan.skill_id}",
    }
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(exports.items()))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run a Soridormi M7 skill into a safe high-level command plan."
    )
    parser.add_argument("skill", nargs="?", help="Skill id to dry-run, e.g. walk_velocity.")
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument("--profile", default=DEFAULT_SKILL_PROFILE, help="Policy profile hint for locomotion skills.")
    parser.add_argument("--args", default="{}", help="Skill parameter JSON object.")
    parser.add_argument("--list", action="store_true", help="List executable dry-run skill ids.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--shell-env",
        action="store_true",
        help="Print shell exports for single-segment MuJoCo skill execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    registry = SkillExecutionRegistry.from_manifest_path(args.manifest, default_profile=args.profile)

    if args.list:
        payload = {"executable_skills": list(registry.executable_skill_ids())}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Soridormi executable dry-run skills")
            print("===================================")
            for skill_id in payload["executable_skills"]:
                print(f"- {skill_id}")
        return 0

    if not args.skill:
        parser.error("skill is required unless --list is used")

    try:
        parameters = _load_json_args(args.args)
        plan = registry.create_plan(args.skill, parameters, profile=args.profile)
    except SkillExecutionError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"Skill dry-run failed: {exc}")
        return 2

    if args.shell_env:
        try:
            print(plan_shell_exports(plan))
        except SkillExecutionError as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
            else:
                print(f"Skill dry-run failed: {exc}")
            return 2
    elif args.json:
        print(json.dumps({"ok": True, "plan": plan.to_dict()}, indent=2, sort_keys=True))
    else:
        print("Soridormi skill dry-run")
        print("========================")
        print(plan.summary)
        if plan.commands:
            print("Commands:")
            for command in plan.commands:
                print(
                    f"- {command.label}: vx={command.vx_mps:.3f} vy={command.vy_mps:.3f} "
                    f"yaw={command.yaw_radps:.3f} duration={command.duration_s:.2f}s"
                )
        if plan.keyframes:
            print("Keyframes:")
            for keyframe in plan.keyframes:
                targets = ", ".join(
                    f"{name}={value:.3f}" for name, value in sorted(keyframe.positions_by_name.items())
                )
                print(f"- {keyframe.label}: {targets} duration={keyframe.duration_s:.2f}s")
        print("No robot, simulator, or hardware command was executed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
