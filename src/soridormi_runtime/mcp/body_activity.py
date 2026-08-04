from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from soridormi_runtime.skill_execution import SkillExecutionRegistry, SkillPlan


ABILITY_CLASS_LOCOMOTION = "locomotion_whole_body"
ABILITY_CLASS_SUBTLE_EXPRESSION = "subtle_expression"
BODY_ABILITY_CLASSES = {
    ABILITY_CLASS_LOCOMOTION,
    ABILITY_CLASS_SUBTLE_EXPRESSION,
}
CONTROL_COUPLING_PRIMARY = "primary_body_controller"
CONTROL_COUPLING_OVERLAY = "body_command_overlay"
CONTROL_COUPLING_INDEPENDENT = "independent_output"
CONTROL_COUPLING_STANDALONE = "standalone_body_motion"
CONTROL_COUPLINGS = {
    CONTROL_COUPLING_PRIMARY,
    CONTROL_COUPLING_OVERLAY,
    CONTROL_COUPLING_INDEPENDENT,
    CONTROL_COUPLING_STANDALONE,
}
MAX_ACTIVITY_MEMBERS = 8
ACTIVITY_STATUSES = {
    "planned",
    "running",
    "completed",
    "completed_with_degradation",
    "cancelled",
    "failed",
}
TERMINAL_ACTIVITY_STATUSES = {
    "completed",
    "completed_with_degradation",
    "cancelled",
    "failed",
}


@dataclass(frozen=True)
class BodyActivityMemberPlan:
    member_id: str
    skill_id: str
    plan: SkillPlan
    optional: bool
    ability_class: str
    control_coupling: str
    write_resources: tuple[str, ...]
    concurrency_envelope: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "skill_id": self.skill_id,
            "optional": self.optional,
            "ability_class": self.ability_class,
            "control_coupling": self.control_coupling,
            "write_resources": list(self.write_resources),
            "concurrency_envelope": dict(self.concurrency_envelope),
            "execution": self.plan.execution,
            "parameters": dict(self.plan.parameters or {}),
            "estimated_duration_s": self.plan.total_duration_s,
            "summary": self.plan.summary,
        }


@dataclass
class BodyActivityPlanRecord:
    plan_id: str
    coordination_id: str | None
    members: tuple[BodyActivityMemberPlan, ...]
    created_at: float
    estimated_duration_s: float
    status: str = "planned"
    started_at: float | None = None
    completed_at: float | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None
    member_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_ACTIVITY_STATUSES

    @property
    def primary_member(self) -> BodyActivityMemberPlan | None:
        return next(
            (
                member
                for member in self.members
                if member.control_coupling == CONTROL_COUPLING_PRIMARY
            ),
            None,
        )

    @property
    def head_overlay_member(self) -> BodyActivityMemberPlan | None:
        return next(
            (
                member
                for member in self.members
                if member.control_coupling == CONTROL_COUPLING_OVERLAY
            ),
            None,
        )

    @property
    def independent_members(self) -> tuple[BodyActivityMemberPlan, ...]:
        return tuple(
            member
            for member in self.members
            if member.control_coupling == CONTROL_COUPLING_INDEPENDENT
        )

    @property
    def standalone_members(self) -> tuple[BodyActivityMemberPlan, ...]:
        return tuple(
            member
            for member in self.members
            if member.control_coupling == CONTROL_COUPLING_STANDALONE
        )

    def to_dict(self, *, mode: str, backend: str, dry_run_only: bool) -> dict[str, Any]:
        return {
            "schema_version": "soridormi.body_activity.v1",
            "plan_id": self.plan_id,
            "coordination_id": self.coordination_id,
            "mode": mode,
            "backend": backend,
            "status": self.status,
            "terminal": self.terminal,
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
            "failure_reason": self.failure_reason,
            "estimated_duration_s": self.estimated_duration_s,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run_only": dry_run_only,
            "members": [member.to_dict() for member in self.members],
            "member_results": dict(self.member_results),
            "resource_claims": sorted(
                {
                    resource
                    for member in self.members
                    for resource in member.write_resources
                }
            ),
            "safety_authority": "soridormi",
            "speech_owner": "chromie",
            "one_final_motor_command_authority": True,
        }


def body_activity_capabilities_payload(*, mode: str, backend: str) -> dict[str, Any]:
    return {
        "schema_version": "soridormi.body_activity_capabilities.v1",
        "mode": mode,
        "backend": backend,
        "speech_owner": "chromie",
        "speech_is_external_peer_lane": True,
        "ability_classes": [
            {
                "ability_class": ABILITY_CLASS_SUBTLE_EXPRESSION,
                "description": (
                    "Visual expressions or bounded body-command overlays that do not "
                    "own the primary locomotion objective."
                ),
            },
            {
                "ability_class": ABILITY_CLASS_LOCOMOTION,
                "description": (
                    "The single primary locomotion or whole-body controller for an activity."
                ),
            },
        ],
        "control_couplings": [
            CONTROL_COUPLING_INDEPENDENT,
            CONTROL_COUPLING_OVERLAY,
            CONTROL_COUPLING_PRIMARY,
            CONTROL_COUPLING_STANDALONE,
        ],
        "concurrency_rules": {
            "max_primary_locomotion_members": 1,
            "many_compatible_subtle_expressions": True,
            "one_writer_per_resource": True,
            "head_overlays_are_composed_into_the_final_motor_command": True,
            "independent_visual_outputs_do_not_write_motor_commands": True,
            "standalone_body_motion_conflicts_with_primary_locomotion": True,
            "emergency_stop_preempts_every_body_member": True,
        },
        "coordination": {
            "coordination_id_supported": True,
            "external_speaking_lane_started_by": "chromie_runtime_coordinator",
            "soridormi_never_authors_speech_meaning": True,
        },
    }


def _concurrency_contract(skill: Mapping[str, Any]) -> dict[str, Any]:
    value = skill.get("concurrency")
    if not isinstance(value, dict):
        raise ValueError(
            f"skill {skill.get('id')!r} has no declared concurrency contract"
        )
    ability_class = str(value.get("ability_class") or "")
    control_coupling = str(value.get("control_coupling") or "")
    write_resources = value.get("write_resources")
    if ability_class not in BODY_ABILITY_CLASSES:
        raise ValueError(
            f"skill {skill.get('id')!r} has unsupported ability_class {ability_class!r}"
        )
    if control_coupling not in CONTROL_COUPLINGS:
        raise ValueError(
            f"skill {skill.get('id')!r} has unsupported control_coupling "
            f"{control_coupling!r}"
        )
    if not isinstance(write_resources, list) or not all(
        isinstance(item, str) and item for item in write_resources
    ):
        raise ValueError(
            f"skill {skill.get('id')!r} must declare concurrency.write_resources"
        )
    return value


def _validate_overlay_envelope(member: BodyActivityMemberPlan) -> None:
    envelope = dict(member.concurrency_envelope)
    parameters = dict(member.plan.parameters or {})
    yaw_limit = envelope.get("max_abs_head_yaw_rad_during_locomotion")
    pitch_limit = envelope.get("max_abs_head_pitch_rad_during_locomotion")
    yaw = parameters.get("target_yaw_rad", parameters.get("head_yaw_rad"))
    pitch = parameters.get("target_pitch_rad", parameters.get("head_pitch_rad"))
    if yaw_limit is not None and yaw is not None and abs(float(yaw)) > float(yaw_limit):
        raise ValueError(
            f"activity member {member.member_id!r} head yaw {float(yaw):.3f} exceeds "
            f"the locomotion overlay envelope {float(yaw_limit):.3f}"
        )
    if pitch_limit is not None and pitch is not None and abs(float(pitch)) > float(pitch_limit):
        raise ValueError(
            f"activity member {member.member_id!r} head pitch {float(pitch):.3f} exceeds "
            f"the locomotion overlay envelope {float(pitch_limit):.3f}"
        )


def create_body_activity_plan(
    registry: SkillExecutionRegistry,
    args: Mapping[str, Any],
) -> BodyActivityPlanRecord:
    allowed_top_level = {
        "coordination_id",
        "members",
        "profile",
        "chromie_intent",
    }
    unknown = set(args) - allowed_top_level
    if unknown:
        raise ValueError(
            "unsupported body activity field(s): " + ", ".join(sorted(unknown))
        )
    raw_members = args.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("body activity members must be a non-empty list")
    if len(raw_members) > MAX_ACTIVITY_MEMBERS:
        raise ValueError(
            f"body activity may contain at most {MAX_ACTIVITY_MEMBERS} members"
        )
    profile = args.get("profile")
    coordination_id = str(args.get("coordination_id") or "").strip() or None
    members: list[BodyActivityMemberPlan] = []
    seen_member_ids: set[str] = set()
    resource_owner: dict[str, str] = {}
    primary_count = 0

    for index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, dict):
            raise ValueError(f"body activity member {index} must be an object")
        unknown_member = set(raw_member) - {
            "member_id",
            "skill_id",
            "parameters",
            "optional",
        }
        if unknown_member:
            raise ValueError(
                f"body activity member {index} has unsupported field(s): "
                + ", ".join(sorted(unknown_member))
            )
        skill_id = str(raw_member.get("skill_id") or "").strip()
        if not skill_id:
            raise ValueError(f"body activity member {index} requires skill_id")
        member_id = str(raw_member.get("member_id") or skill_id).strip()
        if not member_id:
            raise ValueError(f"body activity member {index} requires member_id")
        if member_id in seen_member_ids:
            raise ValueError(f"duplicate body activity member_id: {member_id}")
        seen_member_ids.add(member_id)
        parameters = raw_member.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError(
                f"body activity member {member_id!r} parameters must be an object"
            )
        plan = registry.create_plan(skill_id, parameters, profile=profile)
        skill = registry.skills[skill_id]
        contract = _concurrency_contract(skill)
        ability_class = str(contract["ability_class"])
        control_coupling = str(contract["control_coupling"])
        write_resources = tuple(str(item) for item in contract["write_resources"])
        for resource in write_resources:
            previous = resource_owner.get(resource)
            if previous is not None:
                raise ValueError(
                    f"body activity resource {resource!r} is requested by both "
                    f"{previous!r} and {member_id!r}"
                )
            resource_owner[resource] = member_id
        if control_coupling == CONTROL_COUPLING_PRIMARY:
            primary_count += 1
        member = BodyActivityMemberPlan(
            member_id=member_id,
            skill_id=skill_id,
            plan=plan,
            optional=bool(raw_member.get("optional", False)),
            ability_class=ability_class,
            control_coupling=control_coupling,
            write_resources=write_resources,
            concurrency_envelope=dict(contract.get("locomotion_envelope") or {}),
        )
        members.append(member)

    if primary_count > 1:
        raise ValueError("body activity may contain at most one primary locomotion member")

    has_primary = primary_count == 1
    for member in members:
        if has_primary and member.control_coupling == CONTROL_COUPLING_STANDALONE:
            raise ValueError(
                f"skill {member.skill_id!r} is standalone body motion and cannot "
                "run with primary locomotion"
            )
        if has_primary and member.control_coupling == CONTROL_COUPLING_OVERLAY:
            _validate_overlay_envelope(member)

    estimated_duration_s = max(member.plan.total_duration_s for member in members)
    return BodyActivityPlanRecord(
        plan_id=f"soridormi-activity-plan-{uuid.uuid4().hex[:12]}",
        coordination_id=coordination_id,
        members=tuple(members),
        created_at=time.time(),
        estimated_duration_s=estimated_duration_s,
    )
