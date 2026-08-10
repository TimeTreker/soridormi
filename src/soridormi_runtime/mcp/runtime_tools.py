from __future__ import annotations

import asyncio
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from soridormi_api import MotorCommand, RobotState, VisualExpressionCommand
from soridormi_runtime.controller import HoldPositionController
from soridormi_runtime.main import make_controller, make_robot
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scripted_head_skill import (
    DEFAULT_MAX_HEAD_VELOCITY_RADPS,
    HEAD_JOINT_NAMES,
    DEFAULT_TRANSITION_FRACTION,
    SUPPORTED_SCRIPTED_SKILLS,
    command_positions_by_name,
    effective_duration_for_trajectory,
    joint_positions_by_name,
    keyframe_steps_for_durations,
    motor_command_from_targets,
    plan_head_pose_trajectory,
    resolve_keyframe_targets_for_execution,
    scaled_keyframe_durations,
    validate_scripted_head_plan,
)
from soridormi_runtime.skill_execution import (
    SkillExecutionRegistry,
    simulated_resource_outcome,
)
from soridormi_runtime.skill_manifest import (
    DEFAULT_SKILL_MANIFEST,
    parameters_schema_for_skill,
)
from soridormi_runtime.sync_preroll import preroll_sync_simulator
from soridormi_runtime.visual_expression_skill import (
    SUPPORTED_VISUAL_EXPRESSION_SKILLS,
    validate_visual_expression_plan,
)

from .body_activity import (
    BodyActivityMemberPlan,
    BodyActivityPlanRecord,
    CONTROL_COUPLING_INDEPENDENT,
    body_activity_capabilities_payload,
    compile_body_activity,
    skill_concurrency_projection,
)
from .local_tools import (
    MotionPlan,
    NamedSkillPlan,
    SoridormiLocalToolService,
    validate_chromie_intent,
)
from .source_identity import current_source_revision
from .task_tools import EmbodiedTaskStore, task_capabilities_payload


class RuntimeRobot(Protocol):
    def read_state(self) -> RobotState: ...

    def send_motor_command(self, command: MotorCommand) -> None: ...


class RuntimeController(Protocol):
    command: PolicyCommand

    def compute(self, state: RobotState) -> MotorCommand: ...


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class SoridormiRuntimeToolService:
    """MCP tool service backed by Soridormi's real runtime control interfaces."""

    robot: RuntimeRobot
    controller: RuntimeController
    mode: str = "sim"
    backend: str = "runtime"
    source_revision: str | None = field(default_factory=current_source_revision)
    control_hz: float = 50.0
    plans: dict[str, MotionPlan] = field(default_factory=dict)
    skill_plans: dict[str, NamedSkillPlan] = field(default_factory=dict)
    activity_plans: dict[str, BodyActivityPlanRecord] = field(default_factory=dict)
    skill_registry: SkillExecutionRegistry = field(
        default_factory=lambda: SkillExecutionRegistry.from_manifest_path(
            DEFAULT_SKILL_MANIFEST
        )
    )
    task_store: EmbodiedTaskStore = field(default_factory=EmbodiedTaskStore)
    emergency_stop: bool = False
    active_task: dict[str, Any] | None = None
    active_lanes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _motion_stop_requested: bool = field(default=False, init=False, repr=False)
    _locomotion_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _head_overlay_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _independent_output_locks: dict[str, asyncio.Lock] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _robot_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _last_state: RobotState | None = field(default=None, init=False, repr=False)
    _simulated_carried_resource: str | None = field(default=None, init=False, repr=False)
    _robot_executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        mode: str,
        robot_factory: Callable[[], RuntimeRobot] = make_robot,
        controller_factory: Callable[[], RuntimeController] = make_controller,
    ) -> SoridormiRuntimeToolService:
        if mode != "sim":
            raise ValueError(
                "the runtime MCP adapter currently supports sim mode only; "
                "HardwareRobot is not implemented"
            )
        controller = controller_factory()
        if not hasattr(controller, "command"):
            raise ValueError(
                "the runtime MCP adapter requires SORIDORMI_RUNTIME_MODE=onnx_policy"
            )

        # RobotApiClient owns a ZeroMQ REQ socket. ZeroMQ sockets are strictly
        # thread-affine, so both construction and every subsequent robot call
        # must happen on the same worker thread. Repeated asyncio.to_thread()
        # calls are unsafe because the default pool may select a different
        # thread for each operation.
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="soridormi-robot",
        )
        try:
            robot = executor.submit(robot_factory).result()
            last_state: RobotState | None = None

            if _env_bool("SORIDORMI_RESET_AT_START", default=False):
                resetter = getattr(robot, "reset", None)
                if callable(resetter):
                    executor.submit(resetter).result()

            if _env_bool("SORIDORMI_SIM_SYNC_STEP", default=False):
                last_state = executor.submit(robot.read_state).result()
                preroll_steps = max(
                    0,
                    _env_int("SORIDORMI_SIM_PREROLL_STEPS", default=0),
                )
                if preroll_steps:
                    last_state = executor.submit(
                        preroll_sync_simulator,
                        robot,
                        last_state,
                        preroll_steps,
                    ).result()

            service = cls(
                robot=robot,
                controller=controller,
                mode=mode,
                backend="runtime",
                control_hz=float(os.environ.get("CONTROL_HZ", "50")),
            )
            service._robot_executor = executor
            service._last_state = last_state
            return service
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    async def call_tool(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args = args or {}
        if tool_name == "soridormi.robot.get_status":
            return await self.get_status()
        if tool_name == "soridormi.robot.get_mode":
            return {"mode": self.mode}
        if tool_name == "soridormi.robot.get_battery":
            state = await self._read_state()
            battery = state.battery
            return {
                "percent": battery.percent if battery is not None else None,
                "critical": bool(
                    battery is not None
                    and battery.percent is not None
                    and battery.percent <= 10.0
                ),
            }
        if tool_name == "soridormi.motion.create_plan":
            return self.create_motion_plan(args)
        if tool_name == "soridormi.motion.execute_plan":
            return await self.execute_motion_plan(str(args.get("plan_id", "")))
        if tool_name == "soridormi.motion.stop":
            return await self.stop_motion(cancelled=False)
        if tool_name == "soridormi.motion.cancel":
            return await self.stop_motion(cancelled=True)
        if tool_name == "soridormi.skill.list":
            return self.list_runtime_skills()
        if tool_name == "soridormi.skill.create_plan":
            return self.create_runtime_skill_plan(args)
        if tool_name == "soridormi.skill.execute_plan":
            return await self.execute_runtime_skill_plan(
                str(args.get("plan_id", ""))
            )
        if tool_name == "soridormi.activity.get_capabilities":
            return body_activity_capabilities_payload(
                mode=self.mode,
                backend=self.backend,
            )
        if tool_name in {"soridormi.activity.compile", "soridormi.activity.create_plan"}:
            return self.create_runtime_activity_plan(args)
        if tool_name in {"soridormi.activity.execute", "soridormi.activity.execute_plan"}:
            return await self.execute_runtime_activity_plan(
                str(args.get("compiled_activity_id") or args.get("plan_id") or "")
            )
        if tool_name == "soridormi.activity.status":
            return self.runtime_activity_status(str(args.get("compiled_activity_id") or args.get("plan_id") or ""))
        if tool_name == "soridormi.activity.cancel":
            return await self.cancel_runtime_activity(
                str(args.get("compiled_activity_id") or args.get("plan_id") or ""),
                reason=str(args.get("reason") or "cancelled by caller"),
            )
        body_safe_idle = self._body_safe_idle()
        if tool_name == "soridormi.task.get_capabilities":
            return task_capabilities_payload(
                mode=self.mode,
                backend=self.backend,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
                skill_registry=self.skill_registry,
            )
        if tool_name == "soridormi.task.preview":
            return self.task_store.preview_task(
                args,
                mode=self.mode,
                backend=self.backend,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
                skill_registry=self.skill_registry,
            )
        if tool_name == "soridormi.task.submit":
            return self.task_store.submit_task(
                args,
                mode=self.mode,
                backend=self.backend,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
                skill_registry=self.skill_registry,
            )
        if tool_name == "soridormi.task.status":
            return self.task_store.task_status(
                args,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
            )
        if tool_name == "soridormi.task.events":
            return self.task_store.task_events(
                args,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
            )
        if tool_name == "soridormi.task.cancel":
            return self.task_store.cancel_task(
                args,
                emergency_stop=self.emergency_stop,
                safe_idle=body_safe_idle,
            )
        if tool_name == "soridormi.safety.monitor_motion":
            return {
                "ok": not self.emergency_stop,
                "active": bool(self.active_lanes),
                "active_lanes": sorted(self.active_lanes),
                "event": "emergency_stop" if self.emergency_stop else None,
                "safe_idle": self._body_safe_idle(),
            }
        if tool_name == "soridormi.safety.emergency_stop":
            return await self.emergency_stop_motion(
                reason=str(args.get("reason", "unspecified"))
            )
        raise KeyError(f"unknown Soridormi runtime tool: {tool_name}")


    def _runtime_skill_ids(self) -> tuple[str, ...]:
        supported: list[str] = []
        for skill_id in self.skill_registry.executable_skill_ids():
            execution = self.skill_registry.skills[skill_id].get("execution")
            if execution in {"policy_velocity", "skill_wrapper"}:
                supported.append(skill_id)
            elif execution == "scripted_keyframe" and skill_id in SUPPORTED_SCRIPTED_SKILLS:
                supported.append(skill_id)
            elif (
                execution == "visual_expression"
                and skill_id in SUPPORTED_VISUAL_EXPRESSION_SKILLS
            ):
                supported.append(skill_id)
            elif (
                execution == "composite"
                and (self.skill_registry.skills[skill_id].get("metadata") or {}).get(
                    "simulation_mock"
                ) == "resource_acquisition_delivery"
                and self.mode == "sim"
            ):
                supported.append(skill_id)
        return tuple(supported)

    @staticmethod
    def _parameters_schema(skill: dict[str, Any]) -> dict[str, Any]:
        return parameters_schema_for_skill(skill)

    def list_runtime_skills(self) -> dict[str, Any]:
        skills: list[dict[str, Any]] = []
        for skill_id in self._runtime_skill_ids():
            skill = self.skill_registry.skills[skill_id]
            execution = str(skill.get("execution") or "")
            if execution == "visual_expression":
                default_effects = ["visual_expression"]
                default_safety_class = "low_risk_action"
            else:
                default_effects = ["physical_motion"]
                default_safety_class = "physical_motion"
            metadata = skill.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            skills.append(
                {
                    "skill_id": skill_id,
                    "version": "0.1.0",
                    "available": True,
                    "description": str(skill.get("description") or ""),
                    "parameters_schema": self._parameters_schema(skill),
                    "interruptible": bool(
                        (skill.get("safety") or {}).get("interruptible", True)
                    ),
                    "effects": list(skill.get("effects") or default_effects),
                    "safety_class": str(
                        skill.get("safety_class") or default_safety_class
                    ),
                    "requires_confirmation": bool(
                        skill.get("requires_confirmation", self.mode != "sim")
                    ),
                    "when_to_use": str(skill.get("description") or ""),
                    "execution": execution,
                    "notes": str(skill.get("notes") or ""),
                    "semantic_speed_presets_mps": dict(
                        skill.get("semantic_speed_presets_mps") or {}
                    ),
                    "metadata": metadata,
                    **skill_concurrency_projection(skill),
                }
            )
        return {"mode": self.mode, "skills": skills}

    def _set_active_lane(self, lane: str, metadata: dict[str, Any]) -> None:
        self.active_lanes[lane] = dict(metadata)
        self._refresh_active_task()

    def _clear_active_lane(self, lane: str) -> None:
        self.active_lanes.pop(lane, None)
        self._refresh_active_task()

    def _refresh_active_task(self) -> None:
        if not self.active_lanes:
            self.active_task = None
            return
        if len(self.active_lanes) == 1:
            lane, metadata = next(iter(self.active_lanes.items()))
            self.active_task = {"lane": lane, **dict(metadata)}
            return
        coordination_ids = {
            str(metadata.get("coordination_id"))
            for metadata in self.active_lanes.values()
            if metadata.get("coordination_id")
        }
        self.active_task = {
            "kind": "concurrent_body_activity",
            "coordination_id": next(iter(coordination_ids)) if len(coordination_ids) == 1 else None,
            "lanes": {
                lane: dict(metadata)
                for lane, metadata in sorted(self.active_lanes.items())
            },
        }

    def _body_safe_idle(self) -> bool:
        physical_lanes = {
            lane
            for lane in self.active_lanes
            if lane == "locomotion" or lane == "head_overlay" or lane.startswith("standalone_body")
        }
        return not physical_lanes and not self.emergency_stop

    def create_runtime_activity_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        validate_chromie_intent(args.get("chromie_intent"))
        record = compile_body_activity(self.skill_registry, args)
        self.activity_plans[record.plan_id] = record
        return record.to_dict(
            mode=self.mode,
            backend=self.backend,
            dry_run_only=False,
        )

    def runtime_activity_status(self, plan_id: str) -> dict[str, Any]:
        return self._activity_record(plan_id).to_dict(
            mode=self.mode,
            backend=self.backend,
            dry_run_only=False,
        )

    async def cancel_runtime_activity(self, plan_id: str, *, reason: str) -> dict[str, Any]:
        record = self._activity_record(plan_id)
        if record.terminal:
            payload = record.to_dict(
                mode=self.mode,
                backend=self.backend,
                dry_run_only=False,
            )
            payload["cancelled"] = False
            return payload
        record.cancel_requested = True
        record.cancel_reason = reason
        if record.primary_member is not None or record.head_overlay_member is not None or record.standalone_members:
            self._motion_stop_requested = True
            await self._apply_safe_hold()
        payload = record.to_dict(
            mode=self.mode,
            backend=self.backend,
            dry_run_only=False,
        )
        payload["cancelled"] = True
        return payload

    def _activity_record(self, plan_id: str) -> BodyActivityPlanRecord:
        if not plan_id:
            raise ValueError("plan_id is required")
        record = self.activity_plans.get(plan_id)
        if record is None:
            raise KeyError(f"body activity plan not found: {plan_id}")
        return record

    @staticmethod
    def _motion_commands_from_skill_plan(plan: Any) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        for segment in plan.commands:
            remaining = float(segment.duration_s)
            while remaining > 0.0:
                duration_s = min(5.0, remaining)
                commands.append(
                    {
                        "vx": float(segment.vx_mps),
                        "vy": float(segment.vy_mps),
                        "yaw": float(segment.yaw_radps),
                        "duration_s": duration_s,
                        "label": segment.label or plan.skill_id,
                    }
                )
                remaining -= duration_s
        if not commands:
            raise ValueError(
                f"runtime skill {plan.skill_id!r} produced no velocity commands"
            )
        return commands

    def create_runtime_skill_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        validate_chromie_intent(args.get("chromie_intent"))
        skill_id = str(args.get("skill_id", "")).strip()
        if not skill_id:
            raise ValueError("skill_id is required")
        if skill_id not in self._runtime_skill_ids():
            raise ValueError(
                f"skill {skill_id!r} is not supported by the runtime adapter"
            )

        plan = self.skill_registry.create_plan(
            skill_id,
            args.get("parameters") or {},
            profile=args.get("profile"),
        )
        if plan.execution in {"policy_velocity", "skill_wrapper"}:
            motion_result = self.create_motion_plan(
                {"commands": self._motion_commands_from_skill_plan(plan)}
            )
            plan_id = str(motion_result["plan_id"])
        elif plan.execution == "scripted_keyframe":
            validate_scripted_head_plan(plan)
            plan_id = f"soridormi-skill-plan-{uuid.uuid4().hex[:12]}"
        elif plan.execution == "visual_expression":
            validate_visual_expression_plan(plan.skill_id, plan.execution)
            plan_id = f"soridormi-skill-plan-{uuid.uuid4().hex[:12]}"
        elif (
            plan.execution == "composite"
            and (self.skill_registry.skills[skill_id].get("metadata") or {}).get(
                "simulation_mock"
            ) == "resource_acquisition_delivery"
            and self.mode == "sim"
        ):
            motion_result = self.create_motion_plan(
                {"commands": self._motion_commands_from_skill_plan(plan)}
            )
            plan_id = str(motion_result["plan_id"])
        else:  # pragma: no cover - guarded by _runtime_skill_ids
            raise ValueError(
                f"skill {skill_id!r} execution {plan.execution!r} is unsupported"
            )

        self.skill_plans[plan_id] = NamedSkillPlan(
            plan_id=plan_id,
            plan=plan,
            created_at=time.time(),
        )
        return {
            "plan_id": plan_id,
            "skill_id": skill_id,
            "mode": self.mode,
            "summary": plan.summary.replace("Dry-run", "Runtime"),
            "estimated_duration_s": plan.total_duration_s,
            "requires_confirmation": bool(
                self.skill_registry.skills[skill_id].get(
                    "requires_confirmation", self.mode != "sim"
                )
            ),
            "interruptible": bool((plan.safety or {}).get("interruptible", True)),
            "no_motion": plan.execution == "visual_expression",
        }

    async def execute_runtime_skill_plan(self, plan_id: str) -> dict[str, Any]:
        if not plan_id:
            raise ValueError("plan_id is required")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        if stored.plan.execution in {"policy_velocity", "skill_wrapper"}:
            result = await self.execute_motion_plan(plan_id)
        elif stored.plan.execution == "scripted_keyframe":
            result = await self.execute_scripted_head_skill(plan_id)
        elif stored.plan.execution == "visual_expression":
            result = await self.execute_visual_expression_skill(plan_id)
        elif (
            stored.plan.execution == "composite"
            and (
                self.skill_registry.skills[stored.plan.skill_id].get("metadata") or {}
            ).get("simulation_mock") == "resource_acquisition_delivery"
            and self.mode == "sim"
        ):
            if self.emergency_stop:
                raise RuntimeError(
                    "cannot execute resource acquisition while emergency_stop is active"
                )
            resource = (stored.plan.parameters or {}).get("resource")
            description = (
                " ".join(str(resource.get("description") or "").strip().split())
                if isinstance(resource, dict)
                else ""
            )
            if stored.plan.skill_id == "acquire_resource" and self._simulated_carried_resource is not None:
                raise RuntimeError("simulation mock is already carrying a resource")
            if (
                stored.plan.skill_id == "deliver_resource"
                and self._simulated_carried_resource != description
            ):
                raise RuntimeError(
                    "deliver_resource requires the matching simulated acquired resource"
                )
            motion_result = await self.execute_motion_plan(plan_id)
            if motion_result.get("completed") is not True:
                result = {**motion_result, "no_motion": False}
            else:
                if stored.plan.skill_id == "acquire_resource":
                    self._simulated_carried_resource = description
                    summary = "Soridormi simulation mock acquired the physical resource."
                elif stored.plan.skill_id == "deliver_resource":
                    self._simulated_carried_resource = None
                    summary = "Soridormi simulation mock delivered the carried resource."
                else:
                    summary = (
                        "Soridormi simulation mock completed its scripted physical "
                        "resource acquisition and handover sequence."
                    )
                result = {
                    **motion_result,
                    "completed": True,
                    "summary": summary,
                    "no_motion": False,
                    "resource_outcome": simulated_resource_outcome(stored.plan),
                }
        else:  # pragma: no cover - plans are validated at creation
            raise ValueError(
                f"skill {stored.plan.skill_id!r} execution "
                f"{stored.plan.execution!r} is unsupported"
            )
        return {
            **result,
            "skill_id": stored.plan.skill_id,
            "mode": self.mode,
            "no_motion": bool(
                result.get(
                    "no_motion",
                    stored.plan.execution == "visual_expression",
                )
            ),
            "recommendation_only": False,
        }

    async def execute_runtime_activity_plan(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute body activity while emergency_stop is active")
        record = self._activity_record(plan_id)
        if record.status != "planned":
            raise RuntimeError(
                f"body activity {plan_id} cannot execute from status {record.status}"
            )
        record.status = "running"
        record.started_at = time.time()
        record.cancel_requested = False
        record.cancel_reason = None
        self._motion_stop_requested = False

        tasks: list[asyncio.Task[dict[str, Any]]] = []
        has_physical = bool(
            record.primary_member
            or record.head_overlay_member
            or record.standalone_members
        )
        if has_physical:
            tasks.append(asyncio.create_task(self._execute_physical_activity(record)))
        for member in record.independent_members:
            tasks.append(
                asyncio.create_task(
                    self._execute_activity_member(record, member)
                )
            )
        if not tasks:
            record.status = "failed"
            record.failure_reason = "body activity contains no executable members"
            record.completed_at = time.time()
        else:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            errors: list[BaseException] = []
            for task in done:
                if task.cancelled():
                    continue
                error = task.exception()
                if error is not None:
                    errors.append(error)
            if errors:
                record.cancel_requested = True
                self._motion_stop_requested = True
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                record.status = "failed"
                record.failure_reason = "; ".join(str(error) for error in errors)
                if has_physical:
                    await self._apply_safe_hold()
            elif record.cancel_requested or self.emergency_stop or self._motion_stop_requested:
                record.status = "cancelled"
            else:
                optional_failures = [
                    result
                    for result in record.member_results.values()
                    if result.get("status") == "failed" and result.get("optional") is True
                ]
                record.status = (
                    "completed_with_degradation"
                    if optional_failures
                    else "completed"
                )
            record.completed_at = time.time()

        payload = record.to_dict(
            mode=self.mode,
            backend=self.backend,
            dry_run_only=False,
        )
        payload.update(
            {
                "completed": record.status in {
                    "completed",
                    "completed_with_degradation",
                },
                "degraded": record.status == "completed_with_degradation",
                "cancelled": record.status == "cancelled",
                "motor_commands_sent": has_physical and record.started_at is not None,
                "summary": (
                    f"Soridormi body activity {record.status}; "
                    f"{len(record.members)} member(s) reconciled."
                ),
            }
        )
        return payload

    async def _execute_activity_member(
        self,
        record: BodyActivityPlanRecord,
        member: BodyActivityMemberPlan,
    ) -> dict[str, Any]:
        try:
            if member.control_coupling != CONTROL_COUPLING_INDEPENDENT:
                raise RuntimeError(
                    f"activity member {member.member_id} is not an independent output"
                )
            result = await self._execute_visual_activity_member(record, member)
            record.member_results[member.member_id] = result
            return result
        except asyncio.CancelledError:
            record.member_results[member.member_id] = {
                "member_id": member.member_id,
                "skill_id": member.skill_id,
                "status": "cancelled",
                "completed": False,
                "optional": member.optional,
            }
            raise
        except Exception as exc:
            result = {
                "member_id": member.member_id,
                "skill_id": member.skill_id,
                "status": "failed",
                "completed": False,
                "optional": member.optional,
                "reason": str(exc),
            }
            record.member_results[member.member_id] = result
            if member.optional:
                return result
            raise

    async def _execute_visual_activity_member(
        self,
        record: BodyActivityPlanRecord,
        member: BodyActivityMemberPlan,
    ) -> dict[str, Any]:
        plan = member.plan
        validate_visual_expression_plan(plan.skill_id, plan.execution)
        applier = getattr(self.robot, "set_visual_expression", None)
        if not callable(applier):
            raise RuntimeError("robot backend does not support visual expressions")
        resource = member.write_resources[0]
        lane = f"visual:{resource}"
        async with self._independent_output_lock(resource):
            self._set_active_lane(
                lane,
                {
                    "plan_id": record.plan_id,
                    "coordination_id": record.coordination_id,
                    "member_id": member.member_id,
                    "skill_id": member.skill_id,
                    "ability_class": member.ability_class,
                },
            )
            try:
                for index, expression in enumerate(plan.visual_expressions):
                    if self.emergency_stop or record.cancel_requested:
                        await self._apply_visual_expression_command(
                            applier,
                            VisualExpressionCommand(
                                expression="eyes_open",
                                intensity=1.0,
                            ),
                        )
                        return {
                            "member_id": member.member_id,
                            "skill_id": member.skill_id,
                            "status": "cancelled",
                            "completed": False,
                            "optional": member.optional,
                            "command_index": index,
                        }
                    self.active_lanes[lane]["command_index"] = index
                    self._refresh_active_task()
                    await self._apply_visual_expression_command(
                        applier,
                        VisualExpressionCommand(
                            expression=expression.expression,
                            intensity=expression.intensity,
                        ),
                    )
                    await asyncio.sleep(max(0.0, float(expression.duration_s)))
                await self._apply_visual_expression_command(
                    applier,
                    VisualExpressionCommand(
                        expression="eyes_open",
                        intensity=1.0,
                    ),
                )
                return {
                    "member_id": member.member_id,
                    "skill_id": member.skill_id,
                    "status": "completed",
                    "completed": True,
                    "optional": member.optional,
                    "visual_expression_steps": len(plan.visual_expressions),
                }
            finally:
                self._clear_active_lane(lane)

    async def _execute_physical_activity(
        self,
        record: BodyActivityPlanRecord,
    ) -> dict[str, Any]:
        primary = record.primary_member
        head_member = record.head_overlay_member
        standalone = record.standalone_members
        if standalone:
            if len(standalone) != 1:
                raise RuntimeError("only one standalone body member is supported")
            head_member = standalone[0]

        locks: list[asyncio.Lock] = []
        if primary is not None or standalone or (head_member is not None and primary is None):
            locks.append(self._locomotion_lock)
        if head_member is not None:
            locks.append(self._head_overlay_lock)
        for lock in locks:
            await lock.acquire()
        try:
            state = await self._read_state()
            head_trajectory: list[dict[str, float]] = []
            head_duration_s = 0.0
            if head_member is not None:
                head_trajectory, head_duration_s = self._head_trajectory_for_activity(
                    head_member,
                    state,
                )
                self._set_active_lane(
                    "head_overlay" if primary is not None else "standalone_body:head",
                    {
                        "plan_id": record.plan_id,
                        "coordination_id": record.coordination_id,
                        "member_id": head_member.member_id,
                        "skill_id": head_member.skill_id,
                        "ability_class": head_member.ability_class,
                    },
                )
            motion_duration_s = 0.0
            if primary is not None:
                motion_duration_s = sum(
                    float(segment.duration_s) for segment in primary.plan.commands
                )
                self._set_active_lane(
                    "locomotion",
                    {
                        "plan_id": record.plan_id,
                        "coordination_id": record.coordination_id,
                        "member_id": primary.member_id,
                        "skill_id": primary.skill_id,
                        "ability_class": primary.ability_class,
                        "command_index": 0,
                    },
                )
            total_duration_s = max(motion_duration_s, head_duration_s)
            if total_duration_s <= 0.0:
                raise RuntimeError("physical activity has no positive duration")
            period_s = 1.0 / self.control_hz
            total_ticks = max(1, int(math.ceil(total_duration_s * self.control_hz)))
            for tick in range(total_ticks):
                if self.emergency_stop or record.cancel_requested or self._motion_stop_requested:
                    if primary is not None:
                        record.member_results[primary.member_id] = {
                            "member_id": primary.member_id,
                            "skill_id": primary.skill_id,
                            "status": "cancelled",
                            "completed": False,
                            "optional": primary.optional,
                        }
                    if head_member is not None:
                        record.member_results[head_member.member_id] = {
                            "member_id": head_member.member_id,
                            "skill_id": head_member.skill_id,
                            "status": "cancelled",
                            "completed": False,
                            "optional": head_member.optional,
                        }
                    return {"completed": False, "cancelled": True}
                elapsed_s = tick * period_s
                self.controller.command = self._policy_command_for_elapsed(
                    primary,
                    elapsed_s,
                )
                if primary is not None and "locomotion" in self.active_lanes:
                    self.active_lanes["locomotion"]["command_index"] = tick
                head_target = None
                if head_trajectory:
                    head_target = head_trajectory[min(tick, len(head_trajectory) - 1)]
                    lane = "head_overlay" if primary is not None else "standalone_body:head"
                    if lane in self.active_lanes:
                        self.active_lanes[lane]["command_index"] = min(
                            tick,
                            len(head_trajectory) - 1,
                        )
                self._refresh_active_task()
                started_at = asyncio.get_running_loop().time()
                await self._step_composed_controller(head_target)
                await asyncio.sleep(
                    max(
                        0.0,
                        period_s
                        - (asyncio.get_running_loop().time() - started_at),
                    )
                )
            if primary is not None:
                record.member_results[primary.member_id] = {
                    "member_id": primary.member_id,
                    "skill_id": primary.skill_id,
                    "status": "completed",
                    "completed": True,
                    "optional": primary.optional,
                    "estimated_duration_s": motion_duration_s,
                }
            if head_member is not None:
                record.member_results[head_member.member_id] = {
                    "member_id": head_member.member_id,
                    "skill_id": head_member.skill_id,
                    "status": "completed",
                    "completed": True,
                    "optional": head_member.optional,
                    "estimated_duration_s": head_duration_s,
                    "composed_into_final_motor_command": True,
                }
            return {"completed": True}
        except asyncio.CancelledError:
            await asyncio.shield(self._apply_safe_hold())
            raise
        finally:
            self.controller.command = PolicyCommand()
            self._clear_active_lane("locomotion")
            self._clear_active_lane("head_overlay")
            self._clear_active_lane("standalone_body:head")
            for lock in reversed(locks):
                lock.release()

    def _head_trajectory_for_activity(
        self,
        member: BodyActivityMemberPlan,
        state: RobotState,
    ) -> tuple[list[dict[str, float]], float]:
        plan = member.plan
        validate_scripted_head_plan(plan)
        initial_positions = joint_positions_by_name(state)
        initial_controls = command_positions_by_name(state)
        resolved_targets = resolve_keyframe_targets_for_execution(
            plan,
            initial_positions,
        )
        requested_duration_s = sum(
            float(keyframe.duration_s) for keyframe in plan.keyframes
        )
        envelope_velocity = member.concurrency_envelope.get("max_head_velocity_radps")
        max_velocity = (
            float(envelope_velocity)
            if envelope_velocity is not None
            else DEFAULT_MAX_HEAD_VELOCITY_RADPS
        )
        effective_duration_s = effective_duration_for_trajectory(
            requested_duration_s=requested_duration_s,
            targets=resolved_targets,
            max_head_velocity_radps=max_velocity,
            auto_stretch_duration=True,
            keyframe_durations=[
                float(keyframe.duration_s) for keyframe in plan.keyframes
            ],
        )
        keyframe_durations = scaled_keyframe_durations(
            plan,
            effective_duration_s,
        )
        keyframe_steps = keyframe_steps_for_durations(
            keyframe_durations,
            self.control_hz,
        )
        trajectory = plan_head_pose_trajectory(
            plan,
            resolved_targets,
            keyframe_steps,
            start_positions_by_name=initial_controls,
            control_hz=self.control_hz,
            transition_fraction=DEFAULT_TRANSITION_FRACTION,
            max_head_velocity_radps=max_velocity,
        )
        return trajectory, effective_duration_s

    @staticmethod
    def _policy_command_for_elapsed(
        primary: BodyActivityMemberPlan | None,
        elapsed_s: float,
    ) -> PolicyCommand:
        if primary is None:
            return PolicyCommand()
        cursor = 0.0
        for segment in primary.plan.commands:
            cursor += float(segment.duration_s)
            if elapsed_s < cursor:
                return PolicyCommand(
                    x_velocity=float(segment.vx_mps),
                    y_velocity=float(segment.vy_mps),
                    yaw_velocity=float(segment.yaw_radps),
                )
        return PolicyCommand()

    async def _step_composed_controller(
        self,
        head_target: dict[str, float] | None,
    ) -> None:
        async with self._robot_lock:
            state = self._last_state
            if state is None:
                state = await self._call_robot(self.robot.read_state)
            command = self.controller.compute(state)
            if head_target is not None:
                command = self._compose_head_overlay(command, head_target)
            stepper = getattr(self.robot, "step_motor_command", None)
            if callable(stepper):
                self._last_state = await self._call_robot(stepper, command)
            else:
                await self._call_robot(self.robot.send_motor_command, command)
                self._last_state = await self._call_robot(self.robot.read_state)

    @staticmethod
    def _compose_head_overlay(
        command: MotorCommand,
        head_target: dict[str, float],
    ) -> MotorCommand:
        index_by_name = {name: index for index, name in enumerate(command.names)}
        supported = [name for name in HEAD_JOINT_NAMES if name in index_by_name]
        requested = [name for name in HEAD_JOINT_NAMES if name in head_target]
        missing = [name for name in requested if name not in index_by_name]
        if missing:
            raise RuntimeError(
                "runtime motor command is missing head overlay joints: "
                + ", ".join(sorted(missing))
            )
        positions = list(command.positions)
        velocities = list(command.velocities)
        torques = list(command.torques)
        for name in requested:
            index = index_by_name.get(name)
            if index is None:
                continue
            positions[index] = float(head_target[name])
            if index < len(velocities):
                velocities[index] = 0.0
            if index < len(torques):
                torques[index] = 0.0
        return command.model_copy(
            update={
                "positions": positions,
                "velocities": velocities,
                "torques": torques,
            }
        )

    def _independent_output_lock(self, resource: str) -> asyncio.Lock:
        lock = self._independent_output_locks.get(resource)
        if lock is None:
            lock = asyncio.Lock()
            self._independent_output_locks[resource] = lock
        return lock

    async def _apply_visual_expression_command(
        self,
        applier: Callable[..., Any],
        command: VisualExpressionCommand,
    ) -> Any:
        async with self._robot_lock:
            return await self._call_robot(applier, command)

    async def execute_scripted_head_skill(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute skill while emergency_stop is active")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        plan = stored.plan
        validate_scripted_head_plan(plan)

        await self._locomotion_lock.acquire()
        await self._head_overlay_lock.acquire()
        lane = "standalone_body:head"
        try:
            if self.emergency_stop:
                raise RuntimeError("cannot execute skill while emergency_stop is active")
            self._motion_stop_requested = False
            self._set_active_lane(
                lane,
                {
                    "plan_id": plan_id,
                    "skill_id": plan.skill_id,
                    "execution": plan.execution,
                    "started_at": time.time(),
                    "command_index": 0,
                },
            )
            state = await self._read_state()
            initial_positions = joint_positions_by_name(state)
            initial_controls = command_positions_by_name(state)
            resolved_targets = resolve_keyframe_targets_for_execution(
                plan,
                initial_positions,
            )
            requested_duration_s = sum(
                float(keyframe.duration_s) for keyframe in plan.keyframes
            )
            effective_duration_s = effective_duration_for_trajectory(
                requested_duration_s=requested_duration_s,
                targets=resolved_targets,
                max_head_velocity_radps=DEFAULT_MAX_HEAD_VELOCITY_RADPS,
                auto_stretch_duration=True,
                keyframe_durations=[
                    float(keyframe.duration_s) for keyframe in plan.keyframes
                ],
            )
            keyframe_durations = scaled_keyframe_durations(
                plan,
                effective_duration_s,
            )
            keyframe_steps = keyframe_steps_for_durations(
                keyframe_durations,
                self.control_hz,
            )
            trajectory = plan_head_pose_trajectory(
                plan,
                resolved_targets,
                keyframe_steps,
                start_positions_by_name=initial_controls,
                control_hz=self.control_hz,
                transition_fraction=DEFAULT_TRANSITION_FRACTION,
                max_head_velocity_radps=DEFAULT_MAX_HEAD_VELOCITY_RADPS,
            )
            period_s = 1.0 / self.control_hz
            for index, target in enumerate(trajectory):
                if self.emergency_stop or self._motion_stop_requested:
                    return {
                        "completed": False,
                        "stopped": True,
                        "dry_run_only": False,
                        "summary": f"Soridormi runtime stopped skill {plan.skill_id}.",
                    }
                self.active_lanes[lane]["command_index"] = index
                self._refresh_active_task()
                started_at = asyncio.get_running_loop().time()
                state = await self._step_head_target(state, target)
                await asyncio.sleep(
                    max(
                        0.0,
                        period_s - (asyncio.get_running_loop().time() - started_at),
                    )
                )
            return {
                "completed": True,
                "dry_run_only": False,
                "summary": f"Soridormi runtime completed skill {plan.skill_id}.",
                "estimated_duration_s": effective_duration_s,
            }
        except asyncio.CancelledError:
            await asyncio.shield(self._apply_safe_hold())
            raise
        finally:
            self._clear_active_lane(lane)
            self._head_overlay_lock.release()
            self._locomotion_lock.release()

    async def execute_visual_expression_skill(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute skill while emergency_stop is active")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        plan = stored.plan
        validate_visual_expression_plan(plan.skill_id, plan.execution)
        applier = getattr(self.robot, "set_visual_expression", None)
        if not callable(applier):
            raise RuntimeError("robot backend does not support visual expressions")

        lane = "visual:visual.eyes"
        async with self._independent_output_lock("visual.eyes"):
            if self.emergency_stop:
                raise RuntimeError("cannot execute skill while emergency_stop is active")
            self._set_active_lane(
                lane,
                {
                    "plan_id": plan_id,
                    "skill_id": plan.skill_id,
                    "execution": plan.execution,
                    "started_at": time.time(),
                    "command_index": 0,
                },
            )
            try:
                for index, expression in enumerate(plan.visual_expressions):
                    if self.emergency_stop:
                        await self._apply_visual_expression_command(
                            applier,
                            VisualExpressionCommand(
                                expression="eyes_open",
                                intensity=1.0,
                            ),
                        )
                        return {
                            "completed": False,
                            "stopped": True,
                            "dry_run_only": False,
                            "summary": f"Soridormi runtime stopped skill {plan.skill_id}.",
                        }
                    self.active_lanes[lane]["command_index"] = index
                    self._refresh_active_task()
                    await self._apply_visual_expression_command(
                        applier,
                        VisualExpressionCommand(
                            expression=expression.expression,
                            intensity=expression.intensity,
                        ),
                    )
                    await asyncio.sleep(max(0.0, float(expression.duration_s)))
                await self._apply_visual_expression_command(
                    applier,
                    VisualExpressionCommand(expression="eyes_open", intensity=1.0),
                )
                return {
                    "completed": True,
                    "dry_run_only": False,
                    "summary": f"Soridormi runtime completed skill {plan.skill_id}.",
                    "estimated_duration_s": plan.total_duration_s,
                    "visual_expression_steps": len(plan.visual_expressions),
                }
            except asyncio.CancelledError:
                await asyncio.shield(
                    self._apply_visual_expression_command(
                        applier,
                        VisualExpressionCommand(expression="eyes_open", intensity=1.0),
                    )
                )
                raise
            finally:
                self._clear_active_lane(lane)

    async def _call_robot(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run one robot operation without violating ZeroMQ thread affinity."""

        if self._robot_executor is None:
            # Directly constructed services in unit tests may use thread-safe
            # fake robots and do not require a persistent worker.
            return await asyncio.to_thread(func, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._robot_executor,
            partial(func, *args),
        )

    async def _step_head_target(
        self,
        state: RobotState,
        target_positions_by_name: dict[str, float],
    ) -> RobotState:
        command = motor_command_from_targets(state, target_positions_by_name)
        async with self._robot_lock:
            stepper = getattr(self.robot, "step_motor_command", None)
            if callable(stepper):
                self._last_state = await self._call_robot(stepper, command)
            else:
                await self._call_robot(self.robot.send_motor_command, command)
                self._last_state = await self._call_robot(self.robot.read_state)
            return self._last_state

    def create_motion_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        planner = SoridormiLocalToolService(
            mode=self.mode,
            backend=self.backend,
            plans=self.plans,
            emergency_stop=self.emergency_stop,
        )
        result = planner.create_motion_plan(args)
        result["dry_run_only"] = False
        result["summary"] = result["summary"].replace("dry-run ", "runtime ")
        return result

    async def get_status(self) -> dict[str, Any]:
        if self.active_lanes and self._last_state is not None:
            state = self._last_state
        else:
            state = await self._read_state()
        status: dict[str, Any] = {
            "mode": self.mode,
            "backend": self.backend,
            "standing": True,
            "fallen": False,
            "emergency_stop": self.emergency_stop,
            "active_task": dict(self.active_task) if self.active_task is not None else None,
            "active_lanes": {
                lane: dict(metadata)
                for lane, metadata in sorted(self.active_lanes.items())
            },
            "activity_idle": not self.active_lanes,
            "safe_idle": self._body_safe_idle(),
            "robot_time": float(state.time),
        }
        if self.source_revision:
            status["source_revision"] = self.source_revision
        return status

    async def execute_motion_plan(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute motion while emergency_stop is active")
        if not plan_id:
            raise ValueError("plan_id is required")
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"plan not found: {plan_id}")

        lane = "locomotion"
        async with self._locomotion_lock:
            if self.emergency_stop:
                raise RuntimeError("cannot execute motion while emergency_stop is active")
            self._motion_stop_requested = False
            self._set_active_lane(
                lane,
                {
                    "plan_id": plan.plan_id,
                    "started_at": time.time(),
                    "command_index": 0,
                },
            )
            try:
                for index, command in enumerate(plan.commands):
                    self.active_lanes[lane]["command_index"] = index
                    self._refresh_active_task()
                    self.controller.command = PolicyCommand(
                        x_velocity=float(command["vx"]),
                        y_velocity=float(command["vy"]),
                        yaw_velocity=float(command["yaw"]),
                    )
                    completed = await self._run_segment(float(command["duration_s"]))
                    if not completed:
                        return {
                            "completed": False,
                            "stopped": True,
                            "dry_run_only": False,
                            "summary": f"Soridormi runtime stopped plan {plan_id}.",
                        }
                return {
                    "completed": True,
                    "dry_run_only": False,
                    "summary": f"Soridormi runtime completed plan {plan_id}.",
                    "estimated_duration_s": plan.estimated_duration_s,
                }
            except asyncio.CancelledError:
                await asyncio.shield(self._apply_safe_hold())
                raise
            finally:
                self.controller.command = PolicyCommand()
                self._clear_active_lane(lane)

    async def stop_motion(self, *, cancelled: bool) -> dict[str, Any]:
        self._motion_stop_requested = True
        await self._apply_safe_hold()
        key = "cancelled" if cancelled else "stopped"
        return {
            key: True,
            "safe_idle": not self.emergency_stop,
            "summary": "Soridormi runtime motion transitioned to safe hold.",
        }

    async def emergency_stop_motion(self, *, reason: str) -> dict[str, Any]:
        self.emergency_stop = True
        self._motion_stop_requested = True
        await self._apply_safe_hold()
        return {
            "stopped": True,
            "emergency": True,
            "safe_idle": False,
            "reason": reason,
        }

    async def _run_segment(self, duration_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_s
        period_s = 1.0 / self.control_hz
        while loop.time() < deadline:
            if self.emergency_stop or self._motion_stop_requested:
                return False
            started_at = loop.time()
            await self._step_controller()
            await asyncio.sleep(max(0.0, period_s - (loop.time() - started_at)))
        return True

    async def _step_controller(self) -> None:
        async with self._robot_lock:
            state = self._last_state
            if state is None:
                state = await self._call_robot(self.robot.read_state)
            command = self.controller.compute(state)
            stepper = getattr(self.robot, "step_motor_command", None)
            if callable(stepper):
                self._last_state = await self._call_robot(stepper, command)
            else:
                await self._call_robot(self.robot.send_motor_command, command)
                self._last_state = await self._call_robot(self.robot.read_state)

    async def _read_state(self) -> RobotState:
        async with self._robot_lock:
            self._last_state = await self._call_robot(self.robot.read_state)
            return self._last_state

    async def _apply_safe_hold(self) -> None:
        self.controller.command = PolicyCommand()
        async with self._robot_lock:
            state = await self._call_robot(self.robot.read_state)
            command = HoldPositionController().compute(state)
            await self._call_robot(self.robot.send_motor_command, command)
            self._last_state = state
