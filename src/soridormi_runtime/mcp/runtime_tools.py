from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.controller import HoldPositionController
from soridormi_runtime.main import make_controller, make_robot
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scripted_head_skill import (
    DEFAULT_MAX_HEAD_VELOCITY_RADPS,
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
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST
from soridormi_runtime.sync_preroll import preroll_sync_simulator

from .local_tools import MotionPlan, NamedSkillPlan, SoridormiLocalToolService


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
    control_hz: float = 50.0
    plans: dict[str, MotionPlan] = field(default_factory=dict)
    skill_plans: dict[str, NamedSkillPlan] = field(default_factory=dict)
    skill_registry: SkillExecutionRegistry = field(
        default_factory=lambda: SkillExecutionRegistry.from_manifest_path(
            DEFAULT_SKILL_MANIFEST
        )
    )
    emergency_stop: bool = False
    active_task: dict[str, Any] | None = None
    _motion_stop_requested: bool = field(default=False, init=False, repr=False)
    _motion_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _robot_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _last_state: RobotState | None = field(default=None, init=False, repr=False)
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
        if tool_name == "soridormi.safety.monitor_motion":
            return {
                "ok": not self.emergency_stop,
                "active": self.active_task is not None,
                "event": "emergency_stop" if self.emergency_stop else None,
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
            if execution == "policy_velocity":
                supported.append(skill_id)
            elif execution == "scripted_keyframe" and skill_id in SUPPORTED_SCRIPTED_SKILLS:
                supported.append(skill_id)
        return tuple(supported)

    @staticmethod
    def _parameters_schema(skill: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for name, rule in (skill.get("parameters") or {}).items():
            if not isinstance(rule, dict):
                continue
            schema: dict[str, Any] = {}
            if rule.get("type") == "string" or "enum" in rule:
                schema["type"] = "string"
                if isinstance(rule.get("enum"), list):
                    schema["enum"] = list(rule["enum"])
            else:
                schema["type"] = "number"
                if "min" in rule:
                    schema["minimum"] = rule["min"]
                if "max" in rule:
                    schema["maximum"] = rule["max"]
            if "default" in rule:
                schema["default"] = rule["default"]
            properties[name] = schema
        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

    def list_runtime_skills(self) -> dict[str, Any]:
        skills: list[dict[str, Any]] = []
        for skill_id in self._runtime_skill_ids():
            skill = self.skill_registry.skills[skill_id]
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
                    "effects": ["physical_motion"],
                    "safety_class": "physical_motion",
                    "requires_confirmation": self.mode != "sim",
                    "when_to_use": str(skill.get("description") or ""),
                    "execution": skill.get("execution"),
                }
            )
        return {"mode": self.mode, "skills": skills}

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
        if plan.execution == "policy_velocity":
            motion_result = self.create_motion_plan(
                {"commands": self._motion_commands_from_skill_plan(plan)}
            )
            plan_id = str(motion_result["plan_id"])
        elif plan.execution == "scripted_keyframe":
            validate_scripted_head_plan(plan)
            plan_id = f"soridormi-skill-plan-{uuid.uuid4().hex[:12]}"
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
            "requires_confirmation": self.mode != "sim",
            "interruptible": bool((plan.safety or {}).get("interruptible", True)),
            "no_motion": False,
        }

    async def execute_runtime_skill_plan(self, plan_id: str) -> dict[str, Any]:
        if not plan_id:
            raise ValueError("plan_id is required")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        if stored.plan.execution == "policy_velocity":
            result = await self.execute_motion_plan(plan_id)
        elif stored.plan.execution == "scripted_keyframe":
            result = await self.execute_scripted_head_skill(plan_id)
        else:  # pragma: no cover - plans are validated at creation
            raise ValueError(
                f"skill {stored.plan.skill_id!r} execution "
                f"{stored.plan.execution!r} is unsupported"
            )
        return {
            **result,
            "skill_id": stored.plan.skill_id,
            "mode": self.mode,
            "no_motion": False,
            "recommendation_only": False,
        }

    async def execute_scripted_head_skill(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute skill while emergency_stop is active")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        plan = stored.plan
        validate_scripted_head_plan(plan)

        async with self._motion_lock:
            if self.emergency_stop:
                raise RuntimeError("cannot execute skill while emergency_stop is active")
            self._motion_stop_requested = False
            self.active_task = {
                "plan_id": plan_id,
                "skill_id": plan.skill_id,
                "started_at": time.time(),
                "command_index": 0,
            }
            try:
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
                            "summary": (
                                f"Soridormi runtime stopped skill {plan.skill_id}."
                            ),
                        }
                    assert self.active_task is not None
                    self.active_task["command_index"] = index
                    started_at = asyncio.get_running_loop().time()
                    state = await self._step_head_target(state, target)
                    await asyncio.sleep(
                        max(
                            0.0,
                            period_s
                            - (asyncio.get_running_loop().time() - started_at),
                        )
                    )
                return {
                    "completed": True,
                    "dry_run_only": False,
                    "summary": (
                        f"Soridormi runtime completed skill {plan.skill_id}."
                    ),
                    "estimated_duration_s": effective_duration_s,
                }
            except asyncio.CancelledError:
                await asyncio.shield(self._apply_safe_hold())
                raise
            finally:
                self.active_task = None

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
        state = await self._read_state()
        return {
            "mode": self.mode,
            "backend": self.backend,
            "standing": True,
            "fallen": False,
            "emergency_stop": self.emergency_stop,
            "active_task": dict(self.active_task) if self.active_task is not None else None,
            "robot_time": float(state.time),
        }

    async def execute_motion_plan(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute motion while emergency_stop is active")
        if not plan_id:
            raise ValueError("plan_id is required")
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"plan not found: {plan_id}")

        async with self._motion_lock:
            if self.emergency_stop:
                raise RuntimeError("cannot execute motion while emergency_stop is active")
            self._motion_stop_requested = False
            self.active_task = {
                "plan_id": plan.plan_id,
                "started_at": time.time(),
                "command_index": 0,
            }
            try:
                for index, command in enumerate(plan.commands):
                    assert self.active_task is not None
                    self.active_task["command_index"] = index
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
                self.active_task = None

    async def stop_motion(self, *, cancelled: bool) -> dict[str, Any]:
        self._motion_stop_requested = True
        await self._apply_safe_hold()
        key = "cancelled" if cancelled else "stopped"
        return {
            key: True,
            "summary": "Soridormi runtime motion transitioned to safe hold.",
        }

    async def emergency_stop_motion(self, *, reason: str) -> dict[str, Any]:
        self.emergency_stop = True
        self._motion_stop_requested = True
        await self._apply_safe_hold()
        return {
            "stopped": True,
            "emergency": True,
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
