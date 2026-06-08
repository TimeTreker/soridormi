from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.controller import HoldPositionController
from soridormi_runtime.main import make_controller, make_robot
from soridormi_runtime.policy_command import PolicyCommand

from .local_tools import MotionPlan, SoridormiLocalToolService


class RuntimeRobot(Protocol):
    def read_state(self) -> RobotState: ...

    def send_motor_command(self, command: MotorCommand) -> None: ...


class RuntimeController(Protocol):
    command: PolicyCommand

    def compute(self, state: RobotState) -> MotorCommand: ...


@dataclass
class SoridormiRuntimeToolService:
    """MCP tool service backed by Soridormi's real runtime control interfaces."""

    robot: RuntimeRobot
    controller: RuntimeController
    mode: str = "sim"
    backend: str = "runtime"
    control_hz: float = 50.0
    plans: dict[str, MotionPlan] = field(default_factory=dict)
    emergency_stop: bool = False
    active_task: dict[str, Any] | None = None
    _motion_stop_requested: bool = field(default=False, init=False, repr=False)
    _motion_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _robot_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _last_state: RobotState | None = field(default=None, init=False, repr=False)

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
        return cls(
            robot=robot_factory(),
            controller=controller,
            mode=mode,
            backend="runtime",
            control_hz=float(os.environ.get("CONTROL_HZ", "50")),
        )

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
            state = await asyncio.to_thread(self.robot.read_state)
            command = self.controller.compute(state)
            stepper = getattr(self.robot, "step_motor_command", None)
            if callable(stepper):
                self._last_state = await asyncio.to_thread(stepper, command)
            else:
                await asyncio.to_thread(self.robot.send_motor_command, command)
                self._last_state = state

    async def _read_state(self) -> RobotState:
        async with self._robot_lock:
            self._last_state = await asyncio.to_thread(self.robot.read_state)
            return self._last_state

    async def _apply_safe_hold(self) -> None:
        self.controller.command = PolicyCommand()
        async with self._robot_lock:
            state = await asyncio.to_thread(self.robot.read_state)
            command = HoldPositionController().compute(state)
            await asyncio.to_thread(self.robot.send_motor_command, command)
            self._last_state = state
