from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_MAX_COMMANDS = 8
_MAX_TOTAL_DURATION_S = 20.0
_LIMITS = {
    "vx": (-0.2, 0.2),
    "vy": (-0.1, 0.1),
    "yaw": (-0.4, 0.4),
    "duration_s": (0.05, 5.0),
}


@dataclass(frozen=True)
class MotionPlan:
    plan_id: str
    commands: tuple[dict[str, Any], ...]
    created_at: float
    estimated_duration_s: float
    summary: str
    dry_run_only: bool = True


@dataclass
class SoridormiLocalToolService:
    """Safe in-process implementation of Soridormi MCP-style tools.

    This is not a network MCP server yet. It is the robot-side tool core that a
    future stdio/HTTP MCP server can wrap. Motion execution is intentionally
    dry-run only here; it never talks to motors or the runtime control loop.
    """

    mode: str = "sim"
    backend: str = "local_tool_dry_run"
    plans: dict[str, MotionPlan] = field(default_factory=dict)
    emergency_stop: bool = False

    def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        if tool_name == "soridormi.robot.get_status":
            return self.get_status()
        if tool_name == "soridormi.robot.get_mode":
            return {"mode": self.mode}
        if tool_name == "soridormi.robot.get_battery":
            return {"percent": None, "critical": False}
        if tool_name == "soridormi.motion.create_plan":
            return self.create_motion_plan(args)
        if tool_name == "soridormi.motion.execute_plan":
            return self.execute_motion_plan(str(args.get("plan_id", "")))
        if tool_name == "soridormi.motion.stop":
            return {"stopped": True, "summary": "Soridormi local dry-run stop accepted."}
        if tool_name == "soridormi.motion.cancel":
            return {"cancelled": True, "summary": "Soridormi local dry-run motion cancel accepted."}
        if tool_name == "soridormi.safety.monitor_motion":
            return {"ok": not self.emergency_stop, "event": "emergency_stop" if self.emergency_stop else None}
        if tool_name == "soridormi.safety.emergency_stop":
            self.emergency_stop = True
            return {"stopped": True, "emergency": True, "reason": args.get("reason", "unspecified")}
        raise KeyError(f"unknown Soridormi local tool: {tool_name}")

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "standing": True,
            "fallen": False,
            "emergency_stop": self.emergency_stop,
            "active_task": None,
        }

    def create_motion_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        commands = args.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ValueError("commands must be a non-empty list")
        if len(commands) > _MAX_COMMANDS:
            raise ValueError(f"commands may contain at most {_MAX_COMMANDS} entries")

        normalized: list[dict[str, Any]] = []
        total_duration = 0.0
        for index, command in enumerate(commands):
            if not isinstance(command, dict):
                raise ValueError(f"command[{index}] must be an object")
            normalized_command: dict[str, Any] = {}
            for field_name, (minimum, maximum) in _LIMITS.items():
                if field_name not in command:
                    raise ValueError(f"command[{index}] missing {field_name}")
                value = float(command[field_name])
                if value < minimum or value > maximum:
                    raise ValueError(f"command[{index}].{field_name}={value} outside [{minimum}, {maximum}]")
                normalized_command[field_name] = value
            if "label" in command:
                normalized_command["label"] = str(command["label"])
            total_duration += normalized_command["duration_s"]
            normalized.append(normalized_command)

        if total_duration > _MAX_TOTAL_DURATION_S:
            raise ValueError(f"motion plan duration {total_duration:.2f}s exceeds {_MAX_TOTAL_DURATION_S:.2f}s")

        plan_id = f"soridormi-plan-{uuid.uuid4().hex[:12]}"
        summary = f"Soridormi dry-run motion plan with {len(normalized)} command(s), {total_duration:.2f}s total."
        self.plans[plan_id] = MotionPlan(
            plan_id=plan_id,
            commands=tuple(normalized),
            created_at=time.time(),
            estimated_duration_s=total_duration,
            summary=summary,
        )
        return {
            "plan_id": plan_id,
            "summary": summary,
            "estimated_duration_s": total_duration,
            "requires_confirmation": True,
            "dry_run_only": True,
        }

    def execute_motion_plan(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute motion while emergency_stop is active")
        if not plan_id:
            raise ValueError("plan_id is required")
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"plan not found: {plan_id}")
        return {
            "completed": True,
            "dry_run_only": True,
            "summary": f"Soridormi local dry-run accepted plan {plan_id}; no robot motion was sent.",
            "estimated_duration_s": plan.estimated_duration_s,
        }
