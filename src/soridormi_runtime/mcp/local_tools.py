from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from soridormi_runtime.skill_execution import SkillExecutionRegistry, SkillPlan
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST

_MAX_COMMANDS = 8
_MAX_TOTAL_DURATION_S = 20.0
_NO_FAULT_RESULT = object()
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


@dataclass(frozen=True)
class NamedSkillPlan:
    plan_id: str
    plan: SkillPlan
    created_at: float


@dataclass
class FaultInjection:
    scenario_id: str | None = None
    remaining_calls: int = 0

    def configure(self, scenario_id: str) -> dict[str, Any]:
        if scenario_id not in PROVIDER_READINESS_SCENARIOS:
            raise ValueError(f"unsupported fault scenario: {scenario_id}")
        self.scenario_id = scenario_id
        self.remaining_calls = 1
        return {"configured": True, "scenario_id": scenario_id}

    def clear(self) -> dict[str, Any]:
        previous = self.scenario_id
        self.scenario_id = None
        self.remaining_calls = 0
        return {"cleared": True, "previous_scenario_id": previous}

    def active(self, scenario_id: str, tool_name: str) -> bool:
        if self.scenario_id != scenario_id or self.remaining_calls <= 0:
            return False
        expected_tool = FAULT_TOOL_BY_SCENARIO.get(scenario_id)
        if expected_tool != tool_name:
            return False
        self.remaining_calls -= 1
        return True


PROVIDER_READINESS_SCENARIOS = (
    "success",
    "catalog_restart",
    "skill_unavailable",
    "plan_jitter",
    "plan_timeout",
    "plan_disconnect",
    "malformed_plan",
    "monitor_refused",
    "monitor_timeout",
    "monitor_status_drop",
    "execute_incomplete",
    "execute_skill_mismatch",
    "execute_timeout",
    "execute_disconnect",
    "runtime_timeout_cancel",
    "operator_cancel",
)
FAULT_TOOL_BY_SCENARIO = {
    "catalog_restart": "soridormi.skill.list",
    "skill_unavailable": "soridormi.skill.list",
    "plan_jitter": "soridormi.skill.create_plan",
    "plan_timeout": "soridormi.skill.create_plan",
    "plan_disconnect": "soridormi.skill.create_plan",
    "malformed_plan": "soridormi.skill.create_plan",
    "monitor_refused": "soridormi.safety.monitor_motion",
    "monitor_timeout": "soridormi.safety.monitor_motion",
    "monitor_status_drop": "soridormi.safety.monitor_motion",
    "execute_incomplete": "soridormi.skill.execute_plan",
    "execute_skill_mismatch": "soridormi.skill.execute_plan",
    "execute_timeout": "soridormi.skill.execute_plan",
    "execute_disconnect": "soridormi.skill.execute_plan",
    "runtime_timeout_cancel": "soridormi.skill.execute_plan",
    "operator_cancel": "soridormi.skill.execute_plan",
}


@dataclass
class SoridormiLocalToolService:
    """Safe in-process implementation of Soridormi MCP-style tools.

    Motion execution is intentionally dry-run only here; it never talks to
    motors or the runtime control loop.
    """

    mode: str = "sim"
    backend: str = "local_tool_dry_run"
    plans: dict[str, MotionPlan] = field(default_factory=dict)
    skill_plans: dict[str, NamedSkillPlan] = field(default_factory=dict)
    emergency_stop: bool = False
    skill_registry: SkillExecutionRegistry = field(
        default_factory=lambda: SkillExecutionRegistry.from_manifest_path(
            DEFAULT_SKILL_MANIFEST
        )
    )
    faults: FaultInjection = field(default_factory=FaultInjection)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        with self._lock:
            if tool_name == "soridormi.testing.configure_fault":
                return self.faults.configure(str(args.get("scenario_id", "")))
            if tool_name == "soridormi.testing.clear_faults":
                return self.faults.clear()
            scenario_id = self._consume_fault(tool_name)
        if scenario_id:
            injected = self._apply_fault(scenario_id)
            if injected is not _NO_FAULT_RESULT:
                return injected
        with self._lock:
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
            if tool_name == "soridormi.skill.list":
                return self.list_skills()
            if tool_name == "soridormi.skill.create_plan":
                return self.create_skill_plan(args)
            if tool_name == "soridormi.skill.execute_plan":
                return self.execute_skill_plan(str(args.get("plan_id", "")))
            if tool_name == "soridormi.safety.monitor_motion":
                return {
                    "ok": not self.emergency_stop,
                    "event": "emergency_stop" if self.emergency_stop else None,
                }
            if tool_name == "soridormi.safety.emergency_stop":
                self.emergency_stop = True
                return {
                    "stopped": True,
                    "emergency": True,
                    "reason": args.get("reason", "unspecified"),
                }
            raise KeyError(f"unknown Soridormi local tool: {tool_name}")

    def _consume_fault(self, tool_name: str) -> str | None:
        scenario_id = self.faults.scenario_id
        if not scenario_id or not self.faults.active(scenario_id, tool_name):
            return None
        return scenario_id

    def _apply_fault(self, scenario_id: str) -> dict[str, Any] | object:
        if scenario_id == "catalog_restart":
            raise ConnectionError("injected provider restart during catalog lookup")
        if scenario_id == "skill_unavailable":
            payload = self.list_skills()
            for skill in payload["skills"]:
                if skill["skill_id"] == "nod_yes":
                    skill["available"] = False
                    skill["unavailable_reason"] = "injected provider not calibrated"
            return payload
        if scenario_id == "plan_jitter":
            time.sleep(0.02)
            return _NO_FAULT_RESULT
        if scenario_id in {"plan_timeout", "monitor_timeout", "execute_timeout"}:
            raise TimeoutError(f"injected {scenario_id}")
        if scenario_id in {
            "plan_disconnect",
            "monitor_status_drop",
            "execute_disconnect",
        }:
            raise ConnectionError(f"injected {scenario_id}")
        if scenario_id == "malformed_plan":
            return {
                "plan_id": "",
                "skill_id": "nod_yes",
                "mode": self.mode,
                "summary": "injected empty plan identity",
            }
        if scenario_id == "monitor_refused":
            return {"ok": False, "event": "injected blocked workspace"}
        if scenario_id == "execute_incomplete":
            return {
                "completed": False,
                "skill_id": "nod_yes",
                "mode": self.mode,
                "no_motion": True,
                "recommendation_only": self.mode == "hardware_shadow",
            }
        if scenario_id == "execute_skill_mismatch":
            return {
                "completed": True,
                "skill_id": "wave_hand",
                "mode": self.mode,
                "no_motion": True,
                "recommendation_only": self.mode == "hardware_shadow",
            }
        if scenario_id in {"runtime_timeout_cancel", "operator_cancel"}:
            time.sleep(5)
            return _NO_FAULT_RESULT
        return _NO_FAULT_RESULT

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "standing": True,
            "fallen": False,
            "emergency_stop": self.emergency_stop,
            "active_task": None,
        }

    def list_skills(self) -> dict[str, Any]:
        skills = []
        for skill_id in self.skill_registry.executable_skill_ids():
            skill = self.skill_registry.skills[skill_id]
            parameters = skill.get("parameters", {})
            properties: dict[str, Any] = {}
            for name, rule in parameters.items():
                if not isinstance(rule, dict):
                    continue
                schema: dict[str, Any] = {}
                if rule.get("type") == "string":
                    schema["type"] = "string"
                    if isinstance(rule.get("enum"), list):
                        schema["enum"] = rule["enum"]
                else:
                    schema["type"] = "number"
                    if "min" in rule:
                        schema["minimum"] = rule["min"]
                    if "max" in rule:
                        schema["maximum"] = rule["max"]
                properties[name] = schema
            skills.append(
                {
                    "skill_id": skill_id,
                    "version": "0.1.0",
                    "available": True,
                    "parameters_schema": {
                        "type": "object",
                        "properties": properties,
                        "additionalProperties": False,
                    },
                    "interruptible": bool(
                        (skill.get("safety") or {}).get("interruptible", True)
                    ),
                }
            )
        return {"mode": self.mode, "skills": skills}

    def create_skill_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(args.get("skill_id", ""))
        if not skill_id:
            raise ValueError("skill_id is required")
        plan = self.skill_registry.create_plan(
            skill_id,
            args.get("parameters") or {},
            profile=args.get("profile"),
        )
        plan_id = f"soridormi-skill-plan-{uuid.uuid4().hex[:12]}"
        self.skill_plans[plan_id] = NamedSkillPlan(
            plan_id=plan_id,
            plan=plan,
            created_at=time.time(),
        )
        return {
            "plan_id": plan_id,
            "skill_id": skill_id,
            "mode": self.mode,
            "summary": plan.summary,
            "estimated_duration_s": plan.total_duration_s,
            "requires_confirmation": self.mode != "sim",
            "interruptible": bool((plan.safety or {}).get("interruptible", True)),
        }

    def execute_skill_plan(self, plan_id: str) -> dict[str, Any]:
        if self.emergency_stop:
            raise RuntimeError("cannot execute skill while emergency_stop is active")
        if not plan_id:
            raise ValueError("plan_id is required")
        stored = self.skill_plans.get(plan_id)
        if stored is None:
            raise KeyError(f"skill plan not found: {plan_id}")
        return {
            "completed": True,
            "skill_id": stored.plan.skill_id,
            "mode": self.mode,
            "no_motion": True,
            "recommendation_only": self.mode == "hardware_shadow",
            "summary": (
                f"Soridormi {self.mode} accepted named skill "
                f"{stored.plan.skill_id}; no robot command was sent."
            ),
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
