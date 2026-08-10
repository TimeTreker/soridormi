from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .dag_contract import build_soridormi_dag_contract
from .task_tools import KNOWN_TASK_TYPES, TASK_PHASES

SafetyClass = Literal[
    "safe_read",
    "planning_only",
    "low_risk_action",
    "physical_motion",
    "safety_critical",
    "restricted",
]
FailureStrategy = Literal[
    "retry",
    "ask_user",
    "skip",
    "continue_with_default",
    "goto",
    "abort_task",
    "stop_and_report",
    "emergency_stop",
]


class TransportSpec(BaseModel):
    kind: str = "stdio"
    module: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class AgentStatus(BaseModel):
    available: bool = True
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ToolAvailability(BaseModel):
    available: bool = True
    modes: list[str] = Field(default_factory=lambda: ["sim", "hardware_dry_run"])
    requires: list[str] = Field(default_factory=list)
    reason: str | None = None


class ExecutionPolicy(BaseModel):
    can_run_parallel: bool = True
    exclusive_group: str | None = None
    timeout_s: float | None = Field(default=None, gt=0)
    idempotent: bool = True
    side_effect_free: bool = True


class ConfirmationPolicy(BaseModel):
    required: bool = False
    reason: str | None = None
    required_in_modes: list[str] = Field(default_factory=list)
    skippable_in_modes: list[str] = Field(default_factory=list)


class MonitoringPolicy(BaseModel):
    requires_safety_monitor: bool = False
    recommended_monitor_tools: list[str] = Field(default_factory=list)
    hard_interrupt_events: list[str] = Field(default_factory=list)


class FailurePolicy(BaseModel):
    strategy: FailureStrategy = "abort_task"
    target: str | None = None
    message: str | None = None
    default_output: dict[str, Any] | None = None
    max_attempts: int | None = Field(default=None, ge=1)
    backoff_s: float | None = Field(default=None, ge=0)
    then: "FailurePolicy | None" = None


class ToolCapability(BaseModel):
    name: str
    agent_id: str
    display_name: str | None = None
    description: str = ""
    version: str = "0.1.0"
    llm_visible: bool = True
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    effects: list[str] = Field(default_factory=list)
    safety_class: SafetyClass = "safe_read"
    availability: ToolAvailability = Field(default_factory=ToolAvailability)
    execution: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    confirmation: ConfirmationPolicy = Field(default_factory=ConfirmationPolicy)
    monitoring: MonitoringPolicy = Field(default_factory=MonitoringPolicy)
    failure_modes: list[str] = Field(default_factory=list)
    default_failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)
    llm_hints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def hide_restricted_tools(self) -> "ToolCapability":
        if self.safety_class == "restricted":
            self.llm_visible = False
        return self


class AgentManifest(BaseModel):
    agent_id: str
    display_name: str | None = None
    description: str = ""
    version: str = "0.1.0"
    llm_visible: bool = True
    transport: TransportSpec = Field(default_factory=TransportSpec)
    status: AgentStatus = Field(default_factory=AgentStatus)
    tags: list[str] = Field(default_factory=list)
    tools: list[ToolCapability] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_ownership(self) -> "AgentManifest":
        seen: set[str] = set()
        for tool in self.tools:
            if tool.agent_id != self.agent_id:
                raise ValueError(f"tool {tool.name!r} has mismatched agent_id {tool.agent_id!r}")
            if tool.name in seen:
                raise ValueError(f"duplicate tool: {tool.name}")
            seen.add(tool.name)
        return self


class CapabilityBundle(BaseModel):
    schema_version: str = "0.1"
    source: str = "soridormi"
    agents: list[AgentManifest] = Field(default_factory=list)
    dag_contract: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _chromie_intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Chromie-side proposal metadata. Soridormi treats it as advisory "
            "intent, never as a hardware command."
        ),
        "properties": {
            "execution_mode": {"type": "string", "const": "proposed"},
            "execution_semantics": {
                "type": "string",
                "const": "proposal_from_chromie",
            },
            "requires_runtime_validation": {"type": "boolean", "const": True},
            "interaction_id": {"type": "string"},
            "request_id": {"type": "string"},
            "skill_id": {"type": "string"},
            "upstream_skill_id": {"type": "string"},
            "source_component": {"type": "string"},
            "requires_live_perception": {"type": "boolean"},
            "perception_dependency": {"type": "string"},
            "physical_state_source": {
                "type": "string",
                "const": "soridormi_runtime",
            },
            "chromie_must_not_provide_physical_coordinates": {
                "type": "boolean",
                "const": True,
            },
            "soridormi_owns_pose_estimation": {
                "type": "boolean",
                "const": True,
            },
        },
        "required": [
            "execution_mode",
            "execution_semantics",
            "requires_runtime_validation",
        ],
        "additionalProperties": True,
    }


def build_soridormi_capability_bundle(*, mode: str = "sim") -> CapabilityBundle:
    """Return Soridormi's robot-body capability manifest.

    Chromie owns the global registry. Soridormi only declares the capabilities of
    the robot runtime boundary it owns. There are intentionally no
    `chromie.speak` or user-confirmation tools here.
    """

    safe_modes = ["sim", "hardware_shadow", "hardware_dry_run"]
    robot_agent = AgentManifest(
        agent_id="soridormi.robot",
        display_name="Soridormi Robot State Agent",
        description="Read-only Soridormi robot status, mode, and battery state.",
        transport=TransportSpec(kind="local_cli", command="python", args=["-m", "soridormi_runtime.mcp.call_tool"]),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "robot", "state"],
        tools=[
            ToolCapability(
                name="soridormi.robot.get_status",
                agent_id="soridormi.robot",
                display_name="Get Soridormi status",
                description="Read Soridormi status including mode, backend, safety state, and active task metadata.",
                input_schema=_object_schema({}),
                output_schema=_object_schema({
                    "mode": {"type": "string"},
                    "backend": {"type": "string"},
                    "standing": {"type": "boolean"},
                    "fallen": {"type": "boolean"},
                    "emergency_stop": {"type": "boolean"},
                    "active_task": {"type": ["object", "null"]},
                    "active_lanes": {"type": "object"},
                    "activity_idle": {"type": "boolean"},
                    "safe_idle": {"type": "boolean"},
                    "robot_time": {"type": "number"},
                    "source_revision": {
                        "type": "string",
                        "description": "Provider-reported source revision for evidence binding.",
                    },
                }),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=True, timeout_s=1.0, idempotent=True, side_effect_free=True),
                default_failure_policy=FailurePolicy(strategy="abort_task"),
                llm_hints={"when_to_use": "Use before planning or executing robot movement."},
            ),
            ToolCapability(
                name="soridormi.robot.get_mode",
                agent_id="soridormi.robot",
                description="Read the current Soridormi runtime mode such as sim, hardware_dry_run, or hardware.",
                input_schema=_object_schema({}),
                output_schema=_object_schema({"mode": {"type": "string"}}, required=["mode"]),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=True, timeout_s=1.0, idempotent=True, side_effect_free=True),
            ),
            ToolCapability(
                name="soridormi.robot.get_battery",
                agent_id="soridormi.robot",
                description="Read battery percentage and power safety status when available.",
                input_schema=_object_schema({}),
                output_schema=_object_schema({"percent": {"type": ["number", "null"]}, "critical": {"type": "boolean"}}),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"], reason="Sim may return null battery percent."),
                execution=ExecutionPolicy(can_run_parallel=True, timeout_s=1.0, idempotent=True, side_effect_free=True),
            ),
        ],
    )

    motion_command_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "vx": {"type": "number", "minimum": -0.2, "maximum": 0.2},
                "vy": {"type": "number", "minimum": -0.1, "maximum": 0.1},
                "yaw": {"type": "number", "minimum": -0.4, "maximum": 0.4},
                "duration_s": {"type": "number", "minimum": 0.05, "maximum": 5.0},
                "label": {"type": "string"},
            },
            "required": ["vx", "vy", "yaw", "duration_s"],
        },
        "minItems": 1,
        "maxItems": 8,
    }
    motion_agent = AgentManifest(
        agent_id="soridormi.motion",
        display_name="Soridormi Motion Agent",
        description="Create and execute short velocity-command motion plans. Does not expose raw motor commands.",
        transport=TransportSpec(kind="local_cli", command="python", args=["-m", "soridormi_runtime.mcp.call_tool"]),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "motion", "robot"],
        tools=[
            ToolCapability(
                name="soridormi.motion.create_plan",
                agent_id="soridormi.motion",
                display_name="Create Soridormi motion plan",
                description="Create a bounded short-range velocity-command plan. This does not move the robot.",
                input_schema=_object_schema({"commands": motion_command_schema}, required=["commands"]),
                output_schema=_object_schema({
                    "plan_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "estimated_duration_s": {"type": "number"},
                    "requires_confirmation": {"type": "boolean"},
                }, required=["plan_id", "summary"]),
                effects=["planning_only", "creates_plan"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(can_run_parallel=True, timeout_s=2.0, idempotent=True, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=["invalid_command", "duration_too_long", "velocity_limit_exceeded"],
                default_failure_policy=FailurePolicy(strategy="ask_user"),
                llm_hints={
                    "when_to_use": "Use for short movements like walking forward a little, turning, or stopping.",
                    "when_not_to_use": "Do not use for named-location navigation; use a future nav agent instead.",
                },
            ),
            ToolCapability(
                name="soridormi.motion.execute_plan",
                agent_id="soridormi.motion",
                display_name="Execute Soridormi motion plan",
                description="Execute a previously created and validated Soridormi motion plan.",
                input_schema=_object_schema({"plan_id": {"type": "string", "minLength": 1}}, required=["plan_id"]),
                output_schema=_object_schema({"completed": {"type": "boolean"}, "summary": {"type": "string"}}),
                effects=["physical_motion"],
                safety_class="physical_motion",
                availability=ToolAvailability(modes=safe_modes, requires=["robot_standing", "not_emergency_stopped"]),
                execution=ExecutionPolicy(can_run_parallel=False, exclusive_group="soridormi.robot_motion", timeout_s=15.0, idempotent=False, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=True, reason="This tool can move the robot.", required_in_modes=["hardware_dry_run", "hardware"], skippable_in_modes=["sim"]),
                monitoring=MonitoringPolicy(
                    requires_safety_monitor=True,
                    recommended_monitor_tools=["soridormi.safety.monitor_motion"],
                    hard_interrupt_events=["fall_detected", "emergency_stop", "collision_risk", "low_battery_critical"],
                ),
                failure_modes=["plan_not_found", "plan_expired", "motion_failed", "safety_interrupted", "timeout"],
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
                llm_hints={"when_to_use": "Use only after create_plan succeeds and Chromie has handled user confirmation."},
            ),
            ToolCapability(
                name="soridormi.motion.stop",
                agent_id="soridormi.motion",
                description="Stop current Soridormi motion as quickly as the safe command layer allows.",
                input_schema=_object_schema({}),
                output_schema=_object_schema({"stopped": {"type": "boolean"}, "safe_idle": {"type": "boolean"}}),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=False, exclusive_group="soridormi.robot_motion", timeout_s=1.0, idempotent=True, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=["runtime_unreachable"],
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
            ),
            ToolCapability(
                name="soridormi.motion.cancel",
                agent_id="soridormi.motion",
                description="Cancel the current Soridormi motion plan and transition to stop.",
                input_schema=_object_schema({}),
                output_schema=_object_schema({"cancelled": {"type": "boolean"}, "safe_idle": {"type": "boolean"}}),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=False, exclusive_group="soridormi.robot_motion", timeout_s=1.0, idempotent=True, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=False),
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
            ),
        ],
    )

    safety_agent = AgentManifest(
        agent_id="soridormi.safety",
        display_name="Soridormi Safety Agent",
        description="Soridormi safety validation, motion monitoring, and emergency stop capabilities.",
        transport=TransportSpec(kind="local_cli", command="python", args=["-m", "soridormi_runtime.mcp.call_tool"]),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "safety"],
        tools=[
            ToolCapability(
                name="soridormi.safety.monitor_motion",
                agent_id="soridormi.safety",
                description="Monitor safety state while a Soridormi physical-motion node is running.",
                input_schema=_object_schema({"during_node_id": {"type": "string"}}),
                output_schema=_object_schema({
                    "event": {"type": ["string", "null"]},
                    "ok": {"type": "boolean"},
                    "safe_idle": {"type": "boolean"},
                }),
                effects=["read_only", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=True, exclusive_group=None, timeout_s=60.0, idempotent=False, side_effect_free=False),
                monitoring=MonitoringPolicy(requires_safety_monitor=False),
                failure_modes=["monitor_unavailable", "fall_detected", "emergency_stop", "collision_risk"],
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
            ),
            ToolCapability(
                name="soridormi.safety.emergency_stop",
                agent_id="soridormi.safety",
                description="Hard-stop Soridormi and abort robot motion. This safety tool can preempt other tasks.",
                input_schema=_object_schema({"reason": {"type": "string"}}),
                output_schema=_object_schema({
                    "stopped": {"type": "boolean"},
                    "emergency": {"type": "boolean"},
                    "safe_idle": {"type": "boolean"},
                }),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=[*safe_modes, "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=False, exclusive_group="soridormi.robot_motion", timeout_s=1.0, idempotent=True, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=["runtime_unreachable"],
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
                llm_hints={"when_to_use": "Use immediately for fall, collision risk, emergency stop, or user stop/cancel."},
            ),
        ],
    )

    skill_agent = AgentManifest(
        agent_id="soridormi.skill",
        display_name="Soridormi Named Skill Agent",
        description=(
            "Discover, plan, and execute named body skills through opaque "
            "Soridormi-owned plans."
        ),
        transport=TransportSpec(
            kind="local_cli",
            command="python",
            args=["-m", "soridormi_runtime.mcp.call_tool"],
        ),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "skill", "body-skill"],
        tools=[
            ToolCapability(
                name="soridormi.skill.list",
                agent_id="soridormi.skill",
                description="List versioned named skills and parameter schemas.",
                input_schema=_object_schema({}),
                output_schema=_object_schema(
                    {
                        "mode": {"type": "string"},
                        "skills": {"type": "array", "items": {"type": "object"}},
                    },
                    required=["mode", "skills"],
                ),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=2.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
            ),
            ToolCapability(
                name="soridormi.skill.create_plan",
                agent_id="soridormi.skill",
                description=(
                    "Validate a named skill and create an opaque Soridormi-owned "
                    "plan. Planning does not move the robot."
                ),
                input_schema=_object_schema(
                    {
                        "skill_id": {"type": "string", "minLength": 1},
                        "parameters": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "profile": {"type": "string"},
                        "chromie_intent": _chromie_intent_schema(),
                    },
                    required=["skill_id"],
                ),
                output_schema=_object_schema(
                    {
                        "plan_id": {"type": "string"},
                        "skill_id": {"type": "string"},
                        "mode": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    required=["plan_id", "skill_id", "summary"],
                ),
                effects=["planning_only", "creates_plan"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=2.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                llm_hints={
                    "chromie_intent_contract": (
                        "Chromie sends proposal metadata only; Soridormi validates, "
                        "plans, monitors, and may refuse every physical execution."
                    )
                },
            ),
            ToolCapability(
                name="soridormi.skill.execute_plan",
                agent_id="soridormi.skill",
                description=(
                    "Execute an opaque named-skill plan. The runtime adapter may "
                    "move the MuJoCo robot in sim mode; hardware shadow/dry-run "
                    "profiles remain no-motion until a hardware adapter is implemented."
                ),
                input_schema=_object_schema(
                    {"plan_id": {"type": "string", "minLength": 1}},
                    required=["plan_id"],
                ),
                output_schema=_object_schema(
                    {
                        "completed": {"type": "boolean"},
                        "skill_id": {"type": "string"},
                        "mode": {"type": "string"},
                        "no_motion": {"type": "boolean"},
                        "recommendation_only": {"type": "boolean"},
                        "summary": {"type": "string"},
                        "resource_outcome": {
                            "type": "object",
                            "description": (
                                "Provider evidence for acquire_and_deliver_resource "
                                "skills; omitted for unrelated named skills."
                            ),
                        },
                    },
                    required=["completed", "skill_id", "no_motion"],
                ),
                effects=["physical_motion", "named_skill_execution"],
                safety_class="physical_motion",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=False,
                    exclusive_group="soridormi.robot_motion",
                    timeout_s=30.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(
                    required=True,
                    reason="This exercises a body-skill provider contract.",
                    required_in_modes=["hardware_shadow", "hardware_dry_run"],
                    skippable_in_modes=["sim"],
                ),
                monitoring=MonitoringPolicy(
                    requires_safety_monitor=True,
                    recommended_monitor_tools=[
                        "soridormi.safety.monitor_motion"
                    ],
                    hard_interrupt_events=["emergency_stop"],
                ),
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
            ),
        ],
    )

    activity_member_schema = {
        "type": "object",
        "properties": {
            "member_id": {"type": "string", "minLength": 1},
            "skill_id": {"type": "string", "minLength": 1},
            "parameters": {"type": "object", "additionalProperties": True},
            "optional": {"type": "boolean"},
        },
        "required": ["skill_id"],
        "additionalProperties": False,
    }
    activity_plan_schema = _object_schema(
        {
            "coordination_id": {"type": "string", "minLength": 1},
            "members": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": activity_member_schema,
            },
            "profile": {"type": "string"},
            "chromie_intent": _chromie_intent_schema(),
        },
        required=["members"],
    )
    activity_plan_schema["additionalProperties"] = False
    activity_status_schema = _object_schema(
        {
            "schema_version": {"type": "string"},
            "compiled_activity_id": {"type": "string"},
            "plan_id": {"type": "string"},
            "coordination_id": {"type": ["string", "null"]},
            "mode": {"type": "string"},
            "backend": {"type": "string"},
            "status": {
                "type": "string",
                "enum": [
                    "planned",
                    "running",
                    "completed",
                    "completed_with_degradation",
                    "cancelled",
                    "failed",
                ],
            },
            "terminal": {"type": "boolean"},
            "cancel_requested": {"type": "boolean"},
            "cancel_reason": {"type": ["string", "null"]},
            "failure_reason": {"type": ["string", "null"]},
            "estimated_duration_s": {"type": "number"},
            "dry_run_only": {"type": "boolean"},
            "members": {"type": "array", "items": {"type": "object"}},
            "member_results": {"type": "object"},
            "resource_claims": {"type": "array", "items": {"type": "string"}},
            "safety_authority": {"type": "string", "const": "soridormi"},
            "speech_owner": {"type": "string", "const": "chromie"},
            "one_final_motor_command_authority": {"type": "boolean", "const": True},
            "semantic_role": {"type": "string", "const": "deterministic_embodied_compiler"},
            "cognitive_planning": {"type": "boolean", "const": False},
            "llm_required": {"type": "boolean", "const": False},
        },
        required=[
            "schema_version",
            "compiled_activity_id",
            "plan_id",
            "status",
            "terminal",
            "members",
            "resource_claims",
            "safety_authority",
            "speech_owner",
            "one_final_motor_command_authority",
            "semantic_role",
            "cognitive_planning",
            "llm_required",
        ],
    )
    activity_agent = AgentManifest(
        agent_id="soridormi.activity",
        display_name="Soridormi Concurrent Body Activity Agent",
        description=(
            "Validate and execute exact coordinated body activities containing one "
            "primary locomotion member plus compatible subtle-expression members. "
            "Speech remains a peer Chromie execution lane."
        ),
        transport=TransportSpec(
            kind="local_cli",
            command="python",
            args=["-m", "soridormi_runtime.mcp.call_tool"],
        ),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "body-activity", "concurrency", "safety"],
        tools=[
            ToolCapability(
                name="soridormi.activity.get_capabilities",
                agent_id="soridormi.activity",
                description=(
                    "Read Soridormi's body ability classes, resource model, "
                    "concurrency rules, and Chromie speech ownership boundary."
                ),
                input_schema=_object_schema({}),
                output_schema=_object_schema(
                    {
                        "schema_version": {"type": "string"},
                        "speech_owner": {"type": "string", "const": "chromie"},
                        "semantic_role": {"type": "string", "const": "deterministic_embodied_compiler"},
                        "cognitive_planning": {"type": "boolean", "const": False},
                        "llm_required": {"type": "boolean", "const": False},
                        "canonical_tools": {"type": "object"},
                        "compatibility_aliases": {"type": "object"},
                        "ability_classes": {"type": "array", "items": {"type": "object"}},
                        "control_couplings": {"type": "array", "items": {"type": "string"}},
                        "concurrency_rules": {"type": "object"},
                        "coordination": {"type": "object"},
                    },
                    required=[
                        "schema_version",
                        "speech_owner",
                        "semantic_role",
                        "cognitive_planning",
                        "llm_required",
                        "ability_classes",
                        "control_couplings",
                        "concurrency_rules",
                    ],
                ),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
            ),
            ToolCapability(
                name="soridormi.activity.compile",
                agent_id="soridormi.activity",
                description=(
                    "Validate exact body members, resource compatibility, bounded "
                    "head-overlay envelopes, and compile an opaque executable body activity. This is deterministic embodied compilation, not cognitive planning."
                ),
                input_schema=activity_plan_schema,
                output_schema=activity_status_schema,
                effects=["planning_only", "compiles_body_activity", "resource_validation"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=3.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                llm_hints={
                    "when_to_use": (
                        "Use after the authoritative planner has selected exact body "
                        "skills that must run concurrently. Do not include speech here."
                    ),
                    "speech_boundary": (
                        "Chromie's Speaking Execution Lane owns speech and singing; "
                        "link peer execution with coordination_id only."
                    ),
                },
            ),
            ToolCapability(
                name="soridormi.activity.execute",
                agent_id="soridormi.activity",
                description=(
                    "Execute a validated concurrent body-activity plan. Soridormi "
                    "composes one final motor command and may reject, constrain, stop, "
                    "or recover independently for physical safety."
                ),
                input_schema=_object_schema(
                    {"compiled_activity_id": {"type": "string", "minLength": 1}},
                    required=["compiled_activity_id"],
                ),
                output_schema=activity_status_schema,
                effects=["physical_motion", "visual_expression", "body_activity_execution"],
                safety_class="physical_motion",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=False,
                    exclusive_group="soridormi.body_activity_scheduler",
                    timeout_s=60.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(
                    required=True,
                    reason="A coordinated body activity may move the robot.",
                    required_in_modes=["hardware_shadow", "hardware_dry_run"],
                    skippable_in_modes=["sim"],
                ),
                monitoring=MonitoringPolicy(
                    requires_safety_monitor=True,
                    recommended_monitor_tools=["soridormi.safety.monitor_motion"],
                    hard_interrupt_events=["emergency_stop"],
                ),
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
            ),
            ToolCapability(
                name="soridormi.activity.create_plan",
                agent_id="soridormi.activity",
                description=(
                    "Compatibility alias for soridormi.activity.compile. "
                    "It deterministically compiles exact body members and does not perform cognitive planning."
                ),
                llm_visible=False,
                input_schema=activity_plan_schema,
                output_schema=activity_status_schema,
                effects=["planning_only", "compiles_body_activity", "resource_validation"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=3.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
            ),
            ToolCapability(
                name="soridormi.activity.execute_plan",
                agent_id="soridormi.activity",
                description="Compatibility alias for soridormi.activity.execute.",
                llm_visible=False,
                input_schema=_object_schema(
                    {"plan_id": {"type": "string", "minLength": 1}},
                    required=["plan_id"],
                ),
                output_schema=activity_status_schema,
                effects=["physical_motion", "visual_expression", "body_activity_execution"],
                safety_class="physical_motion",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=False,
                    exclusive_group="soridormi.body_activity_scheduler",
                    timeout_s=60.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(
                    required=True,
                    reason="A coordinated body activity may move the robot.",
                    required_in_modes=["hardware_shadow", "hardware_dry_run"],
                    skippable_in_modes=["sim"],
                ),
                monitoring=MonitoringPolicy(
                    requires_safety_monitor=True,
                    recommended_monitor_tools=["soridormi.safety.monitor_motion"],
                    hard_interrupt_events=["emergency_stop"],
                ),
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
            ),
            ToolCapability(
                name="soridormi.activity.status",
                agent_id="soridormi.activity",
                description="Read per-member and aggregate body-activity status.",
                input_schema=_object_schema(
                    {"compiled_activity_id": {"type": "string", "minLength": 1}},
                    required=["compiled_activity_id"],
                ),
                output_schema=activity_status_schema,
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
            ),
            ToolCapability(
                name="soridormi.activity.cancel",
                agent_id="soridormi.activity",
                description=(
                    "Cancel one coordinated body activity. Physical safety may still "
                    "escalate independently to emergency stop or recovery."
                ),
                input_schema=_object_schema(
                    {
                        "compiled_activity_id": {"type": "string", "minLength": 1},
                        "reason": {"type": "string"},
                    },
                    required=["compiled_activity_id"],
                ),
                output_schema=activity_status_schema,
                effects=["physical_motion", "safety_control", "activity_lifecycle"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=2.0,
                    idempotent=True,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(required=False),
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
            ),
        ],
    )

    context_object_schema = {"type": "object", "additionalProperties": True}
    task_submit_schema = _object_schema(
        {
            "client_task_ref": {"type": "string", "minLength": 1, "maxLength": 128},
            "task_type": {"type": "string", "enum": list(KNOWN_TASK_TYPES)},
            "summary": {"type": "string"},
            "parameters": context_object_schema,
            "task_context": context_object_schema,
            "environment_context": context_object_schema,
            "safety_constraints": context_object_schema,
            "timeout_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 300},
            "cancellation_policy": {
                "type": "string",
                "enum": [
                    "best_effort_stop",
                    "cancel_before_execution",
                    "emergency_stop_on_timeout",
                ],
            },
        },
        required=["task_type"],
    )
    task_submit_schema["additionalProperties"] = False
    task_status_payload_schema = _object_schema(
        {
            "task_id": {"type": "string"},
            "client_task_ref": {"type": ["string", "null"]},
            "idempotent_replay": {"type": "boolean"},
            "task_type": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["accepted", "completed", "cancelled", "failed", "refused"],
            },
            "phase": {"type": "string", "enum": list(TASK_PHASES)},
            "terminal": {"type": "boolean"},
            "allowed_next_phases": {
                "type": "array",
                "items": {"type": "string", "enum": list(TASK_PHASES)},
            },
            "accepted": {"type": "boolean"},
            "mode": {"type": "string"},
            "backend": {"type": "string"},
            "summary": {"type": "string"},
            "reason_code": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"]},
            "safe_idle": {"type": "boolean"},
            "no_motion": {"type": "boolean"},
            "execution_mode": {"type": "string"},
            "skill_id": {"type": ["string", "null"]},
            "skill_summary": {"type": ["string", "null"]},
            "skill_sequence": {"type": "array", "items": {"type": "object"}},
            "plan_steps": {"type": "array", "items": {"type": "object"}},
            "task_graph": {"type": "object"},
            "blocked_subsystems": {
                "type": "array",
                "items": {"type": "string"},
            },
            "recommended_next_actions": {
                "type": "array",
                "items": {"type": "object"},
            },
            "estimated_duration_s": {"type": ["number", "null"]},
            "created_at": {"type": "number"},
            "updated_at": {"type": "number"},
            "deadline_at": {"type": "number"},
            "timeout_s": {"type": "number"},
            "expired": {"type": "boolean"},
            "timeout_elapsed_s": {"type": ["number", "null"]},
            "cancellation_policy": {"type": "string"},
            "events_count": {"type": "integer"},
        },
        required=[
            "task_id",
            "client_task_ref",
            "idempotent_replay",
            "task_type",
            "status",
            "phase",
            "terminal",
            "accepted",
            "safe_idle",
            "no_motion",
            "deadline_at",
            "expired",
        ],
    )
    task_preview_payload_schema = _object_schema(
        {
            key: value
            for key, value in task_status_payload_schema["properties"].items()
            if key != "task_id"
        }
        | {
            "preview_id": {"type": "string"},
            "persistent": {"type": "boolean"},
            "submit_tool": {"type": "string"},
            "would_record_task_on_submit": {"type": "boolean"},
        },
        required=[
            "preview_id",
            "task_type",
            "status",
            "phase",
            "terminal",
            "accepted",
            "safe_idle",
            "no_motion",
            "persistent",
            "would_record_task_on_submit",
        ],
    )
    task_agent = AgentManifest(
        agent_id="soridormi.task",
        display_name="Soridormi Embodied Task Agent",
        description=(
            "Contract-first embodied task API. It records structured goals and "
            "refuses missing, unsafe, or unsupported capability paths; this "
            "surface does not execute robot motion yet."
        ),
        transport=TransportSpec(
            kind="local_cli",
            command="python",
            args=["-m", "soridormi_runtime.mcp.call_tool"],
        ),
        status=AgentStatus(
            available=True,
            details={"mode": mode, "execution_mode": "contract_only"},
        ),
        tags=["soridormi", "task", "embodied-goal", "contract-only"],
        tools=[
            ToolCapability(
                name="soridormi.task.get_capabilities",
                agent_id="soridormi.task",
                display_name="Get Soridormi task capabilities",
                description=(
                    "Read Soridormi-owned embodied task readiness, required "
                    "subsystems, missing subsystems, and no-motion task API status."
                ),
                input_schema=_object_schema({}),
                output_schema=_object_schema(
                    {
                        "schema_version": {"type": "string"},
                        "mode": {"type": "string"},
                        "backend": {"type": "string"},
                        "emergency_stop": {"type": "boolean"},
                        "safe_idle": {"type": "boolean"},
                        "readiness_profile": {"type": "string"},
                        "task_api_no_motion": {"type": "boolean"},
                        "physical_execution_note": {"type": "string"},
                        "ready_subsystems": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unsafe_task_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "executable_skill_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "task_types": {"type": "array", "items": {"type": "object"}},
                    },
                    required=[
                        "schema_version",
                        "mode",
                        "backend",
                        "readiness_profile",
                        "task_api_no_motion",
                        "ready_subsystems",
                        "task_types",
                    ],
                ),
                effects=["read_only", "task_capability_readiness"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
                confirmation=ConfirmationPolicy(required=False),
                llm_hints={
                    "when_to_use": (
                        "Use before planning embodied tasks to inspect what "
                        "Soridormi can dry-run, hold, redirect, or fail closed."
                    ),
                    "physical_execution_boundary": (
                        "This reports task readiness only. It does not authorize "
                        "physical motion or raw low-level control."
                    ),
                },
            ),
            ToolCapability(
                name="soridormi.task.preview",
                agent_id="soridormi.task",
                display_name="Preview Soridormi embodied task",
                description=(
                    "Preview Soridormi's no-motion interpretation of a structured "
                    "embodied task goal without creating a persistent task record."
                ),
                input_schema=task_submit_schema,
                output_schema=task_preview_payload_schema,
                effects=["planning_only", "embodied_task_preview", "no_motion_contract"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=2.0,
                    idempotent=False,
                    side_effect_free=True,
                ),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=[
                    "unsupported_task_type",
                    "unsafe_task",
                    "missing_navigation_pipeline",
                    "missing_perception_pipeline",
                    "missing_manipulation_capability",
                    "emergency_stop_active",
                    "low_level_control_rejected",
                ],
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
                llm_hints={
                    "when_to_use": (
                        "Use before task submission when Chromie needs to explain, "
                        "clarify, or verify Soridormi's embodied interpretation."
                    ),
                    "when_not_to_use": (
                        "Do not treat preview as task execution or as a persistent "
                        "task id; use soridormi.task.submit to create a task record."
                    ),
                    "no_motion_until": "task_execution_state_machine",
                    "forbidden_fields": [
                        "action_14d",
                        "joint_targets",
                        "motor_commands",
                        "torque_commands",
                        "actuator_ctrl",
                    ],
                    "plan_step_boundary": (
                        "plan_steps describe Soridormi-owned embodied planning "
                        "layers and never contain raw low-level robot control."
                    ),
                    "next_action_boundary": (
                        "recommended_next_actions are routing hints for Chromie; "
                        "they are not execution receipts or low-level commands."
                    ),
                },
            ),
            ToolCapability(
                name="soridormi.task.submit",
                agent_id="soridormi.task",
                display_name="Submit Soridormi embodied task",
                description=(
                    "Submit a structured embodied task goal. The contract "
                    "surface validates and records the request, but returns "
                    "no_motion=true until task execution is implemented."
                ),
                input_schema=task_submit_schema,
                output_schema=task_status_payload_schema,
                effects=["planning_only", "embodied_task_request", "no_motion_contract"],
                safety_class="planning_only",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=2.0,
                    idempotent=False,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=[
                    "unsupported_task_type",
                    "unsafe_task",
                    "missing_navigation_pipeline",
                    "missing_perception_pipeline",
                    "missing_manipulation_capability",
                    "emergency_stop_active",
                    "low_level_control_rejected",
                ],
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
                llm_hints={
                    "when_to_use": (
                        "Use for structured embodied goals that should remain "
                        "Soridormi-owned instead of becoming velocity recipes."
                    ),
                    "when_not_to_use": (
                        "Do not use for immediate stop/emergency behavior; call "
                        "soridormi.motion.stop, soridormi.motion.cancel, or "
                        "soridormi.safety.emergency_stop."
                    ),
                    "no_motion_until": "task_execution_state_machine",
                    "forbidden_fields": [
                        "action_14d",
                        "joint_targets",
                        "motor_commands",
                        "torque_commands",
                        "actuator_ctrl",
                    ],
                    "plan_step_boundary": (
                        "plan_steps describe Soridormi-owned embodied planning "
                        "layers and never contain raw low-level robot control."
                    ),
                    "next_action_boundary": (
                        "recommended_next_actions are routing hints for Chromie; "
                        "they are not execution receipts or low-level commands."
                    ),
                },
            ),
            ToolCapability(
                name="soridormi.task.status",
                agent_id="soridormi.task",
                description="Read a previously submitted Soridormi embodied task status.",
                input_schema=_object_schema(
                    {
                        "task_id": {"type": "string", "minLength": 1},
                        "client_task_ref": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                    },
                ),
                output_schema=task_status_payload_schema,
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
            ),
            ToolCapability(
                name="soridormi.task.events",
                agent_id="soridormi.task",
                description="Read structured lifecycle events for a Soridormi embodied task.",
                input_schema=_object_schema(
                    {
                        "task_id": {"type": "string", "minLength": 1},
                        "client_task_ref": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "after_sequence": {"type": "integer", "minimum": 0},
                    }
                ),
                output_schema=_object_schema(
                    {
                        "schema_version": {"type": "string"},
                        "task_id": {"type": "string"},
                        "client_task_ref": {"type": ["string", "null"]},
                        "status": {"type": "string"},
                        "phase": {"type": "string"},
                        "terminal": {"type": "boolean"},
                        "safe_idle": {"type": "boolean"},
                        "deadline_at": {"type": "number"},
                        "expired": {"type": "boolean"},
                        "timeout_elapsed_s": {"type": ["number", "null"]},
                        "events": {"type": "array", "items": {"type": "object"}},
                        "returned_count": {"type": "integer"},
                        "latest_sequence": {"type": "integer"},
                        "next_after_sequence": {"type": "integer"},
                        "has_more": {"type": "boolean"},
                        "poll_recommendation": {"type": "object"},
                    },
                    required=[
                        "schema_version",
                        "task_id",
                        "client_task_ref",
                        "status",
                        "phase",
                        "terminal",
                        "safe_idle",
                        "deadline_at",
                        "expired",
                        "events",
                        "returned_count",
                        "latest_sequence",
                        "next_after_sequence",
                        "has_more",
                        "poll_recommendation",
                    ],
                ),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=True,
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=True,
                ),
            ),
            ToolCapability(
                name="soridormi.task.cancel",
                agent_id="soridormi.task",
                description="Cancel a non-terminal Soridormi embodied task request.",
                input_schema=_object_schema(
                    {
                        "task_id": {"type": "string", "minLength": 1},
                        "client_task_ref": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "reason": {"type": "string"},
                    },
                ),
                output_schema=_object_schema(
                    {
                        "task_id": {"type": "string"},
                        "client_task_ref": {"type": ["string", "null"]},
                        "cancelled": {"type": "boolean"},
                        "status": {"type": "string"},
                        "phase": {"type": "string", "enum": list(TASK_PHASES)},
                        "terminal": {"type": "boolean"},
                        "safe_idle": {"type": "boolean"},
                        "reason_code": {"type": ["string", "null"]},
                    },
                    required=["task_id", "cancelled", "status", "phase", "terminal", "safe_idle"],
                ),
                effects=["safety_control", "task_lifecycle"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=safe_modes),
                execution=ExecutionPolicy(
                    can_run_parallel=False,
                    exclusive_group="soridormi.robot_motion",
                    timeout_s=1.0,
                    idempotent=True,
                    side_effect_free=False,
                ),
                confirmation=ConfirmationPolicy(required=False),
                default_failure_policy=FailurePolicy(strategy="stop_and_report"),
            ),
        ],
    )

    scenario_ids = [
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
    ]
    testing_agent = AgentManifest(
        agent_id="soridormi.testing",
        display_name="Soridormi Test Control Agent",
        description="Test-only provider fault injection; never model visible.",
        llm_visible=False,
        transport=TransportSpec(
            kind="local_cli",
            command="python",
            args=["-m", "soridormi_runtime.mcp.call_tool"],
        ),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "testing", "fault-injection"],
        tools=[
            ToolCapability(
                name="soridormi.testing.configure_fault",
                agent_id="soridormi.testing",
                description="Configure one deterministic provider fault.",
                llm_visible=False,
                input_schema=_object_schema(
                    {"scenario_id": {"type": "string", "enum": scenario_ids}},
                    required=["scenario_id"],
                ),
                output_schema=_object_schema(
                    {
                        "configured": {"type": "boolean"},
                        "scenario_id": {"type": "string"},
                    },
                    required=["configured", "scenario_id"],
                ),
                effects=["test_control"],
                safety_class="restricted",
                availability=ToolAvailability(modes=safe_modes),
            ),
            ToolCapability(
                name="soridormi.testing.clear_faults",
                agent_id="soridormi.testing",
                description="Clear all configured provider faults.",
                llm_visible=False,
                input_schema=_object_schema({}),
                output_schema=_object_schema(
                    {"cleared": {"type": "boolean"}},
                    required=["cleared"],
                ),
                effects=["test_control"],
                safety_class="restricted",
                availability=ToolAvailability(modes=safe_modes),
            ),
        ],
    )

    return CapabilityBundle(
        source="soridormi",
        agents=[
            robot_agent,
            motion_agent,
            skill_agent,
            activity_agent,
            task_agent,
            safety_agent,
            testing_agent,
        ],
        dag_contract=build_soridormi_dag_contract(mode=mode),
        metadata={
            "provider_readiness": {
                "safe_modes": safe_modes,
                "fault_injection": {
                    "configure_tool": "soridormi.testing.configure_fault",
                    "clear_tool": "soridormi.testing.clear_faults",
                    "supported_scenarios": scenario_ids,
                },
            }
        },
    )
