from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .dag_contract import build_soridormi_dag_contract

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


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def build_soridormi_capability_bundle(*, mode: str = "sim") -> CapabilityBundle:
    """Return Soridormi's robot-body capability manifest.

    Chromie owns the global registry. Soridormi only declares the capabilities of
    the robot runtime boundary it owns. There are intentionally no
    `chromie.speak` or user-confirmation tools here.
    """

    robot_agent = AgentManifest(
        agent_id="soridormi.robot",
        display_name="Soridormi Robot State Agent",
        description="Read-only Soridormi robot status, mode, and battery state.",
        transport=TransportSpec(kind="stdio", command="python", args=["-m", "soridormi_runtime.mcp.server"]),
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
                }),
                effects=["read_only"],
                safety_class="safe_read",
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
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
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
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
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"], reason="Sim may return null battery percent."),
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
        transport=TransportSpec(kind="stdio", command="python", args=["-m", "soridormi_runtime.mcp.server"]),
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
                availability=ToolAvailability(modes=["sim", "hardware_dry_run"]),
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
                availability=ToolAvailability(modes=["sim", "hardware_dry_run"], requires=["robot_standing", "not_emergency_stopped"]),
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
                output_schema=_object_schema({"stopped": {"type": "boolean"}}),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
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
                output_schema=_object_schema({"cancelled": {"type": "boolean"}}),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
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
        transport=TransportSpec(kind="stdio", command="python", args=["-m", "soridormi_runtime.mcp.server"]),
        status=AgentStatus(available=True, details={"mode": mode}),
        tags=["soridormi", "safety"],
        tools=[
            ToolCapability(
                name="soridormi.safety.monitor_motion",
                agent_id="soridormi.safety",
                description="Monitor safety state while a Soridormi physical-motion node is running.",
                input_schema=_object_schema({"during_node_id": {"type": "string"}}),
                output_schema=_object_schema({"event": {"type": ["string", "null"]}, "ok": {"type": "boolean"}}),
                effects=["read_only", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
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
                output_schema=_object_schema({"stopped": {"type": "boolean"}}),
                effects=["physical_motion", "safety_control"],
                safety_class="safety_critical",
                availability=ToolAvailability(modes=["sim", "hardware_dry_run", "hardware"]),
                execution=ExecutionPolicy(can_run_parallel=False, exclusive_group="soridormi.robot_motion", timeout_s=1.0, idempotent=True, side_effect_free=False),
                confirmation=ConfirmationPolicy(required=False),
                failure_modes=["runtime_unreachable"],
                default_failure_policy=FailurePolicy(strategy="emergency_stop"),
                llm_hints={"when_to_use": "Use immediately for fall, collision risk, emergency stop, or user stop/cancel."},
            ),
        ],
    )

    return CapabilityBundle(
        source="soridormi",
        agents=[robot_agent, motion_agent, safety_agent],
        dag_contract=build_soridormi_dag_contract(mode=mode),
    )
