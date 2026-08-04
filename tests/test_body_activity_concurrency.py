from __future__ import annotations

import asyncio

import pytest

from soridormi_api import (
    IMUState,
    JointState,
    MotorCommand,
    RobotState,
    VisualExpressionCommand,
)
from soridormi_runtime.mcp.body_activity import (
    body_activity_capabilities_payload,
    create_body_activity_plan,
)
from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scripted_head_skill import HEAD_JOINT_NAMES
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST


class ConcurrentRobot:
    def __init__(self) -> None:
        self.time = 0.0
        self.names = [*HEAD_JOINT_NAMES, "left_knee", "right_knee"]
        self.positions = [0.0] * len(self.names)
        self.commands: list[MotorCommand] = []
        self.visual_expressions: list[VisualExpressionCommand] = []

    def read_state(self) -> RobotState:
        return RobotState(
            time=self.time,
            joints=JointState(
                names=list(self.names),
                positions=list(self.positions),
                velocities=[0.0] * len(self.names),
                torques=[0.0] * len(self.names),
            ),
            imu=IMUState(),
            base_position_xyz=[0.0, 0.0, 0.3],
            actuator_ctrl=list(self.positions),
        )

    def send_motor_command(self, command: MotorCommand) -> None:
        self.commands.append(command)
        self.positions = list(command.positions)
        self.time += 0.01

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self.send_motor_command(command)
        return self.read_state()

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        self.visual_expressions.append(command)
        self.time += 0.001
        return command.expression


class ConcurrentController:
    def __init__(self) -> None:
        self.command = PolicyCommand()
        self.seen_commands: list[PolicyCommand] = []

    def compute(self, state: RobotState) -> MotorCommand:
        self.seen_commands.append(self.command)
        count = len(state.joints.names)
        return MotorCommand(
            names=list(state.joints.names),
            positions=list(state.joints.positions),
            velocities=[0.0] * count,
            kp=[5.0] * count,
            kd=[0.1] * count,
            torques=[0.0] * count,
        )


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry.from_manifest_path(DEFAULT_SKILL_MANIFEST)


def _concurrent_members(*, walk_duration_s: float = 0.5) -> list[dict[str, object]]:
    return [
        {
            "member_id": "walk",
            "skill_id": "walk_velocity",
            "parameters": {
                "vx_mps": 0.12,
                "vy_mps": 0.0,
                "yaw_radps": 0.0,
                "duration_s": walk_duration_s,
            },
        },
        {
            "member_id": "gaze",
            "skill_id": "look_at_person",
            "parameters": {
                "target_ref": "person_1",
                "target_yaw_rad": 0.08,
                "target_pitch_rad": -0.04,
                "duration_s": 0.4,
                "hold_fraction": 0.5,
                "end_mode": "hold_target",
            },
        },
        {
            "member_id": "blink",
            "skill_id": "blink_eyes",
            "parameters": {
                "count": 2,
                "closed_duration_s": 0.15,
                "open_duration_s": 0.15,
                "intensity": 1.0,
            },
            "optional": True,
        },
    ]


def test_body_activity_capabilities_preserve_chromie_speech_boundary() -> None:
    payload = body_activity_capabilities_payload(mode="sim", backend="runtime")

    assert payload["speech_owner"] == "chromie"
    assert payload["speech_is_external_peer_lane"] is True
    assert payload["concurrency_rules"]["max_primary_locomotion_members"] == 1
    assert payload["concurrency_rules"]["one_writer_per_resource"] is True
    assert payload["concurrency_rules"]["emergency_stop_preempts_every_body_member"] is True


def test_activity_plan_accepts_walk_gaze_and_blink() -> None:
    record = create_body_activity_plan(
        _registry(),
        {
            "coordination_id": "interaction-1",
            "members": _concurrent_members(),
        },
    )

    assert record.coordination_id == "interaction-1"
    assert record.primary_member is not None
    assert record.primary_member.skill_id == "walk_velocity"
    assert record.head_overlay_member is not None
    assert record.head_overlay_member.skill_id == "look_at_person"
    assert [member.skill_id for member in record.independent_members] == ["blink_eyes"]
    assert {
        resource
        for member in record.members
        for resource in member.write_resources
    } == {"body.primary_motion", "body.head_pose", "visual.eyes"}


def test_activity_plan_rejects_two_primary_locomotion_members() -> None:
    with pytest.raises(ValueError, match="body.primary_motion|one primary"):
        create_body_activity_plan(
            _registry(),
            {
                "members": [
                    {
                        "member_id": "walk",
                        "skill_id": "walk_velocity",
                        "parameters": {"duration_s": 0.5},
                    },
                    {
                        "member_id": "turn",
                        "skill_id": "turn_in_place",
                        "parameters": {"duration_s": 0.5},
                    },
                ]
            },
        )


def test_activity_plan_rejects_standalone_gesture_during_locomotion() -> None:
    with pytest.raises(ValueError, match="body.primary_motion|standalone"):
        create_body_activity_plan(
            _registry(),
            {
                "members": [
                    {
                        "member_id": "walk",
                        "skill_id": "walk_velocity",
                        "parameters": {"duration_s": 0.5},
                    },
                    {
                        "member_id": "bow",
                        "skill_id": "bow",
                        "parameters": {"duration_s": 2.0},
                    },
                ]
            },
        )


def test_activity_plan_rejects_head_overlay_outside_locomotion_envelope() -> None:
    members = _concurrent_members()
    members[1]["parameters"] = {
        "target_ref": "person_1",
        "target_yaw_rad": 0.30,
        "target_pitch_rad": 0.0,
        "duration_s": 0.4,
    }

    with pytest.raises(ValueError, match="head yaw.*exceeds"):
        create_body_activity_plan(_registry(), {"members": members})


def test_local_activity_service_is_contract_only() -> None:
    service = SoridormiLocalToolService()
    plan = service.call_tool(
        "soridormi.activity.create_plan",
        {"coordination_id": "coord-1", "members": _concurrent_members()},
    )
    result = service.call_tool(
        "soridormi.activity.execute_plan",
        {"plan_id": plan["plan_id"]},
    )

    assert plan["status"] == "planned"
    assert result["completed"] is True
    assert result["no_motion"] is True
    assert result["dry_run_only"] is True
    assert result["speech_owner"] == "chromie"


def test_runtime_executes_walk_gaze_and_blink_concurrently() -> None:
    async def exercise() -> None:
        robot = ConcurrentRobot()
        controller = ConcurrentController()
        service = SoridormiRuntimeToolService(
            robot=robot,
            controller=controller,
            control_hz=100.0,
        )
        plan = await service.call_tool(
            "soridormi.activity.create_plan",
            {
                "coordination_id": "song-walk-1",
                "members": _concurrent_members(),
            },
        )
        execution = asyncio.create_task(
            service.call_tool(
                "soridormi.activity.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        )
        for _ in range(1000):
            if {"locomotion", "head_overlay", "visual:visual.eyes"}.issubset(
                service.active_lanes
            ):
                break
            await asyncio.sleep(0.001)
        assert {"locomotion", "head_overlay", "visual:visual.eyes"}.issubset(
            service.active_lanes
        )
        status = await service.call_tool("soridormi.robot.get_status", {})
        assert status["safe_idle"] is False
        assert status["activity_idle"] is False
        assert status["active_task"]["kind"] == "concurrent_body_activity"

        result = await asyncio.wait_for(execution, timeout=3.0)

        assert result["completed"] is True
        assert result["status"] == "completed"
        assert result["coordination_id"] == "song-walk-1"
        assert result["one_final_motor_command_authority"] is True
        assert result["member_results"]["gaze"][
            "composed_into_final_motor_command"
        ] is True
        assert any(command.x_velocity > 0.0 for command in controller.seen_commands)
        head_yaw_index = robot.names.index("head_yaw")
        assert any(abs(command.positions[head_yaw_index]) > 0.01 for command in robot.commands)
        assert any(item.expression == "eyes_closed" for item in robot.visual_expressions)
        assert robot.visual_expressions[-1].expression == "eyes_open"
        assert service.active_lanes == {}
        final_status = await service.call_tool("soridormi.robot.get_status", {})
        assert final_status["safe_idle"] is True
        assert final_status["activity_idle"] is True

    asyncio.run(exercise())


def test_runtime_activity_cancel_preempts_physical_and_visual_members() -> None:
    async def exercise() -> None:
        robot = ConcurrentRobot()
        service = SoridormiRuntimeToolService(
            robot=robot,
            controller=ConcurrentController(),
            control_hz=100.0,
        )
        members = _concurrent_members(walk_duration_s=1.0)
        members[2]["parameters"] = {
            "count": 6,
            "closed_duration_s": 0.2,
            "open_duration_s": 0.2,
            "intensity": 1.0,
        }
        plan = await service.call_tool(
            "soridormi.activity.create_plan",
            {"members": members},
        )
        execution = asyncio.create_task(
            service.call_tool(
                "soridormi.activity.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        )
        while "locomotion" not in service.active_lanes:
            await asyncio.sleep(0)
        cancelled = await service.call_tool(
            "soridormi.activity.cancel",
            {"plan_id": plan["plan_id"], "reason": "user changed request"},
        )
        result = await asyncio.wait_for(execution, timeout=3.0)

        assert cancelled["cancelled"] is True
        assert result["cancelled"] is True
        assert result["status"] == "cancelled"
        assert robot.visual_expressions[-1].expression == "eyes_open"
        assert service.active_lanes == {}

    asyncio.run(exercise())
