from __future__ import annotations

import asyncio

import pytest

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
from soridormi_runtime.policy_command import PolicyCommand


class FakeRobot:
    def __init__(self) -> None:
        self.time = 0.0
        self.commands: list[MotorCommand] = []

    def read_state(self) -> RobotState:
        return RobotState(
            time=self.time,
            joints=JointState(
                names=["joint"],
                positions=[0.0],
                velocities=[0.0],
                torques=[0.0],
            ),
            imu=IMUState(),
        )

    def send_motor_command(self, command: MotorCommand) -> None:
        self.commands.append(command)
        self.time += 0.01


class FakeController:
    def __init__(self) -> None:
        self.command = PolicyCommand()
        self.seen_commands: list[PolicyCommand] = []

    def compute(self, state: RobotState) -> MotorCommand:
        self.seen_commands.append(self.command)
        return MotorCommand(
            names=state.joints.names,
            positions=state.joints.positions,
            velocities=[self.command.x_velocity],
            kp=[5.0],
            kd=[0.1],
            torques=[0.0],
        )


def _service(*, control_hz: float = 200.0) -> SoridormiRuntimeToolService:
    return SoridormiRuntimeToolService(
        robot=FakeRobot(),
        controller=FakeController(),
        control_hz=control_hz,
    )


def test_runtime_service_executes_bounded_plan_through_controller() -> None:
    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 0.08, "vy": 0.0, "yaw": 0.0, "duration_s": 0.05}]},
        )

        result = await service.call_tool(
            "soridormi.motion.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["dry_run_only"] is False
        assert result["completed"] is True
        assert result["dry_run_only"] is False
        assert service.active_task is None
        assert service.controller.command == PolicyCommand()
        assert any(command.x_velocity == 0.08 for command in service.controller.seen_commands)

    asyncio.run(exercise())


def test_runtime_stop_preempts_long_running_plan() -> None:
    async def exercise() -> None:
        service = _service(control_hz=100.0)
        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 0.05, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}]},
        )
        execution = asyncio.create_task(
            service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        )
        while service.active_task is None:
            await asyncio.sleep(0)

        stopped = await service.call_tool("soridormi.motion.stop", {})
        result = await asyncio.wait_for(execution, timeout=1.0)

        assert stopped["stopped"] is True
        assert result["completed"] is False
        assert result["stopped"] is True
        assert service.active_task is None

    asyncio.run(exercise())


def test_runtime_emergency_stop_preempts_and_persists() -> None:
    async def exercise() -> None:
        service = _service(control_hz=100.0)
        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 0.05, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}]},
        )
        execution = asyncio.create_task(
            service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        )
        while service.active_task is None:
            await asyncio.sleep(0)

        emergency = await service.call_tool(
            "soridormi.safety.emergency_stop",
            {"reason": "test"},
        )
        result = await asyncio.wait_for(execution, timeout=1.0)
        status = await service.call_tool("soridormi.robot.get_status", {})

        assert emergency["stopped"] is True
        assert result["completed"] is False
        assert status["emergency_stop"] is True
        with pytest.raises(RuntimeError, match="emergency_stop"):
            await service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )

    asyncio.run(exercise())


def test_runtime_request_cancellation_applies_safe_hold() -> None:
    async def exercise() -> None:
        service = _service(control_hz=100.0)
        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 0.05, "vy": 0.0, "yaw": 0.0, "duration_s": 1.0}]},
        )
        execution = asyncio.create_task(
            service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        )
        while service.active_task is None:
            await asyncio.sleep(0)

        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution

        assert service.active_task is None
        assert service.controller.command == PolicyCommand()
        assert service.robot.commands

    asyncio.run(exercise())


def test_runtime_service_rejects_non_policy_controller() -> None:
    with pytest.raises(ValueError, match="onnx_policy"):
        SoridormiRuntimeToolService.from_env(
            mode="sim",
            robot_factory=FakeRobot,
            controller_factory=lambda: object(),  # type: ignore[arg-type]
        )


def test_runtime_service_rejects_hardware_until_backend_exists() -> None:
    with pytest.raises(ValueError, match="HardwareRobot"):
        SoridormiRuntimeToolService.from_env(
            mode="hardware_dry_run",
            robot_factory=FakeRobot,
            controller_factory=FakeController,
        )
