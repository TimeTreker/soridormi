from __future__ import annotations

import asyncio
import threading

import pytest

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
from soridormi_runtime.scripted_head_skill import HEAD_JOINT_NAMES
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




class ThreadAffineRobot(FakeRobot):
    def __init__(self) -> None:
        super().__init__()
        self.owner_thread_id = threading.get_ident()

    def _assert_owner_thread(self) -> None:
        assert threading.get_ident() == self.owner_thread_id

    def read_state(self) -> RobotState:
        self._assert_owner_thread()
        return super().read_state()

    def send_motor_command(self, command: MotorCommand) -> None:
        self._assert_owner_thread()
        super().send_motor_command(command)

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self._assert_owner_thread()
        self.send_motor_command(command)
        return self.read_state()


class StartupSyncRobot(ThreadAffineRobot):
    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0
        self.step_count = 0

    def reset(self) -> str:
        self._assert_owner_thread()
        self.reset_count += 1
        self.time = 0.0
        return "reset"

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self._assert_owner_thread()
        self.step_count += 1
        return super().step_motor_command(command)


class AdvancingReadSyncRobot(FakeRobot):
    """Model the simulator API: both get_state and step_command advance time."""

    def __init__(self, *, dt: float = 0.02) -> None:
        super().__init__()
        self.dt = dt
        self.read_count = 0
        self.step_count = 0

    def read_state(self) -> RobotState:
        self.read_count += 1
        self.time += self.dt
        return super().read_state()

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        self.step_count += 1
        self.commands.append(command)
        self.time += self.dt
        return super().read_state()


class HeadFakeRobot:
    def __init__(self) -> None:
        self.time = 0.0
        self.commands: list[MotorCommand] = []
        self.names = [*HEAD_JOINT_NAMES, "left_knee"]
        self.positions = [0.0] * len(self.names)

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


class FakeController:
    def __init__(self) -> None:
        self.command = PolicyCommand()
        self.seen_commands: list[PolicyCommand] = []
        self.seen_state_times: list[float] = []

    def compute(self, state: RobotState) -> MotorCommand:
        self.seen_commands.append(self.command)
        self.seen_state_times.append(float(state.time))
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
        assert stopped["safe_idle"] is True
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
        assert emergency["safe_idle"] is False
        assert result["completed"] is False
        assert status["emergency_stop"] is True
        assert status["safe_idle"] is False
        with pytest.raises(RuntimeError, match="emergency_stop"):
            await service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )

    asyncio.run(exercise())


def test_runtime_task_api_completes_skill_dry_run_without_motion() -> None:
    async def exercise() -> None:
        service = _service()

        submitted = await service.call_tool(
            "soridormi.task.submit",
            {
                "task_type": "perform_gesture",
                "summary": "nod twice",
                "parameters": {"gesture": "nod_yes", "count": 2},
            },
        )
        status = await service.call_tool(
            "soridormi.task.status",
            {"task_id": submitted["task_id"]},
        )
        events = await service.call_tool(
            "soridormi.task.events",
            {"task_id": submitted["task_id"]},
        )

        assert submitted["accepted"] is True
        assert submitted["status"] == "completed"
        assert submitted["phase"] == "completed"
        assert submitted["terminal"] is True
        assert submitted["execution_mode"] == "skill_dry_run"
        assert submitted["no_motion"] is True
        assert submitted["skill_id"] == "nod_yes"
        assert status["task_type"] == "perform_gesture"
        assert status["phase"] == "completed"
        assert events["schema_version"] == "soridormi.task_events.v1"
        assert events["terminal"] is True
        assert events["safe_idle"] is True
        assert events["latest_sequence"] == events["next_after_sequence"]
        assert events["poll_recommendation"]["action"] == "stop_polling"
        assert service.active_task is None

    asyncio.run(exercise())


def test_runtime_task_preview_does_not_create_status_record() -> None:
    async def exercise() -> None:
        service = _service()

        preview = await service.call_tool(
            "soridormi.task.preview",
            {
                "task_type": "navigate_to_location",
                "summary": "walk forward to the house",
                "parameters": {"target_label": "house"},
            },
        )

        assert preview["preview_id"].startswith("soridormi-preview-")
        assert preview["persistent"] is False
        assert preview["reason_code"] == "missing_navigation_pipeline"
        assert preview["plan_steps"]
        assert preview["task_graph"]["schema_version"] == "soridormi.task_graph.v1"
        assert preview["task_graph"]["task_ref"] == preview["preview_id"]
        assert preview["task_graph"]["raw_control_allowed"] is False
        with pytest.raises(KeyError, match="task not found"):
            await service.call_tool(
                "soridormi.task.status",
                {"task_id": preview["preview_id"]},
            )

    asyncio.run(exercise())


def test_runtime_task_get_capabilities_reports_readiness() -> None:
    async def exercise() -> None:
        service = _service()

        payload = await service.call_tool("soridormi.task.get_capabilities", {})
        by_type = {
            task["task_type"]: task
            for task in payload["task_types"]
        }

        assert payload["schema_version"] == "soridormi.task_capabilities.v1"
        assert payload["mode"] == "sim"
        assert payload["task_api_no_motion"] is True
        assert "nod_yes" in payload["executable_skill_ids"]
        assert by_type["perform_gesture"]["readiness"] == "skill_dry_run_ready"
        assert by_type["deliver_object"]["readiness"] == "future_blocked"
        assert "manipulation_capability" in by_type["deliver_object"]["missing_subsystems"]

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




def test_runtime_from_env_keeps_robot_on_one_worker_thread() -> None:
    service = SoridormiRuntimeToolService.from_env(
        mode="sim",
        robot_factory=ThreadAffineRobot,
        controller_factory=FakeController,
    )

    async def exercise() -> None:
        status = await service.call_tool("soridormi.robot.get_status", {})
        assert status["mode"] == "sim"

        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {
                "commands": [
                    {
                        "vx": 0.05,
                        "vy": 0.0,
                        "yaw": 0.0,
                        "duration_s": 0.05,
                    }
                ]
            },
        )
        result = await service.call_tool(
            "soridormi.motion.execute_plan",
            {"plan_id": plan["plan_id"]},
        )
        assert result["completed"] is True

    try:
        asyncio.run(exercise())
    finally:
        assert service._robot_executor is not None
        service._robot_executor.shutdown(wait=True)


def test_runtime_from_env_applies_profile_reset_and_sync_preroll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SORIDORMI_RESET_AT_START", "1")
    monkeypatch.setenv("SORIDORMI_SIM_SYNC_STEP", "1")
    monkeypatch.setenv("SORIDORMI_SIM_PREROLL_STEPS", "2")

    robots: list[StartupSyncRobot] = []

    def robot_factory() -> StartupSyncRobot:
        robot = StartupSyncRobot()
        robots.append(robot)
        return robot

    service = SoridormiRuntimeToolService.from_env(
        mode="sim",
        robot_factory=robot_factory,
        controller_factory=FakeController,
    )

    try:
        robot = robots[0]
        assert robot.reset_count == 1
        assert robot.step_count == 2
        assert len(robot.commands) == 2
        assert service._last_state is not None
        assert service._last_state.time == pytest.approx(0.02)
    finally:
        assert service._robot_executor is not None
        service._robot_executor.shutdown(wait=True)


def test_sync_step_controller_reuses_previous_step_state() -> None:
    async def exercise() -> None:
        robot = AdvancingReadSyncRobot()
        controller = FakeController()
        service = SoridormiRuntimeToolService(
            robot=robot,
            controller=controller,
            control_hz=50.0,
        )
        service._last_state = robot.read_state()

        await service._step_controller()
        await service._step_controller()

        assert robot.read_count == 1
        assert robot.step_count == 2
        assert robot.time == pytest.approx(0.06)
        assert controller.seen_state_times == pytest.approx([0.02, 0.04])

    asyncio.run(exercise())


def test_runtime_service_lists_velocity_and_scripted_head_skills() -> None:
    async def exercise() -> None:
        service = _service()
        catalog = await service.call_tool("soridormi.skill.list", {})
        skills = {item["skill_id"]: item for item in catalog["skills"]}

        assert "walk_velocity" in skills
        assert skills["walk_velocity"]["available"] is True
        assert skills["walk_velocity"]["description"]
        assert "vx_mps" in skills["walk_velocity"]["parameters_schema"]["properties"]
        assert "nod_yes" in skills
        assert skills["nod_yes"]["available"] is True
        assert skills["nod_yes"]["execution"] == "scripted_keyframe"
        assert "duration_s" in skills["nod_yes"]["parameters_schema"]["properties"]

    asyncio.run(exercise())


def test_runtime_service_executes_named_velocity_skill() -> None:
    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "walk_velocity",
                "parameters": {
                    "vx_mps": 0.15,
                    "vy_mps": 0.0,
                    "yaw_radps": 0.0,
                    "duration_s": 0.5,
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["skill_id"] == "walk_velocity"
        assert plan["no_motion"] is False
        assert result["completed"] is True
        assert result["skill_id"] == "walk_velocity"
        assert result["no_motion"] is False
        assert any(
            command.x_velocity == 0.15
            for command in service.controller.seen_commands
        )

    asyncio.run(exercise())


def test_runtime_service_executes_named_scripted_head_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "soridormi_runtime.mcp.runtime_tools.asyncio.sleep",
        no_sleep,
    )

    async def exercise() -> None:
        robot = HeadFakeRobot()
        service = SoridormiRuntimeToolService(
            robot=robot,
            controller=FakeController(),
            control_hz=50.0,
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "nod_yes",
                "parameters": {
                    "count": 2,
                    "amplitude": "small",
                    "duration_s": 1.0,
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        head_pitch_index = robot.names.index("head_pitch")
        commanded_pitch = [
            command.positions[head_pitch_index] for command in robot.commands
        ]
        assert result["completed"] is True
        assert result["skill_id"] == "nod_yes"
        assert result["no_motion"] is False
        assert min(commanded_pitch) < -0.10
        assert max(commanded_pitch) > 0.05
        assert commanded_pitch[-1] == pytest.approx(0.0)

    asyncio.run(exercise())


def test_runtime_service_rejects_unsupported_named_skill() -> None:
    async def exercise() -> None:
        service = _service()
        with pytest.raises(ValueError, match="not supported by the runtime adapter"):
            await service.call_tool(
                "soridormi.skill.create_plan",
                {"skill_id": "wave_hand", "parameters": {}},
            )

    asyncio.run(exercise())
