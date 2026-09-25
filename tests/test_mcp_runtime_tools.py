from __future__ import annotations

import asyncio
import threading

import pytest

from soridormi_api import (
    IMUState,
    JointState,
    MotorCommand,
    RobotState,
    VisualArmPoseCommand,
    VisualExpressionCommand,
)
from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scripted_head_skill import HEAD_JOINT_NAMES
from soridormi_runtime.skill_execution import MIN_FORWARD_WALK_SPEED_MPS


class FakeRobot:
    def __init__(self) -> None:
        self.time = 0.0
        self.commands: list[MotorCommand] = []
        self.visual_expressions: list[VisualExpressionCommand] = []
        self.visual_arm_poses: list[VisualArmPoseCommand] = []

    def read_state(self) -> RobotState:
        return RobotState(
            time=self.time,
            reset_count=0,
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

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        self.visual_expressions.append(command)
        self.time += 0.001
        return f"visual expression applied: {command.expression}"

    def set_visual_arm_pose(self, command: VisualArmPoseCommand) -> str:
        self.visual_arm_poses.append(command)
        self.time += 0.001
        return f"visual arm pose applied: {command.pose}"


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
            reset_count=0,
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


class ResourceFakeRobot:
    def __init__(self) -> None:
        self.time = 0.0
        self.commands: list[MotorCommand] = []
        self.visual_arm_poses: list[VisualArmPoseCommand] = []
        self.names = [
            "left_ankle",
            "left_hip_pitch",
            "left_knee",
            "right_ankle",
            "right_hip_pitch",
            "right_knee",
        ]
        self.positions = [-0.25, -0.25, 0.50, 0.25, 0.25, -0.50]
        self.initial_positions = list(self.positions)

    def read_state(self) -> RobotState:
        return RobotState(
            time=self.time,
            reset_count=0,
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

    def set_visual_arm_pose(self, command: VisualArmPoseCommand) -> str:
        self.visual_arm_poses.append(command)
        return f"visual arm pose applied: {command.pose}"


class ResettingResourceFakeRobot(ResourceFakeRobot):
    def __init__(self, *, reset_after_commands: int) -> None:
        super().__init__()
        self.reset_count = 0
        self.reset_after_commands = reset_after_commands

    def read_state(self) -> RobotState:
        return super().read_state().model_copy(update={"reset_count": self.reset_count})

    def send_motor_command(self, command: MotorCommand) -> None:
        super().send_motor_command(command)
        if len(self.commands) == self.reset_after_commands:
            self.reset_count += 1
            self.time = 0.0

    def observe_scene(self) -> dict[str, object]:
        return {
            "mocked_simulation": True,
            "objects": [{
                "description": "bottle of milk", "distance_m": 2.0, "bearing_rad": 0.0,
            }, {
                "description": "user", "distance_m": 1.5, "bearing_rad": 0.0,
            }],
        }


class ObservedResourceFakeRobot(ResourceFakeRobot):
    def __init__(
        self, *, description: str = "bottle of milk", user_description: str = "user"
    ) -> None:
        super().__init__()
        self.description = description
        self.user_description = user_description
        self.distance_m = 1.25
        self.user_distance_m = 1.45

    def send_motor_command(self, command: MotorCommand) -> None:
        super().send_motor_command(command)
        if command.velocities and command.velocities[0] > 0:
            if self.distance_m > 0.9:
                self.distance_m -= 0.02
            else:
                self.user_distance_m -= 0.02

    def observe_scene(self) -> dict[str, object]:
        return {
            "mocked_simulation": True,
            "objects": [{
                "description": self.description,
                "distance_m": self.distance_m,
                "bearing_rad": 0.0,
            }, {
                "description": self.user_description,
                "distance_m": self.user_distance_m,
                "bearing_rad": 0.0,
            }],
        }


class MultipleWaterBottlesFakeRobot(ObservedResourceFakeRobot):
    def __init__(self) -> None:
        super().__init__(description="bottle of water")
        self.other_distance_m = 1.4

    def send_motor_command(self, command: MotorCommand) -> None:
        super().send_motor_command(command)
        if command.velocities and command.velocities[0] > 0:
            self.other_distance_m = 0.75

    def observe_scene(self) -> dict[str, object]:
        scene = super().observe_scene()
        objects = scene["objects"]
        assert isinstance(objects, list)
        objects[0]["object_ref"] = "water-nearest"
        objects[0]["bearing_rad"] = 0.2
        objects.append({
            "object_ref": "water-other", "description": "bottle of water",
            "distance_m": self.other_distance_m, "bearing_rad": 0.2,
        })
        objects.append({
            "object_ref": "water-third", "description": "bottle of water",
            "distance_m": 1.5, "bearing_rad": 0.2,
        })
        return scene


class FakeController:
    def __init__(self) -> None:
        self.command = PolicyCommand()
        self.seen_commands: list[PolicyCommand] = []
        self.seen_state_times: list[float] = []

    def compute(self, state: RobotState) -> MotorCommand:
        self.seen_commands.append(self.command)
        self.seen_state_times.append(float(state.time))
        n = len(state.joints.names)
        return MotorCommand(
            names=list(state.joints.names),
            positions=list(state.joints.positions),
            velocities=[self.command.x_velocity] * n,
            kp=[5.0] * n,
            kd=[0.1] * n,
            torques=[0.0] * n,
        )


def _service(*, control_hz: float = 200.0) -> SoridormiRuntimeToolService:
    return SoridormiRuntimeToolService(
        robot=FakeRobot(),
        controller=FakeController(),
        control_hz=control_hz,
    )


def _resource_service(*, control_hz: float = 50.0) -> SoridormiRuntimeToolService:
    return SoridormiRuntimeToolService(
        robot=ResourceFakeRobot(),
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


def test_motion_refuses_simulator_without_reset_generation() -> None:
    class ResetlessRobot(FakeRobot):
        def read_state(self) -> RobotState:
            return super().read_state().model_copy(update={"reset_count": None})

    async def exercise() -> None:
        service = SoridormiRuntimeToolService(
            robot=ResetlessRobot(), controller=FakeController()
        )
        plan = await service.call_tool(
            "soridormi.motion.create_plan",
            {"commands": [{"vx": 0.12, "vy": 0.0, "yaw": 0.0, "duration_s": 0.05}]},
        )
        with pytest.raises(RuntimeError, match="reset generation unavailable"):
            await service.call_tool(
                "soridormi.motion.execute_plan", {"plan_id": plan["plan_id"]}
            )
        assert service.robot.commands == []

    asyncio.run(exercise())


def test_runtime_skill_plan_rejects_invalid_chromie_proposal_metadata() -> None:
    async def exercise() -> None:
        service = _service()
        with pytest.raises(ValueError, match="requires_runtime_validation"):
            await service.call_tool(
                "soridormi.skill.create_plan",
                {
                    "skill_id": "nod_yes",
                    "chromie_intent": {
                        "execution_mode": "proposed",
                        "execution_semantics": "proposal_from_chromie",
                        "requires_runtime_validation": False,
                    },
                },
            )

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
        by_type = {task["task_type"]: task for task in payload["task_types"]}

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


def test_runtime_service_lists_velocity_scripted_head_and_visual_skills() -> None:
    async def exercise() -> None:
        service = _service()
        catalog = await service.call_tool("soridormi.skill.list", {})
        skills = {item["skill_id"]: item for item in catalog["skills"]}

        assert "walk_velocity" in skills
        assert skills["walk_velocity"]["available"] is True
        assert skills["walk_velocity"]["description"]
        assert "vx_mps" in skills["walk_velocity"]["parameters_schema"]["properties"]
        assert "walk_forward" in skills
        assert skills["walk_forward"]["execution"] == "skill_wrapper"
        assert skills["walk_forward"]["semantic_speed_presets_mps"]["slow"] == pytest.approx(
            MIN_FORWARD_WALK_SPEED_MPS
        )
        assert skills["walk_forward"]["parameters_schema"]["properties"]["speed"]["enum"] == [
            "slow",
            "normal",
            "medium",
            "quick",
            "fast_limited",
        ]
        turn = skills["turn_in_place"]
        assert "yaw_radps" in turn["parameters_schema"]["properties"]
        assert turn["metadata"]["semantic_facade"]["input_schema"]["properties"][
            "direction"
        ]["enum"] == ["left", "right"]
        assert turn["metadata"]["semantic_facade"]["provider_realizations"][
            "yaw_radps"
        ]["positive_direction"] == "left"
        assert {"acquire_resource", "deliver_resource", "acquire_and_deliver_resource"} <= set(
            skills
        )
        resource_skill = skills["acquire_and_deliver_resource"]
        assert resource_skill["execution"] == "composite"
        assert resource_skill["timeout_s"] == 660.0
        assert resource_skill["parameters_schema"]["properties"]["speed"]["default"] == "normal"
        assert resource_skill["parameters_schema"]["properties"]["resource"]["type"] == "object"
        assert resource_skill["metadata"]["semantic_scope"]["resource_kinds"] == ["physical_object"]
        assert resource_skill["metadata"]["semantic_scope"]["delivery_modes"] == [
            "physical_handover"
        ]
        assert resource_skill["metadata"]["resource_contract"]["result_field"] == (
            "resource_outcome"
        )
        assert "nod_yes" in skills
        assert skills["nod_yes"]["available"] is True
        assert skills["nod_yes"]["execution"] == "scripted_keyframe"
        assert "duration_s" in skills["nod_yes"]["parameters_schema"]["properties"]
        assert "blink_eyes" in skills
        assert skills["blink_eyes"]["execution"] == "visual_expression"
        assert skills["blink_eyes"]["effects"] == ["visual_expression"]
        assert skills["blink_eyes"]["safety_class"] == "low_risk_action"
        assert skills["blink_eyes"]["metadata"]["behavior_domains"] == [
            "social_attention",
            "facial_expression",
        ]
        assert {"wave_hand", "celebrate", "hug_gesture"} <= set(skills)
        assert skills["wave_hand"]["execution"] == "visual_arm_gesture"
        assert skills["wave_hand"]["effects"] == ["visual_expression"]
        assert skills["wave_hand"]["resource_claims"] == ["visual.arms"]

    asyncio.run(exercise())


def test_runtime_acquire_resource_uses_pickup_pose_and_restores_body() -> None:
    async def exercise() -> None:
        service = _resource_service()
        robot = service.robot
        assert isinstance(robot, ResourceFakeRobot)
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_resource",
                "parameters": {
                    "resource": {
                        "kind": "physical_object",
                        "description": "a cup of water",
                    },
                    "source": {"status": "unknown"},
                },
            },
        )
        with pytest.raises(RuntimeError, match="provider skill execution"):
            await service.call_tool(
                "soridormi.motion.execute_plan",
                {"plan_id": plan["plan_id"]},
            )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        left_knee = robot.names.index("left_knee")
        right_knee = robot.names.index("right_knee")
        assert result["completed"] is True
        assert result["resource_outcome"]["resource_acquired"] is True
        assert result["visual_arm_pose"] == "hold"
        assert [pose.pose for pose in robot.visual_arm_poses] == [
            "reach",
            "hold",
            "hold",
        ]
        assert max(command.positions[left_knee] for command in robot.commands) > 0.70
        assert min(command.positions[right_knee] for command in robot.commands) < -0.70
        assert robot.positions == pytest.approx(robot.initial_positions)

    asyncio.run(exercise())


def test_runtime_service_executes_simulated_resource_acquisition_delivery() -> None:
    async def exercise() -> None:
        service = SoridormiRuntimeToolService(
            robot=ObservedResourceFakeRobot(), controller=FakeController(), control_hz=20.0
        )
        robot = service.robot
        assert isinstance(robot, ObservedResourceFakeRobot)
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {
                        "kind": "physical_object",
                        "description": "bottle of milk",
                        "quantity": "one",
                        "attributes": {},
                    },
                    "source": {
                        "status": "unknown",
                        "description": "",
                        "bindings": {},
                    },
                    "recipient": {
                        "description": "user",
                        "referent_id": None,
                    },
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["skill_id"] == "acquire_and_deliver_resource"
        assert plan["no_motion"] is False
        assert result["completed"] is True
        assert result["no_motion"] is False
        assert result["skill_id"] == "acquire_and_deliver_resource"
        assert result["resource_outcome"]["resource_acquired"] is True
        assert result["resource_outcome"]["resource_delivered"] is True
        assert result["resource_outcome"]["resource_description"] == "bottle of milk"
        assert result["resource_outcome"]["mocked_simulation"] is True
        assert result["visual_arm_pose"] == "rest"
        assert [pose.pose for pose in robot.visual_arm_poses] == [
            "reach",
            "hold",
            "place",
            "rest",
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("speed, expected_mps", [(None, 0.12), ("quick", 0.16)])
def test_known_resource_walks_to_observed_scene_object_before_mock_handover(
    speed: str | None, expected_mps: float
) -> None:
    async def exercise() -> None:
        robot = ObservedResourceFakeRobot()
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {"kind": "physical_object", "description": "bottle of milk"},
                    "source": {"status": "known", "description": "table ahead", "bindings": {
                        "distance": "50 meters", "location": "in front of Chromie"
                    }},
                    "recipient": {"description": "user"},
                    **({"speed": speed} if speed else {}),
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]}
        )

        assert plan["estimated_duration_s"] is None
        assert result["completed"] is True
        assert result["resource_outcome"]["mocked_simulation"] is True
        assert result["approach_outcome"]["observed_distance_start_m"] == 1.25
        assert result["approach_outcome"]["observed_distance_end_m"] <= 0.9
        assert 0.9 < result["return_outcome"]["observed_distance_start_m"] <= 1.45
        assert result["return_outcome"]["observed_distance_end_m"] <= 0.9
        assert any(
            command.x_velocity == pytest.approx(expected_mps)
            for command in service.controller.seen_commands
        )
        assert all(command.x_velocity >= 0 for command in service.controller.seen_commands)
        assert "returned to the observed recipient" in result["summary"]
        public_observation = await service.call_tool("soridormi.robot.observe_scene", {})
        assert "bearing_rad" not in public_observation["objects"][0]

    asyncio.run(exercise())


def test_equivalent_water_bottles_select_nearest_and_hold_object_reference() -> None:
    async def exercise() -> None:
        robot = MultipleWaterBottlesFakeRobot()
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        await service._read_state()
        result = await service._navigate_to_observed_resource(
            "water", speed_mps=0.12, stop_distance_m=0.9,
            select_nearest_equivalent=True,
        )

        assert result["observed_distance_start_m"] == 1.25
        assert result["observed_distance_end_m"] <= 0.9
        assert robot.distance_m <= 0.9
        assert robot.other_distance_m == 0.75
        assert all(
            command.yaw_velocity == 0
            for command in service.controller.seen_commands
            if command.x_velocity > 0
        )

    asyncio.run(exercise())


def test_water_request_completes_mock_delivery_with_multiple_bottles() -> None:
    async def exercise() -> None:
        robot = MultipleWaterBottlesFakeRobot()
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {"skill_id": "acquire_and_deliver_resource", "parameters": {
                "resource": {"kind": "physical_object", "description": "water"},
                "source": {"status": "unknown"},
                "recipient": {"description": "user"},
            }},
        )
        assert service.skill_plans[plan["plan_id"]].plan.parameters["speed"] == "normal"
        result = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]}
        )
        assert result["completed"] is True
        assert result["resource_outcome"]["resource_delivered"] is True
        assert result["resource_outcome"]["mocked_simulation"] is True
        assert result["approach_outcome"]["observed_distance_start_m"] == 1.25
        assert service.active_task is None

    asyncio.run(exercise())


@pytest.mark.parametrize("allow_recovery", [False, True])
def test_resource_route_recovers_pace_only_when_allowed(
    monkeypatch: pytest.MonkeyPatch, allow_recovery: bool
) -> None:
    import soridormi_runtime.mcp.runtime_tools as runtime_tools

    class StationaryResourceRobot(ObservedResourceFakeRobot):
        def send_motor_command(self, command: MotorCommand) -> None:
            ResourceFakeRobot.send_motor_command(self, command)

    async def exercise() -> None:
        monkeypatch.setattr(runtime_tools, "RESOURCE_PACE_RECOVERY_AFTER_S", 0.1)
        robot = StationaryResourceRobot()
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        await service._read_state()
        route = asyncio.create_task(service._navigate_to_observed_resource(
            "bottle of milk", speed_mps=0.12, stop_distance_m=0.9,
            allow_bounded_pace_recovery=allow_recovery,
        ))
        await asyncio.sleep(2.2)
        service._motion_stop_requested = True
        assert await route == {}
        observed_speeds = {
            command.x_velocity for command in service.controller.seen_commands
        }
        assert (0.18 in observed_speeds) is allow_recovery
        assert 0.12 in observed_speeds

    asyncio.run(exercise())


def test_different_matching_resource_types_remain_ambiguous() -> None:
    async def exercise() -> None:
        robot = MultipleWaterBottlesFakeRobot()
        original_observe = robot.observe_scene

        def ambiguous_observe() -> dict[str, object]:
            scene = original_observe()
            objects = scene["objects"]
            assert isinstance(objects, list)
            objects.append({
                "object_ref": "water-glass", "description": "glass of water",
                "distance_m": 1.0, "bearing_rad": 0.0,
            })
            return scene

        robot.observe_scene = ambiguous_observe  # type: ignore[method-assign]
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        with pytest.raises(RuntimeError, match="not uniquely observed"):
            await service._navigate_to_observed_resource(
                "water", speed_mps=0.12, stop_distance_m=0.9,
                select_nearest_equivalent=True,
            )
        assert not robot.commands or all(
            command.velocities[0] == 0 for command in robot.commands
        )

    asyncio.run(exercise())


def test_multiple_observed_people_do_not_select_a_delivery_recipient() -> None:
    async def exercise() -> None:
        robot = MultipleWaterBottlesFakeRobot()
        original_observe = robot.observe_scene

        def multiple_people() -> dict[str, object]:
            scene = original_observe()
            objects = scene["objects"]
            assert isinstance(objects, list)
            objects.append({
                "object_ref": "other-person", "description": "user",
                "distance_m": 1.0, "bearing_rad": 0.0,
            })
            return scene

        robot.observe_scene = multiple_people  # type: ignore[method-assign]
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        with pytest.raises(RuntimeError, match="recipient is not uniquely observed"):
            await service._navigate_to_observed_resource(
                "user", speed_mps=0.12, stop_distance_m=0.9
            )
        assert not robot.commands or all(
            command.velocities[0] == 0 for command in robot.commands
        )

    asyncio.run(exercise())


def test_known_resource_without_matching_observation_refuses_motion() -> None:
    async def exercise() -> None:
        robot = ObservedResourceFakeRobot(description="cup of water")
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {"kind": "physical_object", "description": "bottle of milk"},
                    "source": {"status": "known", "description": "table ahead"},
                    "recipient": {"description": "user"},
                },
            },
        )
        with pytest.raises(RuntimeError, match="not uniquely observed"):
            await service.call_tool("soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]})
        assert not robot.commands or all(
            command.velocities[0] == 0 for command in robot.commands
        )
        assert service.active_task is None

    asyncio.run(exercise())


def test_missing_recipient_prevents_mock_delivery_after_pickup() -> None:
    async def exercise() -> None:
        robot = ObservedResourceFakeRobot(user_description="unrelated person")
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {"kind": "physical_object", "description": "bottle of milk"},
                    "source": {"status": "known"},
                    "recipient": {"description": "user"},
                },
            },
        )
        with pytest.raises(RuntimeError, match="not uniquely observed"):
            await service.call_tool("soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]})
        assert any(command.x_velocity > 0 for command in service.controller.seen_commands)
        assert [pose.pose for pose in robot.visual_arm_poses][-1] == "rest"
        assert service.active_task is None
        assert service.controller.command == PolicyCommand()

    asyncio.run(exercise())


def test_observed_resource_approach_stop_cannot_report_delivery() -> None:
    async def exercise() -> None:
        robot = ObservedResourceFakeRobot()
        robot.distance_m = 5.0
        service = SoridormiRuntimeToolService(
            robot=robot, controller=FakeController(), control_hz=20.0
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {"kind": "physical_object", "description": "bottle of milk"},
                    "source": {"status": "known"},
                    "recipient": {"description": "user"},
                },
            },
        )
        execution = asyncio.create_task(
            service.call_tool("soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]})
        )
        await asyncio.sleep(0.1)
        await service.call_tool("soridormi.motion.stop", {})
        result = await execution
        assert result["completed"] is False
        assert result["stopped"] is True
        assert "resource_outcome" not in result
        assert service.active_task is None
        assert service.controller.command == PolicyCommand()

    asyncio.run(exercise())


@pytest.mark.parametrize("reset_after_commands", [2, 18, 45])
def test_resource_mock_cannot_complete_after_simulator_reset(
    reset_after_commands: int,
) -> None:
    async def exercise() -> None:
        robot = ResettingResourceFakeRobot(
            reset_after_commands=reset_after_commands
        )
        service = SoridormiRuntimeToolService(
            robot=robot,
            controller=FakeController(),
            control_hz=20.0,
        )
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_and_deliver_resource",
                "parameters": {
                    "resource": {"kind": "physical_object", "description": "bottle of milk"},
                    "source": {"status": "known", "description": "table ahead"},
                    "recipient": {"description": "requester"},
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]}
        )
        assert result["completed"] is False
        assert result["reset_detected"] is True
        assert "resource_outcome" not in result
        assert service.active_task is None
        assert service.controller.command == PolicyCommand()

    asyncio.run(exercise())


def test_resource_mock_completion_does_not_depend_on_visual_arm_overlay() -> None:
    async def exercise() -> None:
        service = _resource_service()
        robot = service.robot
        assert isinstance(robot, ResourceFakeRobot)
        robot.set_visual_arm_pose = None  # type: ignore[method-assign]
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_resource",
                "parameters": {
                    "resource": {
                        "kind": "physical_object",
                        "description": "a cup of water",
                    },
                    "source": {"status": "unknown"},
                },
            },
        )

        result = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": plan["plan_id"]}
        )

        assert result["completed"] is True
        assert result["resource_outcome"]["resource_acquired"] is True
        assert result["visual_arm_pose"] is None
        assert result["visual_arm_mocked_simulation"] is False

    asyncio.run(exercise())


def test_runtime_service_executes_granular_resource_chain() -> None:
    async def exercise() -> None:
        service = _resource_service()
        resource = {"kind": "physical_object", "description": "a cup of water"}
        acquire_plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "acquire_resource",
                "parameters": {"resource": resource, "source": {"status": "unknown"}},
            },
        )
        acquired = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": acquire_plan["plan_id"]}
        )
        assert acquired["resource_outcome"]["resource_acquired"] is True
        assert acquired["resource_outcome"]["resource_delivered"] is False
        assert acquired["visual_arm_pose"] == "hold"

        deliver_plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "deliver_resource",
                "parameters": {
                    "resource": resource,
                    "recipient": {"description": "requester"},
                },
            },
        )
        delivered = await service.call_tool(
            "soridormi.skill.execute_plan", {"plan_id": deliver_plan["plan_id"]}
        )
        assert delivered["resource_outcome"]["resource_delivered"] is True
        assert delivered["visual_arm_pose"] == "rest"

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
        assert any(command.x_velocity == 0.15 for command in service.controller.seen_commands)

    asyncio.run(exercise())


def test_runtime_service_executes_semantic_walk_forward_skill() -> None:
    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "walk_forward",
                "parameters": {
                    "speed": "slow",
                    "duration_s": 0.5,
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["skill_id"] == "walk_forward"
        assert plan["no_motion"] is False
        assert result["completed"] is True
        assert result["skill_id"] == "walk_forward"
        assert any(
            command.x_velocity == pytest.approx(MIN_FORWARD_WALK_SPEED_MPS)
            for command in service.controller.seen_commands
        )

    asyncio.run(exercise())


def test_runtime_service_executes_named_scripted_head_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        commanded_pitch = [command.positions[head_pitch_index] for command in robot.commands]
        assert result["completed"] is True
        assert result["skill_id"] == "nod_yes"
        assert result["no_motion"] is False
        assert min(commanded_pitch) < -0.10
        assert max(commanded_pitch) > 0.05
        assert commanded_pitch[-1] == pytest.approx(0.0)

    asyncio.run(exercise())


def test_runtime_service_executes_named_visual_expression_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "soridormi_runtime.mcp.runtime_tools.asyncio.sleep",
        no_sleep,
    )

    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "blink_eyes",
                "parameters": {
                    "count": 1,
                    "closed_duration_s": 0.08,
                    "open_duration_s": 0.12,
                },
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["skill_id"] == "blink_eyes"
        assert plan["no_motion"] is True
        assert result["completed"] is True
        assert result["skill_id"] == "blink_eyes"
        assert result["no_motion"] is True
        assert result["visual_expression_steps"] == 3
        assert [command.expression for command in service.robot.visual_expressions] == [
            "eyes_open",
            "eyes_closed",
            "eyes_open",
            "eyes_open",
        ]

    asyncio.run(exercise())


def test_runtime_service_executes_named_visual_arm_gesture_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "soridormi_runtime.mcp.runtime_tools.asyncio.sleep",
        no_sleep,
    )

    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "wave_hand",
                "parameters": {"side": "left", "count": 1, "duration_s": 1.2},
            },
        )
        result = await service.call_tool(
            "soridormi.skill.execute_plan",
            {"plan_id": plan["plan_id"]},
        )

        assert plan["no_motion"] is True
        assert result["completed"] is True
        assert result["no_motion"] is True
        assert result["visual_arm_pose_steps"] == 4
        assert [(command.pose, command.side) for command in service.robot.visual_arm_poses] == [
            ("rest", "both"),
            ("wave_up", "left"),
            ("wave_out", "left"),
            ("rest", "both"),
            ("rest", "both"),
        ]

    asyncio.run(exercise())


def test_runtime_visual_arm_gesture_restores_pose_after_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_sleep(_delay: float) -> None:
        raise RuntimeError("simulated scheduler failure")

    monkeypatch.setattr(
        "soridormi_runtime.mcp.runtime_tools.asyncio.sleep",
        fail_sleep,
    )

    async def exercise() -> None:
        service = _service()
        plan = await service.call_tool(
            "soridormi.skill.create_plan",
            {
                "skill_id": "celebrate",
                "parameters": {"duration_s": 1.0},
            },
        )

        with pytest.raises(RuntimeError, match="simulated scheduler failure"):
            await service.call_tool(
                "soridormi.skill.execute_plan",
                {"plan_id": plan["plan_id"]},
            )

        assert service.robot.visual_arm_poses[-1] == VisualArmPoseCommand(
            pose="rest", side="both"
        )
        assert service.active_lanes == {}

    asyncio.run(exercise())


def test_runtime_service_rejects_unsupported_named_skill() -> None:
    async def exercise() -> None:
        service = _service()
        with pytest.raises(ValueError, match="not supported by the runtime adapter"):
            await service.call_tool(
                "soridormi.skill.create_plan",
                {"skill_id": "point_direction", "parameters": {}},
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("termination", ["complete", "cancel", "timeout"])
def test_turn_repetition_uses_existing_ordered_execution_and_cancellation(termination):
    async def exercise():
        service = _service(control_hz=100.0)
        created = await service.call_tool("soridormi.skill.create_plan", {
            "skill_id": "turn_in_place", "parameters": {"count": 2, "duration_s": 0.5, "yaw_radps": -0.12},
        })
        execution = asyncio.create_task(service.call_tool("soridormi.skill.execute_plan", {"plan_id": created["plan_id"]}))
        if termination == "cancel":
            await asyncio.sleep(0.05)
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await execution
        elif termination == "timeout":
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(execution, timeout=0.05)
        else:
            result = await execution
            assert result["completed"] is True
        assert service.active_task is None
        assert service.controller.command == PolicyCommand()
        assert any(command.yaw_velocity == -0.12 for command in service.controller.seen_commands)
    asyncio.run(exercise())
