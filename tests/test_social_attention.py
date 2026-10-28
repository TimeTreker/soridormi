"""The long-running attend_to_person skill, driven on a simulated clock without a model."""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Callable

import pytest

from soridormi_api import MotorCommand, VisualExpressionCommand
from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
from soridormi_runtime.social_attention import (
    BLINK_INTERVAL_S,
    FOLLOW_VELOCITY_RADPS,
    GLANCE_INTERVAL_S,
    POSTURE_OFFSETS_RAD,
    AttentionBehavior,
    AttentionLifetime,
    AttentionRequest,
    AttentionRequestError,
)
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest

from test_mcp_runtime_tools import REQUESTER_MARKER, FakeController, HeadFakeRobot

REAL_SLEEP = asyncio.sleep
CONTROL_HZ = 50.0


class SimClock:
    """Simulated time: every awaited sleep advances it, and no test waits in real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += max(float(delay), 0.0)
        await REAL_SLEEP(0)


class PersonRobot(HeadFakeRobot):
    """A head robot whose scene shows the requester at a bearing scripted over time."""

    def __init__(self, clock: SimClock, bearing_at: Callable[[float], float | None]) -> None:
        super().__init__()
        self.clock = clock
        self.bearing_at = bearing_at
        self.visual_expressions: list[tuple[float, str]] = []
        self.head_trace: list[tuple[float, float, float]] = []

    def observe_scene(self) -> dict[str, object]:
        bearing = self.bearing_at(self.clock.now)
        objects = [] if bearing is None else [{**REQUESTER_MARKER, "bearing_rad": bearing}]
        return {"mocked_simulation": True, "objects": objects}

    def send_motor_command(self, command: MotorCommand) -> None:
        super().send_motor_command(command)
        by_name = dict(zip(command.names, command.positions))
        self.head_trace.append((self.clock.now, by_name["head_yaw"], by_name["head_pitch"]))

    def set_visual_expression(self, command: VisualExpressionCommand) -> str:
        self.visual_expressions.append((self.clock.now, command.expression))
        return command.expression


def _attending_service(
    monkeypatch: pytest.MonkeyPatch,
    bearing_at: Callable[[float], float | None],
    *,
    control_hz: float = CONTROL_HZ,
) -> tuple[SoridormiRuntimeToolService, PersonRobot, SimClock]:
    clock = SimClock()
    monkeypatch.setattr("soridormi_runtime.mcp.runtime_tools.asyncio.sleep", clock.sleep)
    robot = PersonRobot(clock, bearing_at)
    service = SoridormiRuntimeToolService(
        robot=robot, controller=FakeController(), control_hz=control_hz
    )
    monkeypatch.setattr(service, "_attention_clock", clock)
    return service, robot, clock


async def _run_until(clock: SimClock, until_s: float, service: SoridormiRuntimeToolService) -> None:
    while clock.now < until_s and service.attention_status()["active"]:
        await REAL_SLEEP(0)


def test_attention_follows_a_person_who_walks_around(monkeypatch: pytest.MonkeyPatch) -> None:
    def walking(now: float) -> float:
        return 0.1 if now < 3.0 else 0.4 if now < 10.0 else -0.3

    service, robot, clock = _attending_service(monkeypatch, walking)

    async def exercise() -> dict[str, Any]:
        started = await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        assert started["started"] is True and started["status"] == "running"
        await _run_until(clock, 17.0, service)
        return await service.call_tool("soridormi.attention.stop", {"reason": "test"})

    stopped = asyncio.run(exercise())

    assert stopped["end_reason"] == "test" and stopped["active"] is False
    assert stopped["motor_commands_sent"] is True
    # Each new position is reached and held at most 0.35 rad/s. A glance-away keeps the
    # head off the person for at most about 2 s and comes at least 7 s apart, so every
    # window below holds the person whatever the run's random glance timing.
    for start, end, bearing in ((0.5, 3.0, 0.1), (5.0, 10.0, 0.4), (12.5, 17.0, -0.3)):
        yaws = [yaw for when, yaw, _ in robot.head_trace if start <= when <= end]
        assert any(abs(yaw - bearing) < 1e-6 for yaw in yaws), (start, yaws[-3:])
    steps = [b[1] - a[1] for a, b in zip(robot.head_trace, robot.head_trace[1:])]
    assert max(abs(step) for step in steps) <= FOLLOW_VELOCITY_RADPS / CONTROL_HZ + 1e-9
    # A listening posture: the gaze is lowered, never raised.
    assert all(pitch <= 0.0 + 1e-9 for _, _, pitch in robot.head_trace)
    assert service.active_lanes == {} and service.active_task is None


def test_attention_never_nods_or_shakes(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.0, control_hz=20.0)

    async def exercise() -> None:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 90.0, service)
        await service.call_tool("soridormi.attention.stop", {})

    asyncio.run(exercise())

    # Away from the person, the head drifts to one side at a time, and the next drift
    # comes at least GLANCE_INTERVAL_S later: no left-right alternation within a glance.
    away = [(when, yaw) for when, yaw, _ in robot.head_trace if abs(yaw) > 0.02]
    episodes: list[list[tuple[float, float]]] = []
    for when, yaw in away:
        if episodes and when - episodes[-1][-1][0] < 0.5:
            episodes[-1].append((when, yaw))
        else:
            episodes.append([(when, yaw)])
    assert len(episodes) >= 5
    for episode in episodes:
        assert len({math.copysign(1.0, yaw) for _, yaw in episode}) == 1
    starts = [episode[0][0] for episode in episodes]
    assert min(b - a for a, b in zip(starts, starts[1:])) >= GLANCE_INTERVAL_S[0]
    # The pitch settles to the listening offset and lowers further only together with a
    # sideways glance (a diagonal, 1 rad down per 3 rad aside), never on its own as a nod.
    listening = POSTURE_OFFSETS_RAD["listening"][1]
    settled = [(yaw, pitch) for when, yaw, pitch in robot.head_trace if when > 1.0]
    assert max(pitch for _, pitch in settled) <= listening + 1e-9
    assert min(pitch for _, pitch in settled) >= listening - 0.05 - 1e-9
    assert all(abs(yaw) >= 3.0 * (listening - pitch) - 1e-6 for yaw, pitch in settled)


def test_natural_blinks_and_a_busy_eye_skips_one(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.0, control_hz=20.0)

    async def exercise() -> None:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 30.0, service)
        async with service._independent_output_lock("visual.eyes"):
            held = len(robot.visual_expressions)
            await _run_until(clock, 50.0, service)
            assert len(robot.visual_expressions) == held
        await service.call_tool("soridormi.attention.stop", {})

    asyncio.run(exercise())

    closes = [when for when, expression in robot.visual_expressions if expression == "eyes_closed"]
    assert len(closes) >= 30.0 / BLINK_INTERVAL_S[1]
    assert robot.visual_expressions[-1][1] == "eyes_open"


def test_a_body_skill_suspends_attention_and_it_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.2)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 2.0, service)
        async with service._locomotion_lock:  # what every physical skill holds
            ticks = len(robot.head_trace)
            await _run_until(clock, 6.0, service)
            assert service.attention_status()["status"] == "suspended"
            assert len(robot.head_trace) == ticks
        await _run_until(clock, 8.0, service)
        status = await service.call_tool("soridormi.attention.status", {})
        await service.call_tool("soridormi.attention.stop", {})
        return status

    status = asyncio.run(exercise())

    assert status["status"] == "running" and status["suspensions"] == 1
    assert [item["status"] for item in status["transitions"]] == ["running", "suspended", "running"]
    assert status["transitions"][-1]["reason"] == "resumed"


def test_a_person_unseen_for_five_seconds_ends_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    def present(now: float) -> float | None:
        if 2.0 <= now < 5.0:  # steps out of view briefly, then returns
            return None
        return 0.1 if now < 9.0 else None

    service, robot, clock = _attending_service(monkeypatch, present)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 30.0, service)
        return service.attention_status()

    status = asyncio.run(exercise())

    # Out of view for 3 s does not end it; from 9 s on, it ends 5 s later.
    assert status["end_reason"] == "target_lost" and status["active"] is False
    assert 14.0 <= clock.now < 14.5


@pytest.mark.parametrize(
    ("tool", "reason"),
    [("soridormi.motion.stop", "motion_stop"), ("soridormi.safety.emergency_stop", "emergency_stop")],
)
def test_stop_and_emergency_stop_end_attention(
    monkeypatch: pytest.MonkeyPatch, tool: str, reason: str
) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.1)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 1.0, service)
        await service.call_tool(tool, {"reason": "test"})
        await _run_until(clock, 2.0, service)
        return service.attention_status()

    status = asyncio.run(exercise())

    assert status["active"] is False and status["end_reason"] == reason


def test_cancelling_another_activity_does_not_end_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.1)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        plan = await service.call_tool(
            "soridormi.activity.create_plan",
            {"members": [{"member_id": "eyes", "skill_id": "blink_eyes", "parameters": {"count": 1}}]},
        )
        await service.call_tool(
            "soridormi.activity.cancel", {"compiled_activity_id": plan["compiled_activity_id"]}
        )
        await _run_until(clock, 2.0, service)
        status = service.attention_status()
        await service.call_tool("soridormi.attention.stop", {})
        return status

    assert asyncio.run(exercise())["status"] == "running"


def test_an_unseen_person_never_starts_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: None)

    result = asyncio.run(service.call_tool("soridormi.attention.start", {"target_ref": "requester"}))

    assert result["started"] is False and result["status"] == "target_unavailable"
    assert result["target_resolution"]["status"] == "not_observed"
    assert robot.commands == [] and service.attention_status()["status"] == "idle"


def test_same_person_changes_posture_and_a_new_person_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.0)
    robot.observe_scene = lambda: {  # type: ignore[method-assign]
        "mocked_simulation": True,
        "objects": [
            {**REQUESTER_MARKER, "bearing_rad": 0.0},
            {"object_ref": "guest", "description": "guest person", "relative_direction": "left",
             "distance_m": 2.0, "bearing_rad": 0.3},
        ],
    }

    async def exercise() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        first = await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 3.0, service)
        thinking = await service.call_tool(
            "soridormi.attention.start", {"target_ref": "requester", "posture": "thinking"}
        )
        await _run_until(clock, 6.0, service)
        switched = await service.call_tool("soridormi.attention.start", {"target_ref": "guest"})
        await service.call_tool("soridormi.attention.stop", {})
        return first, thinking, switched

    first, thinking, switched = asyncio.run(exercise())

    assert thinking["updated"] is True and thinking["attention_id"] == first["attention_id"]
    assert thinking["posture"] == "thinking"
    yaw_offset, pitch_offset = POSTURE_OFFSETS_RAD["thinking"]
    assert any(
        abs(yaw - yaw_offset) < 1e-6 and abs(pitch - pitch_offset) < 1e-6
        for when, yaw, pitch in robot.head_trace if 3.0 < when <= 6.0
    )
    assert switched["started"] is True and switched["attention_id"] != first["attention_id"]
    assert switched["target_ref"] == "guest"


def test_caller_gaze_geometry_and_unknown_arguments_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.0)
    for args in (
        {"target_ref": "requester", "target_yaw_rad": 0.2},
        {"target_ref": "requester", "nod": True},
        {"target_ref": "requester", "posture": "nodding"},
        {"target_ref": "requester", "max_duration_s": 3600},
        {"target_ref": ""},
    ):
        with pytest.raises(AttentionRequestError):
            asyncio.run(service.call_tool("soridormi.attention.start", args))
    assert robot.commands == []


def test_skill_list_offers_attend_to_person_as_long_running(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.0)

    listed = {item["skill_id"]: item for item in service.list_runtime_skills()["skills"]}

    attend = listed["attend_to_person"]
    assert attend["lifecycle"] == "long_running"
    assert attend["lifecycle_tools"] == {
        "start": "soridormi.attention.start",
        "status": "soridormi.attention.status",
        "stop": "soridormi.attention.stop",
    }
    assert "social_attention" in attend["metadata"]["behavior_domains"]
    assert attend["parameters_schema"]["required"] == ["target_ref"]
    assert "target_yaw_rad" not in attend["parameters_schema"]["properties"]
    assert "lifecycle" not in listed["look_at_person"]
    with pytest.raises(ValueError):
        service.create_runtime_skill_plan({"skill_id": "attend_to_person", "parameters": {}})
    local = SoridormiLocalToolService(mode="sim")
    assert "attend_to_person" not in {item["skill_id"] for item in local.list_skills()["skills"]}
    assert local.call_tool("soridormi.attention.start", {"target_ref": "requester"})["started"] is False


def test_behavior_and_lifetime_are_pure_and_bounded() -> None:
    skill = next(
        item for item in load_skill_manifest(DEFAULT_SKILL_MANIFEST)["skills"]
        if item["id"] == "attend_to_person"
    )
    request = AttentionRequest.from_args({"target_ref": "requester"}, skill)
    assert (request.posture, request.target_lost_s, request.max_duration_s) == ("listening", 5.0, 600.0)

    behavior = AttentionBehavior("listening", rng=random.Random(7))
    assert behavior.head_goal(0.0, None, (0.3, -0.1)) == (0.3, -0.1)
    assert behavior.head_goal(0.0, 0.9, (0.0, 0.0))[0] == pytest.approx(0.55)

    lifetime = AttentionLifetime(request)
    assert lifetime.end_reason(1.0, seen=False) is None
    assert lifetime.end_reason(5.9, seen=False) is None
    assert lifetime.end_reason(6.0, seen=False) == "target_lost"
    assert AttentionLifetime(request).end_reason(600.0, seen=True) == "max_duration"


def test_a_person_beyond_the_head_range_is_turned_toward_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turned: list[int] = [0]

    def beyond_until_turned(now: float) -> float:
        # Each walking-arc segment brings the person 0.1 rad closer to straight ahead.
        return min(-1.0 + 0.1 * turned[0], -0.3)

    service, robot, clock = _attending_service(monkeypatch, beyond_until_turned)

    async def one_step_segment(duration_s: float) -> bool:
        # One controller step per segment, so the test does not depend on real time.
        await service._step_controller()
        turned[0] += 1
        return True

    monkeypatch.setattr(service, "_run_segment", one_step_segment)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 4.0, service)
        status = service.attention_status()
        await service.call_tool("soridormi.attention.stop", {})
        return status

    status = asyncio.run(exercise())

    # The turn stops once the person is inside the head range less its margin; the head
    # then follows the person there.
    from soridormi_runtime.mcp.runtime_tools import GAZE_TURN_HEAD_MARGIN_RAD
    reached = -1.0 + 0.1 * turned[0]
    assert abs(reached) <= 0.55 - GAZE_TURN_HEAD_MARGIN_RAD < abs(reached) + 0.1
    assert status["body_turns"] == 1 and status["status"] == "running"
    assert [item["status"] for item in status["transitions"]][:3] == ["running", "turning", "running"]
    assert robot.head_trace[-1][1] == pytest.approx(reached, abs=1e-6)
    assert service.active_lanes == {} and service.controller.command.yaw_velocity == 0.0


def test_a_stop_during_the_observation_never_starts_a_body_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2026-10-28: a stop that lands while attention reads the scene must not be
    followed by the slow walking arc toward a person beyond the head range."""
    monkeypatch.setattr("soridormi_runtime.mcp.runtime_tools.GAZE_TURN_SEGMENT_S", 0.01)
    service, robot, clock = _attending_service(monkeypatch, lambda now: -1.0)
    observed = [0]
    original = robot.observe_scene

    def observe_then_stop() -> dict[str, object]:
        observed[0] += 1
        if observed[0] == 2 and service._attention is not None:  # the loop's first read
            service._attention.request_stop("motion_stop")
        return original()

    robot.observe_scene = observe_then_stop  # type: ignore[method-assign]

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        await _run_until(clock, 2.0, service)
        return service.attention_status()

    status = asyncio.run(exercise())

    assert status["end_reason"] == "motion_stop" and status["body_turns"] == 0
    assert not any(command.yaw_velocity for command in service.controller.seen_commands)


def test_a_task_waiting_for_the_body_interrupts_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("soridormi_runtime.mcp.runtime_tools.GAZE_TURN_SEGMENT_S", 0.01)
    service, robot, clock = _attending_service(monkeypatch, lambda now: -1.0)
    task_ran: list[float] = []

    async def task_skill() -> None:
        async with service._locomotion_lock:  # what every physical skill holds
            task_ran.append(clock.now)
            assert service.attention_status()["status"] == "suspended"
            await REAL_SLEEP(0)

    async def exercise() -> dict[str, Any]:
        await service.call_tool("soridormi.attention.start", {"target_ref": "requester"})
        while service.attention_status()["status"] != "turning" or not service.active_lanes:
            await REAL_SLEEP(0)
        await asyncio.wait_for(task_skill(), timeout=5.0)
        await _run_until(clock, clock.now + 0.5, service)
        status = service.attention_status()
        await service.call_tool("soridormi.attention.stop", {})
        return status

    status = asyncio.run(exercise())

    assert task_ran and status["active"] is True


def test_overlapping_starts_leave_one_run(monkeypatch: pytest.MonkeyPatch) -> None:
    service, robot, clock = _attending_service(monkeypatch, lambda now: 0.1)

    async def exercise() -> list[dict[str, Any]]:
        results = await asyncio.gather(
            service.call_tool("soridormi.attention.start", {"target_ref": "requester"}),
            service.call_tool("soridormi.attention.start", {"target_ref": "requester", "posture": "thinking"}),
        )
        await _run_until(clock, 1.0, service)
        await service.call_tool("soridormi.attention.stop", {})
        return list(results)

    first, second = asyncio.run(exercise())

    assert first["started"] is True and second["updated"] is True
    assert second["attention_id"] == first["attention_id"]
    assert service._attention_task is not None and service._attention_task.done()
