from __future__ import annotations

import asyncio
import math

import pytest

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle
from soridormi_sim.milk_bottle_scene import (
    MILK_BOTTLE_GEOM,
    build_milk_bottle_scene_xml,
    mock_milk_observation,
)
from soridormi_sim.mujoco_backend import FakeMujocoBackend
from soridormi_sim.robot_config import load_robot_config


def test_optional_bottle_is_a_named_noncontact_world_geom() -> None:
    xml = build_milk_bottle_scene_xml("<mujoco><worldbody></worldbody></mujoco>")
    assert f'name="{MILK_BOTTLE_GEOM}"' in xml
    assert 'name="soridormi_mock_milk_bottle_cap"' in xml
    assert 'pos="50 0 0.12"' in xml
    assert 'contype="0" conaffinity="0"' in xml
    with pytest.raises(ValueError, match="already exists"):
        build_milk_bottle_scene_xml(xml)


def test_mock_observation_uses_scene_position_and_robot_heading() -> None:
    base = {
        "robot_xyz": (0.0, 0.0, 0.0),
        "robot_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "robot_time_s": 2.0,
    }
    detected = mock_milk_observation(bottle_xyz=(50.0, 0.0, 0.12), **base)
    assert detected["mocked_simulation"] is True
    assert detected["objects"] == [{
        "object_ref": MILK_BOTTLE_GEOM,
        "description": "bottle of milk",
        "relative_direction": "in front of Chromie",
        "distance_m": 50.0,
    }]
    assert mock_milk_observation(bottle_xyz=None, **base)["objects"] == []
    assert mock_milk_observation(bottle_xyz=(70.0, 0.0, 0.12), **base)["objects"] == []
    turned = {**base, "robot_quat_wxyz": (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))}
    assert mock_milk_observation(bottle_xyz=(50.0, 0.0, 0.12), **turned)["objects"] == []


def test_fake_simulator_and_local_adapter_cannot_invent_a_bottle() -> None:
    backend = FakeMujocoBackend(config=load_robot_config("configs/robots/open_duck_mini_v2.yaml"))
    assert backend.observe_scene()["objects"] == []
    assert SoridormiLocalToolService(mode="sim").call_tool(
        "soridormi.robot.observe_scene", {}
    )["objects"] == []
    with pytest.raises(RuntimeError, match="only in sim mode"):
        SoridormiLocalToolService(mode="hardware_shadow").call_tool(
            "soridormi.robot.observe_scene", {}
        )


def test_scene_tool_is_sim_only_and_reports_mock_source() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    tools = [tool for agent in bundle.agents for tool in agent.tools]
    tool = next(tool for tool in tools if tool.name == "soridormi.robot.observe_scene")
    assert tool.availability.modes == ["sim"]
    assert tool.safety_class == "safe_read"
    assert tool.output_schema["properties"]["mocked_simulation"]["const"] is True


def test_simulator_api_and_runtime_tool_preserve_mock_provenance() -> None:
    pytest.importorskip("zmq")
    from soridormi_api.server import RobotApiServer
    from soridormi_api.types import ApiRequest
    from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService

    backend = FakeMujocoBackend(
        config=load_robot_config("configs/robots/open_duck_mini_v2.yaml")
    )
    response = RobotApiServer(backend)._handle(ApiRequest(kind="observe_scene"))
    assert response.ok is True
    assert response.scene_observation is not None
    assert response.scene_observation["objects"] == []

    class SceneRobot:
        def observe_scene(self):
            return mock_milk_observation(
                bottle_xyz=(50.0, 0.0, 0.12),
                robot_xyz=(0.0, 0.0, 0.0),
                robot_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                robot_time_s=2.0,
            )

    service = SoridormiRuntimeToolService(robot=SceneRobot(), controller=object())
    first = asyncio.run(service.call_tool("soridormi.robot.observe_scene", {}))
    second = asyncio.run(service.call_tool("soridormi.robot.observe_scene", {}))
    assert first["objects"][0]["distance_m"] == 50.0
    assert first["mocked_simulation"] is True
    assert first["observation_sequence"] == 1
    assert second["observation_sequence"] == 2
    assert first["observation_id"] != second["observation_id"]
