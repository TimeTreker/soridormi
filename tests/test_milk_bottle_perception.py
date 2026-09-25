from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path
from xml.etree import ElementTree

import pytest

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle
from soridormi_sim.milk_bottle_scene import (
    MILK_BOTTLE_GEOM,
    MILK_TABLE_GEOM,
    USER_BODY_GEOM,
    build_milk_bottle_scene_xml,
    mock_scene_observation,
)
from soridormi_sim.mujoco_backend import FakeMujocoBackend
from soridormi_sim.robot_config import load_robot_config


def test_optional_scene_places_bottle_on_noncontact_table_ten_metres_ahead() -> None:
    xml = build_milk_bottle_scene_xml("<mujoco><worldbody></worldbody></mujoco>")
    world = ElementTree.fromstring(xml).find("worldbody")
    assert world is not None
    geoms = {geom.attrib["name"]: geom for geom in world.iter("geom")}
    table = geoms[MILK_TABLE_GEOM]
    bottle = geoms[MILK_BOTTLE_GEOM]
    assert "soridormi_mock_milk_bottle_cap" in geoms
    assert len([name for name in geoms if name.startswith("soridormi_mock_milk_table_leg_")]) == 4
    table_body = world.find("body[@name='soridormi_mock_milk_table']")
    bottle_body = world.find("body[@name='scenario_milk_bottle']")
    assert table_body is not None and bottle_body is not None
    assert bottle_body.attrib["mocap"] == "true"
    table_pos = [float(value) for value in table_body.attrib["pos"].split()]
    table_half_size = [float(value) for value in table.attrib["size"].split()]
    bottle_pos = [float(value) for value in bottle_body.attrib["pos"].split()]
    bottle_half_size = [float(value) for value in bottle.attrib["size"].split()]
    assert table_pos[:2] == bottle_pos[:2] == [10.0, 0.0]
    assert bottle_pos[2] - bottle_half_size[1] == pytest.approx(
        table_pos[2] + float(table.attrib["pos"].split()[2]) + table_half_size[2]
    )
    assert all(
        geom.attrib["contype"] == geom.attrib["conaffinity"] == "0"
        for geom in geoms.values()
    )
    with pytest.raises(ValueError, match="already exists"):
        build_milk_bottle_scene_xml(xml)


def test_two_visual_chairs_are_left_of_chromies_initial_heading() -> None:
    world = ElementTree.fromstring(
        build_milk_bottle_scene_xml("<mujoco><worldbody></worldbody></mujoco>")
    ).find("worldbody")
    assert world is not None
    chairs = [body for body in world.findall("body") if "chair" in body.attrib["name"]]
    assert [chair.attrib["name"] for chair in chairs] == [
        "soridormi_mock_left_chair_1",
        "soridormi_mock_left_chair_2",
    ]
    assert [tuple(float(value) for value in chair.attrib["pos"].split()) for chair in chairs] == [
        (1.5, 2.0, 0.0),
        (3.0, 2.0, 0.0),
    ]
    for chair in chairs:
        assert "mocap" not in chair.attrib
        geoms = chair.findall("geom")
        assert len(geoms) == 6  # seat, back, and four legs
        assert all(geom.attrib["contype"] == geom.attrib["conaffinity"] == "0" for geom in geoms)


def test_mock_observation_uses_scene_position_and_robot_heading() -> None:
    base = {
        "robot_xyz": (0.0, 0.0, 0.0),
        "robot_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "robot_time_s": 2.0,
    }
    detected = mock_scene_observation(bottle_xyz=(10.0, 0.0, 0.86), **base)
    assert detected["mocked_simulation"] is True
    assert detected["objects"] == [{
        "object_ref": MILK_BOTTLE_GEOM,
        "description": "bottle of milk",
        "relative_direction": "in front of Chromie",
        "distance_m": 10.0,
        "bearing_rad": 0.0,
    }]
    assert mock_scene_observation(bottle_xyz=None, **base)["objects"] == []
    assert mock_scene_observation(bottle_xyz=(70.0, 0.0, 0.86), **base)["objects"] == []
    turned = {**base, "robot_quat_wxyz": (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))}
    assert mock_scene_observation(bottle_xyz=(10.0, 0.0, 0.86), **turned)["objects"] == []
    user = mock_scene_observation(
        bottle_xyz=None, user_xyz=(0.0, -1.5, 1.0), **base
    )["objects"]
    assert len(user) == 1
    assert user[0]["object_ref"] == USER_BODY_GEOM
    assert user[0]["relative_direction"] == "to Chromie's right"
    assert user[0]["distance_m"] == 1.5
    assert user[0]["bearing_rad"] == pytest.approx(-math.pi / 2)
    assert mock_scene_observation(
        bottle_xyz=None, user_xyz=(0.0, -70.0, 1.0), **base
    )["objects"] == []


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
            return mock_scene_observation(
                bottle_xyz=(10.0, 0.0, 0.86),
                robot_xyz=(0.0, 0.0, 0.0),
                robot_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                robot_time_s=2.0,
            )

    service = SoridormiRuntimeToolService(robot=SceneRobot(), controller=object())
    first = asyncio.run(service.call_tool("soridormi.robot.observe_scene", {}))
    second = asyncio.run(service.call_tool("soridormi.robot.observe_scene", {}))
    assert first["objects"][0]["distance_m"] == 10.0
    assert first["mocked_simulation"] is True
    assert first["observation_sequence"] == 1
    assert second["observation_sequence"] == 2
    assert first["observation_id"] != second["observation_id"]


def test_scene_control_moves_only_declared_dynamic_bottle(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from soridormi_sim.milk_bottle_scene import generate_milk_bottle_scene
    from soridormi_sim.mujoco_backend import MujocoBackend
    from soridormi_sim.scene_control import SceneControl

    sources = [
        Path("workspace/Open_Duck_Playground/playground/open_duck_mini_v2/xmls"),
        Path("/workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls"),
    ]
    source = next((path for path in sources if path.exists()), None)
    if source is None:
        pytest.skip("Open Duck MuJoCo source is unavailable")
    staged = tmp_path / "xmls"
    shutil.copytree(source, staged)
    scene = generate_milk_bottle_scene(staged / "scene_flat_terrain.xml", staged / "scene_test.xml")
    xml = scene.read_text(encoding="utf-8")
    scene.write_text(xml.replace(
        "</worldbody>",
        '<body name="scenario_actor" mocap="true" pos="2 3 0.5">'
        '<geom type="sphere" size="0.1" contype="0" conaffinity="0"/>'
        '</body></worldbody>',
    ), encoding="utf-8")
    backend = MujocoBackend(model_path=str(scene))
    control = SceneControl(backend)
    listed = control.handle({"kind": "list_objects"})
    assert {item["name"] for item in listed["objects"]} == {"scenario_milk_bottle", "scenario_actor"}
    assert backend.observe_scene()["objects"][0]["distance_m"] == 10.0
    moved = control.handle({
        "kind": "set_object_pose",
        "object_name": "scenario_milk_bottle",
        "position_xyz": [6.0, 0.0, 0.86],
    })
    assert moved["ok"] is True
    assert backend.observe_scene()["objects"][0]["distance_m"] == 6.0
    assert control.handle({"kind": "set_object_pose", "object_name": "scenario_actor", "position_xyz": [3, 3, 0.5]})["object"]["position_xyz"] == [3.0, 3.0, 0.5]
    with pytest.raises(ValueError, match="not declared dynamic"):
        control.handle({"kind": "set_object_pose", "object_name": "soridormi_mock_milk_table", "position_xyz": [5, 0, 0]})
    with pytest.raises(ValueError, match="three values"):
        control.handle({"kind": "set_object_pose", "object_name": "scenario_milk_bottle", "position_xyz": [1, 2]})
    with pytest.raises(ValueError, match="finite"):
        control.handle({"kind": "set_object_pose", "object_name": "scenario_milk_bottle", "position_xyz": [float("nan"), 0, 0]})
