from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scenario_runner.scenario import Scenario, load_scenario
from soridormi_sim.milk_bottle_scene import (
    MILK_BOTTLE_GEOM,
    MILK_TABLE_GEOM,
    build_default_world_scene_xml,
)


DEFAULT_SCENARIO = Path("configs/simulation_scenarios/default.json")


def test_default_scenario_owns_dynamic_bottle_separately_from_static_world() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO)
    static_world = build_default_world_scene_xml("<mujoco><worldbody></worldbody></mujoco>")

    assert scenario.world_map.static_layout == "table_chairs"
    assert MILK_TABLE_GEOM in static_world
    assert MILK_BOTTLE_GEOM not in static_world
    assert scenario.dynamic_elements[0].name == "scenario_milk_bottle"
    assert {geom.name for geom in scenario.dynamic_elements[0].geoms} >= {MILK_BOTTLE_GEOM}


def test_scenario_rejects_events_for_undeclared_elements() -> None:
    payload = json.loads(DEFAULT_SCENARIO.read_text(encoding="utf-8"))
    payload["events"] = [{"at_sim_time_s": 0.1, "element_name": "scenario_unknown", "position_xyz": [1, 0, 0.5]}]
    with pytest.raises(ValueError, match="undeclared dynamic element"):
        Scenario.model_validate(payload)


def test_runner_creates_bottle_and_applies_sim_time_event(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("zmq")
    from scenario_runner.main import scene_request

    source = Path("/workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
    if not source.exists():
        pytest.skip("Open Duck MuJoCo world map is unavailable")
    payload = json.loads(DEFAULT_SCENARIO.read_text(encoding="utf-8"))
    payload["dynamic_elements"].append({
        "name": "scenario_actor",
        "position_xyz": [2, 2, 0.5],
        "geoms": [{"name": "scenario_actor_marker", "shape": "sphere", "size": [0.1, 0, 0]}],
    })
    payload["events"] = [
        {"at_sim_time_s": 0.04, "element_name": "scenario_milk_bottle", "position_xyz": [8, 0, 0.86]},
        {"at_sim_time_s": 0.06, "element_name": "scenario_actor", "position_xyz": [3, 2, 0.5]},
    ]
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    environment = dict(os.environ)
    environment.update({
        "SIM_PORT": str(port),
        "SORIDORMI_MUJOCO_VIEWER": "0",
        "SORIDORMI_SCENE_RUN_DIR": str(tmp_path),
    })
    runner = subprocess.Popen(
        [sys.executable, "-m", "scenario_runner", "--scenario", str(scenario_path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            if runner.poll() is not None:
                raise AssertionError(f"scenario runner exited: {runner.stdout.read()}")
            try:
                initial = scene_request(port + 1, {"kind": "list_objects"}, timeout_ms=200)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise AssertionError("scenario runner did not start")
                time.sleep(0.1)
        assert {item["name"] for item in initial["objects"]} == {"scenario_milk_bottle", "scenario_actor"}

        import zmq

        context = zmq.Context.instance()
        robot = context.socket(zmq.REQ)
        robot.setsockopt(zmq.RCVTIMEO, 2000)
        robot.connect(f"tcp://127.0.0.1:{port}")
        try:
            robot.send_json({"kind": "observe_scene"})
            assert robot.recv_json()["scene_observation"]["objects"][0]["distance_m"] == 10.0
            for _ in range(4):
                robot.send_json({"kind": "get_state"})
                assert robot.recv_json()["ok"] is True
            deadline = time.monotonic() + 5
            while True:
                moved = scene_request(port + 1, {"kind": "list_objects"})
                positions = {item["name"]: item["position_xyz"] for item in moved["objects"]}
                if positions["scenario_milk_bottle"][0] == 8.0 and positions["scenario_actor"][0] == 3.0:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("scenario event was not applied")
                time.sleep(0.05)
            robot.send_json({"kind": "observe_scene"})
            assert robot.recv_json()["scene_observation"]["objects"][0]["distance_m"] == 8.0
            robot.send_json({"kind": "reset"})
            assert robot.recv_json()["ok"] is True
            reset_state = scene_request(port + 1, {"kind": "list_objects"})
            assert reset_state["reset_count"] == 1
            assert {item["name"]: item["position_xyz"][0] for item in reset_state["objects"]} == {
                "scenario_milk_bottle": 10.0,
                "scenario_actor": 2.0,
            }
            for _ in range(4):
                robot.send_json({"kind": "get_state"})
                assert robot.recv_json()["ok"] is True
            deadline = time.monotonic() + 5
            while True:
                replayed = scene_request(port + 1, {"kind": "list_objects"})
                positions = {item["name"]: item["position_xyz"][0] for item in replayed["objects"]}
                if positions == {"scenario_milk_bottle": 8.0, "scenario_actor": 3.0}:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("scenario events were not replayed after reset")
                time.sleep(0.05)
        finally:
            robot.close(0)
    finally:
        runner.terminate()
        runner.communicate(timeout=10)
    assert list(tmp_path.glob("scenario-*")) == []
