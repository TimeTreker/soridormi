"""Run a scenario as a process separate from the Soridormi simulator server."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mujoco
import zmq

from soridormi_sim.milk_bottle_scene import generate_default_world_scene
from soridormi_sim.social_eye_scene import (
    VisualArmConfig,
    VisualLegConfig,
    generate_social_eye_scene,
)

from .scenario import Scenario, load_scenario


GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}


def prepare_scene(scenario: Scenario, work_dir: Path, *, visual_overlay: bool = True) -> Path:
    """Create the scenario's dynamic bodies with MjSpec before Soridormi loads it."""

    base = scenario.world_map.base_xml
    if not base.is_file():
        raise FileNotFoundError(f"scenario world map XML not found: {base}")
    xml_dir = work_dir / "xmls"
    shutil.copytree(base.parent, xml_dir, dirs_exist_ok=True)
    current = xml_dir / base.name
    if scenario.world_map.static_layout == "table_chairs":
        current = generate_default_world_scene(current, xml_dir / "scenario_static_world.xml")
    if visual_overlay:
        eyes = os.environ.get("SORIDORMI_MUJOCO_SOCIAL_EYES", "1") == "1"
        arms = os.environ.get("SORIDORMI_MUJOCO_VISUAL_ARMS", "1") == "1"
        if eyes or arms:
            result = generate_social_eye_scene(
                current,
                xml_dir / "scenario_visual_world.xml",
                include_eyes=eyes,
                arm_config=VisualArmConfig() if arms else None,
                leg_config=VisualLegConfig() if arms else None,
            )
            current = Path(result.output_path)

    base_model = mujoco.MjModel.from_xml_path(str(current))
    spec = mujoco.MjSpec.from_file(str(current))
    for element in scenario.dynamic_elements:
        body = spec.worldbody.add_body(
            name=element.name,
            mocap=True,
            pos=list(element.position_xyz),
            quat=list(element.quat_wxyz),
        )
        for geom in element.geoms:
            body.add_geom(
                name=geom.name,
                type=GEOM_TYPES[geom.shape],
                size=list(geom.size),
                pos=list(geom.local_position_xyz),
                rgba=list(geom.rgba),
                contype=1 if geom.collidable else 0,
                conaffinity=1 if geom.collidable else 0,
            )
    spec.compile()
    output = xml_dir / "scenario_compiled.xml"
    output.write_text(spec.to_xml(), encoding="utf-8")
    loaded = mujoco.MjModel.from_xml_path(str(output))
    if (loaded.nq, loaded.nv, loaded.nu) != (base_model.nq, base_model.nv, base_model.nu):
        raise RuntimeError("scenario assembly changed the robot state or actuator contract")
    for actuator_id in range(base_model.nu):
        old_name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        new_name = mujoco.mj_id2name(loaded, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if old_name != new_name:
            raise RuntimeError("scenario assembly changed actuator ordering")
    for element in scenario.dynamic_elements:
        body_id = mujoco.mj_name2id(loaded, mujoco.mjtObj.mjOBJ_BODY, element.name)
        if body_id < 0 or loaded.body_mocapid[body_id] < 0:
            raise RuntimeError(f"scenario dynamic body did not compile: {element.name}")
    return output


def scene_request(port: int, payload: dict[str, object], *, timeout_ms: int = 1000) -> dict[str, Any]:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.send_json(payload)
        response = socket.recv_json()
    finally:
        socket.close()
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"scene control rejected request: {response}")
    return response


def run_scenario(scenario: Scenario, work_dir: Path) -> None:
    model_path = prepare_scene(scenario, work_dir)
    port = int(os.environ.get("SIM_PORT", "5555"))
    for check_port in (port, port + 1):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", check_port)) == 0:
                raise RuntimeError(f"scenario simulator port is already in use: {check_port}")
    environment = dict(os.environ)
    environment["SORIDORMI_SIM_BACKEND"] = "mujoco"
    environment["MUJOCO_MODEL_PATH"] = str(model_path)
    server = subprocess.Popen([sys.executable, "-m", "soridormi_sim.mujoco_server"], env=environment)
    control_port = port + 1
    try:
        deadline = time.monotonic() + 15.0
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Soridormi simulator exited during startup: {server.returncode}")
            try:
                scene_request(control_port, {"kind": "list_objects"}, timeout_ms=200)
                break
            except (RuntimeError, zmq.ZMQError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Soridormi scene control did not start")
                time.sleep(0.1)
        print(f"Scenario runner loaded {scenario.name}: {model_path}", flush=True)
        next_event = 0
        previous_reset_count = 0
        while server.poll() is None:
            state = scene_request(control_port, {"kind": "list_objects"})
            sim_time = float(state["robot_time_s"])
            reset_count = int(state["reset_count"])
            if reset_count != previous_reset_count:
                next_event = 0
                print("Scenario clock reset; replaying timed events", flush=True)
            previous_reset_count = reset_count
            while next_event < len(scenario.events) and scenario.events[next_event].at_sim_time_s <= sim_time:
                event = scenario.events[next_event]
                request: dict[str, object] = {
                    "kind": "set_object_pose",
                    "object_name": event.element_name,
                    "position_xyz": list(event.position_xyz),
                }
                if event.quat_wxyz is not None:
                    request["quat_wxyz"] = list(event.quat_wxyz)
                scene_request(control_port, request)
                print(f"Scenario event {next_event}: {event.element_name} at sim time {sim_time:.3f}s", flush=True)
                next_event += 1
            time.sleep(0.02 if next_event < len(scenario.events) else 0.2)
        raise RuntimeError(f"Soridormi simulator exited: {server.returncode}")
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and play a MuJoCo simulation scenario")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="Build and compile the scene, then exit")
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    scene_root = Path(os.environ.get("SORIDORMI_SCENE_RUN_DIR", "/data/scenes"))
    scene_root.mkdir(parents=True, exist_ok=True)
    if not args.validate:
        def stop_on_terminate(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_on_terminate)
    try:
        with tempfile.TemporaryDirectory(prefix="scenario-", dir=scene_root) as temporary:
            work_dir = Path(temporary)
            if args.validate:
                path = prepare_scene(scenario, work_dir)
                print(json.dumps({"ok": True, "scenario": scenario.name, "dynamic_elements": len(scenario.dynamic_elements), "model_built": path.is_file()}))
                return 0
            run_scenario(scenario, work_dir)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
