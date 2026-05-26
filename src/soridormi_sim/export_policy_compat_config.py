from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from soridormi_sim.robot_config import load_robot_config


def _name_list(model: Any, mujoco: Any, obj_type: Any, count: int) -> list[str]:
    names: list[str] = []
    for i in range(count):
        name = mujoco.mj_id2name(model, obj_type, i)
        if name:
            names.append(str(name))
    return names


def _safe_name_exists(names: list[str], name: str) -> bool:
    return name in set(names)


def build_policy_compat_snippet(robot_config_path: str | None = None) -> dict[str, Any]:
    import mujoco

    config = load_robot_config(robot_config_path)
    model_path = Path(config.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))

    actuator_names = _name_list(model, mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    body_names = _name_list(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    geom_names = _name_list(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    sensor_names = _name_list(model, mujoco, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)

    home_ctrl: list[float] | None = None
    try:
        key = model.keyframe("home")
        home_ctrl = [float(x) for x in key.ctrl]
    except Exception:
        home_ctrl = None

    positions: dict[str, float] = {}
    if home_ctrl is not None and len(home_ctrl) == len(actuator_names):
        positions = {name: float(value) for name, value in zip(actuator_names, home_ctrl)}

    left_body = "foot_assembly" if _safe_name_exists(body_names, "foot_assembly") else ""
    right_body = "foot_assembly_2" if _safe_name_exists(body_names, "foot_assembly_2") else ""
    ground_body = "floor" if _safe_name_exists(body_names, "floor") else ""

    snippet: dict[str, Any] = {
        "default_pose": {
            "positions": positions,
        },
        "action_mapping": {
            "action_scale": 0.25,
            "max_motor_velocity": 5.24,
            "speed_limit_enabled": True,
            "clip_to_limits": True,
        },
        "policy_observation": {
            "accelerometer_bias_xyz": [1.3, 0.0, 0.0],
            "use_state_feet_contacts": True,
            "foot_contact": {
                "left_body": left_body or "foot_assembly",
                "right_body": right_body or "foot_assembly_2",
                "ground_body": ground_body or "floor",
                "left_geoms": [
                    name for name in ["left_foot_bottom_tpu"] if _safe_name_exists(geom_names, name)
                ]
                or ["left_foot_bottom_tpu"],
                "right_geoms": [
                    name for name in ["right_foot_bottom_tpu"] if _safe_name_exists(geom_names, name)
                ]
                or ["right_foot_bottom_tpu"],
                "ground_geoms": [name for name in ["floor"] if _safe_name_exists(geom_names, name)]
                or ["floor"],
            },
        },
        "_inspection": {
            "model_path": str(model_path),
            "actuator_names": actuator_names,
            "sensor_names": sensor_names,
            "has_home_ctrl": home_ctrl is not None,
            "known_contact_bodies_present": {
                "foot_assembly": _safe_name_exists(body_names, "foot_assembly"),
                "foot_assembly_2": _safe_name_exists(body_names, "foot_assembly_2"),
                "floor": _safe_name_exists(body_names, "floor"),
            },
        },
    }
    return snippet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Open Duck policy compatibility YAML snippet.")
    parser.add_argument(
        "--robot-config",
        default=None,
        help="Robot YAML path. Defaults to SORIDORMI_ROBOT_CONFIG or /app/configs/robots/open_duck_mini_v2.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output YAML path. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    snippet = build_policy_compat_snippet(args.robot_config)
    text = yaml.safe_dump(snippet, sort_keys=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote policy compatibility snippet: {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
