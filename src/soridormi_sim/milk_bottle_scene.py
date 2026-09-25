"""Optional MuJoCo scene object for simulation perception tests.

The named geometry is a world object, not a detector result. The mock observer
reads its current MuJoCo position through the simulator API.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree

from .rough_ground_scene import rewrite_relative_includes

MILK_BOTTLE_GEOM = "soridormi_mock_milk_bottle"
MILK_TABLE_GEOM = "soridormi_mock_milk_table_top"
SCENARIO_BOTTLE_BODY = "scenario_milk_bottle"
USER_BODY_GEOM = "soridormi_mock_user_torso"
WATER_BOTTLE_GEOMS = tuple(f"soridormi_mock_water_bottle_{index}" for index in range(1, 4))


def _left_chair_xml(index: int, x: float) -> str:
    """Place a fixed visual chair beside the initial +x-facing robot."""

    name = f"soridormi_mock_left_chair_{index}"
    legs = "".join(
        f'        <geom name="{name}_leg_{side}" type="box" '
        f'size="0.035 0.035 0.205" pos="{leg_x} {leg_y} 0.205" '
        'rgba="0.13 0.27 0.31 1" contype="0" conaffinity="0"/>\n'
        for side, leg_x, leg_y in (
            ("front_left", 0.19, 0.19),
            ("front_right", 0.19, -0.19),
            ("back_left", -0.19, 0.19),
            ("back_right", -0.19, -0.19),
        )
    )
    return (
        f'    <body name="{name}" pos="{x:g} 2 0">\n'
        f'        <geom name="{name}_seat" type="box" '
        'size="0.25 0.25 0.035" pos="0 0 0.43" '
        'rgba="0.16 0.43 0.49 1" contype="0" conaffinity="0"/>\n'
        f'        <geom name="{name}_back" type="box" '
        'size="0.035 0.25 0.21" pos="-0.215 0 0.64" '
        'rgba="0.16 0.43 0.49 1" contype="0" conaffinity="0"/>\n'
        f'{legs}'
        '    </body>\n'
    )


def mock_scene_observation(
    *,
    bottle_xyz: tuple[float, float, float] | None,
    user_xyz: tuple[float, float, float] | None = None,
    water_bottles_xyz: Mapping[str, tuple[float, float, float]] | None = None,
    robot_xyz: tuple[float, float, float],
    robot_quat_wxyz: tuple[float, float, float, float],
    robot_time_s: float,
) -> dict[str, object]:
    """Classify named scene markers in the robot frame, without camera inference."""

    observations: list[dict[str, object]] = []
    w, x, y, z = robot_quat_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def polar(target_xyz: tuple[float, float, float]) -> tuple[float, float]:
        dx = target_xyz[0] - robot_xyz[0]
        dy = target_xyz[1] - robot_xyz[1]
        direction = math.atan2(dy, dx) - yaw
        return math.hypot(dx, dy), math.atan2(math.sin(direction), math.cos(direction))

    if bottle_xyz is not None:
        distance_m, bearing = polar(bottle_xyz)
        # A deliberately simple marker detector: 60 m range and a 90-degree
        # forward field. The simulator geom, not the user's words, supplies pose.
        if distance_m <= 60.0 and abs(bearing) <= math.pi / 4:
            observations.append(
                {
                    "object_ref": MILK_BOTTLE_GEOM,
                    "description": "bottle of milk",
                    "relative_direction": "in front of Chromie",
                    "distance_m": round(distance_m, 3),
                    # Provider-internal steering input. The MCP perception tool
                    # removes this backend-frame value before exposing the scene.
                    "bearing_rad": bearing,
                }
            )
    if user_xyz is not None:
        distance_m, bearing = polar(user_xyz)
        # A named scenario actor is tracked through the simulator scene API,
        # including when behind the robot. This is not camera perception.
        if distance_m <= 60.0:
            if abs(bearing) <= math.pi / 4:
                relative_direction = "in front of Chromie"
            elif abs(bearing) >= 3 * math.pi / 4:
                relative_direction = "behind Chromie"
            elif bearing > 0:
                relative_direction = "to Chromie's left"
            else:
                relative_direction = "to Chromie's right"
            observations.append(
                {
                    "object_ref": USER_BODY_GEOM,
                    "description": "user",
                    "relative_direction": relative_direction,
                    "distance_m": round(distance_m, 3),
                    "bearing_rad": bearing,
                }
            )
    for object_ref in WATER_BOTTLE_GEOMS:
        target_xyz = (water_bottles_xyz or {}).get(object_ref)
        if target_xyz is None:
            continue
        distance_m, bearing = polar(target_xyz)
        # Scenario markers are available all around the robot. This is a
        # simulated scene read, not a camera detection or a visual claim.
        if distance_m > 60.0:
            continue
        if abs(bearing) <= math.pi / 4:
            relative_direction = "in front of Chromie"
        elif abs(bearing) >= 3 * math.pi / 4:
            relative_direction = "behind Chromie"
        elif bearing > 0:
            relative_direction = "to Chromie's left"
        else:
            relative_direction = "to Chromie's right"
        observations.append(
            {
                "object_ref": object_ref,
                "description": "bottle of water",
                "relative_direction": relative_direction,
                "distance_m": round(distance_m, 3),
                "bearing_rad": bearing,
            }
        )
    return {
        "source_kind": "mujoco_scene_marker",
        "mocked_simulation": True,
        "robot_time_s": robot_time_s,
        "objects": observations,
    }


def build_default_world_scene_xml(base_xml: str) -> str:
    """Add only the fixed furniture; scenario-owned objects are added later."""

    if MILK_TABLE_GEOM in base_xml:
        raise ValueError("default world already exists in MuJoCo scene")
    insert_at = base_xml.rfind("</worldbody>")
    if insert_at < 0:
        raise ValueError("base MuJoCo XML does not contain </worldbody>")
    # The official robot starts facing +x, so +y is to its left. All fixture
    # geoms are non-contact and leave body dynamics unchanged.
    scene = (
        '    <!-- Simulation-only fixed table and left-side chairs. -->\n'
        '    <body name="soridormi_mock_milk_table" pos="10 0 0">\n'
        f'      <geom name="{MILK_TABLE_GEOM}" type="box" '
        'size="0.45 0.30 0.04" pos="0 0 0.70" '
        'rgba="0.64 0.43 0.25 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_table_leg_front_left" type="box" '
        'size="0.035 0.035 0.33" pos="-0.35 -0.20 0.33" '
        'rgba="0.48 0.30 0.17 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_table_leg_front_right" type="box" '
        'size="0.035 0.035 0.33" pos="-0.35 0.20 0.33" '
        'rgba="0.48 0.30 0.17 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_table_leg_back_left" type="box" '
        'size="0.035 0.035 0.33" pos="0.35 -0.20 0.33" '
        'rgba="0.48 0.30 0.17 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_table_leg_back_right" type="box" '
        'size="0.035 0.035 0.33" pos="0.35 0.20 0.33" '
        'rgba="0.48 0.30 0.17 1" contype="0" conaffinity="0"/>\n'
        '    </body>\n'
        f'{_left_chair_xml(1, 1.5)}'
        f'{_left_chair_xml(2, 3.0)}'
    )
    xml = base_xml[:insert_at] + scene + base_xml[insert_at:]
    ElementTree.fromstring(xml)
    return xml


def build_milk_bottle_scene_xml(base_xml: str) -> str:
    """Legacy self-contained scene used by explicit --milk-bottle launches."""

    if MILK_BOTTLE_GEOM in base_xml:
        raise ValueError("milk bottle scene already exists in MuJoCo scene")
    static_xml = build_default_world_scene_xml(base_xml)
    insert_at = static_xml.rfind("</worldbody>")
    scene = (
        f'    <body name="{SCENARIO_BOTTLE_BODY}" mocap="true" pos="10 0 0.86">\n'
        f'      <geom name="{MILK_BOTTLE_GEOM}" type="cylinder" '
        'size="0.045 0.12" pos="0 0 0" '
        'rgba="0.96 0.96 0.91 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_bottle_neck" type="cylinder" '
        'size="0.025 0.025" pos="0 0 0.145" '
        'rgba="0.96 0.96 0.91 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_bottle_cap" type="cylinder" '
        'size="0.029 0.015" pos="0 0 0.185" '
        'rgba="0.18 0.40 0.82 1" contype="0" conaffinity="0"/>\n'
        '    </body>\n'
    )
    xml = static_xml[:insert_at] + scene + static_xml[insert_at:]
    ElementTree.fromstring(xml)
    return xml


def generate_milk_bottle_scene(base_path: Path, output_path: Path) -> Path:
    source = base_path.read_text(encoding="utf-8")
    if output_path.parent.resolve() != base_path.parent.resolve():
        source = rewrite_relative_includes(source, base_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_milk_bottle_scene_xml(source), encoding="utf-8")
    return output_path


def generate_default_world_scene(base_path: Path, output_path: Path) -> Path:
    source = base_path.read_text(encoding="utf-8")
    if output_path.parent.resolve() != base_path.parent.resolve():
        source = rewrite_relative_includes(source, base_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_default_world_scene_xml(source), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add an optional table, milk bottle, and two chairs to a MuJoCo scene")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    generator = generate_default_world_scene if args.static_only else generate_milk_bottle_scene
    print(generator(args.base, args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
