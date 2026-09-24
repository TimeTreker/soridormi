"""Optional MuJoCo scene object for simulation perception tests.

The named geometry is a world object, not a detector result. The mock observer
reads its current MuJoCo position through the simulator API.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree

from .rough_ground_scene import rewrite_relative_includes

MILK_BOTTLE_GEOM = "soridormi_mock_milk_bottle"


def mock_milk_observation(
    *,
    bottle_xyz: tuple[float, float, float] | None,
    robot_xyz: tuple[float, float, float],
    robot_quat_wxyz: tuple[float, float, float, float],
    robot_time_s: float,
) -> dict[str, object]:
    """Classify one named scene geom in the robot frame, without camera inference."""

    observations: list[dict[str, object]] = []
    if bottle_xyz is not None:
        dx = bottle_xyz[0] - robot_xyz[0]
        dy = bottle_xyz[1] - robot_xyz[1]
        distance_m = math.hypot(dx, dy)
        w, x, y, z = robot_quat_wxyz
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        # A deliberately simple marker detector: 60 m range and a 90-degree
        # forward field. The simulator geom, not the user's words, supplies pose.
        if distance_m <= 60.0 and abs(bearing) <= math.pi / 4:
            observations.append(
                {
                    "object_ref": MILK_BOTTLE_GEOM,
                    "description": "bottle of milk",
                    "relative_direction": "in front of Chromie",
                    "distance_m": round(distance_m, 3),
                }
            )
    return {
        "source_kind": "mujoco_scene_marker",
        "mocked_simulation": True,
        "robot_time_s": robot_time_s,
        "objects": observations,
    }


def build_milk_bottle_scene_xml(base_xml: str) -> str:
    if MILK_BOTTLE_GEOM in base_xml:
        raise ValueError("milk bottle already exists in MuJoCo scene")
    insert_at = base_xml.rfind("</worldbody>")
    if insert_at < 0:
        raise ValueError("base MuJoCo XML does not contain </worldbody>")
    # The official robot starts facing +x. A visible, non-contact world geom
    # keeps the mock test from changing the robot's dynamics or actuator model.
    bottle = (
        '    <!-- Simulation-only milk bottle for scene observation. -->\n'
        f'    <geom name="{MILK_BOTTLE_GEOM}" type="cylinder" '
        'size="0.045 0.12" pos="50 0 0.12" '
        'rgba="0.96 0.96 0.91 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_bottle_neck" type="cylinder" '
        'size="0.025 0.025" pos="50 0 0.265" '
        'rgba="0.96 0.96 0.91 1" contype="0" conaffinity="0"/>\n'
        '    <geom name="soridormi_mock_milk_bottle_cap" type="cylinder" '
        'size="0.029 0.015" pos="50 0 0.305" '
        'rgba="0.18 0.40 0.82 1" contype="0" conaffinity="0"/>\n'
    )
    xml = base_xml[:insert_at] + bottle + base_xml[insert_at:]
    ElementTree.fromstring(xml)
    return xml


def generate_milk_bottle_scene(base_path: Path, output_path: Path) -> Path:
    source = base_path.read_text(encoding="utf-8")
    if output_path.parent.resolve() != base_path.parent.resolve():
        source = rewrite_relative_includes(source, base_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_milk_bottle_scene_xml(source), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add an optional milk bottle to a MuJoCo scene")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_milk_bottle_scene(args.base, args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
