from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

LEFT_EYE_NAME = "soridormi_left_eye_visual"
RIGHT_EYE_NAME = "soridormi_right_eye_visual"
LEFT_EYE_CLOSED_NAME = "soridormi_left_eye_closed_visual"
RIGHT_EYE_CLOSED_NAME = "soridormi_right_eye_closed_visual"
SOCIAL_EYE_FRAME_BODY_NAME = "soridormi_social_eye_frame"
SOCIAL_EYE_FRAME_ORIGIN_NAME = "soridormi_social_eye_frame_origin"
SOCIAL_EYE_FRAME_X_AXIS_NAME = "soridormi_social_eye_frame_x_axis"
SOCIAL_EYE_FRAME_Y_AXIS_NAME = "soridormi_social_eye_frame_y_axis"
SOCIAL_EYE_FRAME_Z_AXIS_NAME = "soridormi_social_eye_frame_z_axis"
SOCIAL_EYE_GEOM_NAMES = (
    LEFT_EYE_NAME,
    RIGHT_EYE_NAME,
    LEFT_EYE_CLOSED_NAME,
    RIGHT_EYE_CLOSED_NAME,
)
SOCIAL_EYE_COMMENT = "                <!-- Soridormi generated social eye visuals. -->\n"
VISUAL_ARM_COMMENT = "        <!-- Soridormi generated visual-only arms. -->\n"
VISUAL_ARM_ASSET_OPEN = "  <asset>  <!-- Soridormi generated visual-arm shells. -->\n"
VISUAL_ARM_ASSET_CLOSE = "  </asset>  <!-- end Soridormi visual-arm shells -->\n"
DEFAULT_ROBOT_PREFIX = "soridormi_social_eyes_"

VISUAL_ARM_POSES = (
    "rest",
    "reach",
    "hold",
    "place",
    "wave_up",
    "wave_out",
    "celebrate",
    "welcome_open",
    "welcome_close",
)
VISUAL_ARM_SIDES = ("left", "right")
VISUAL_ARM_FINGERS = ("index", "middle", "ring", "little")
VISUAL_ARM_FINGER_COMPONENTS = (
    "mcp",
    "proximal",
    "pip",
    "middle",
    "dip",
    "distal",
)
VISUAL_ARM_THUMB_COMPONENTS = (
    "cmc",
    "metacarpal",
    "mcp",
    "proximal",
    "ip",
    "distal",
)
VISUAL_ARM_COMPONENTS = (
    (
        "upper",
        "elbow",
        "elbow_axle",
        "forearm",
        "cuff",
        "wrist",
        "palm",
    )
    + tuple(
        f"{finger}_{component}"
        for finger in VISUAL_ARM_FINGERS
        for component in VISUAL_ARM_FINGER_COMPONENTS
    )
    + tuple(f"thumb_{component}" for component in VISUAL_ARM_THUMB_COMPONENTS)
)
LEGACY_VISUAL_ARM_COMPONENTS = (
    "hand",
    "thumb",
    *VISUAL_ARM_FINGERS,
    "index_finger",
    "middle_finger",
    "ring_finger",
    "pinky_finger",
)


def visual_arm_geom_name(side: str, pose: str, component: str) -> str:
    return f"soridormi_{side}_arm_{pose}_{component}_visual"


VISUAL_ARM_SHOULDER_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_SHOULDER_MOUNT_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_mount_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_SHOULDER_AXLE_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_axle_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_POSE_GEOM_NAMES = tuple(
    visual_arm_geom_name(side, pose, component)
    for side in VISUAL_ARM_SIDES
    for pose in VISUAL_ARM_POSES
    for component in VISUAL_ARM_COMPONENTS
)
VISUAL_ARM_GEOM_NAMES = (
    VISUAL_ARM_SHOULDER_MOUNT_NAMES
    + VISUAL_ARM_SHOULDER_NAMES
    + VISUAL_ARM_SHOULDER_AXLE_NAMES
    + VISUAL_ARM_POSE_GEOM_NAMES
)

VISUAL_LEG_BODY_NAMES = {
    "left": "knee_and_ankle_assembly_2",
    "right": "knee_and_ankle_assembly_4",
}
VISUAL_LEG_COMPONENTS = ("shin_shell", "ankle")
LEGACY_VISUAL_LEG_COMPONENTS = ("thigh_shell", "knee")


def visual_leg_geom_name(side: str, component: str) -> str:
    return f"soridormi_{side}_leg_{component}_visual"


VISUAL_LEG_GEOM_NAMES = tuple(
    visual_leg_geom_name(side, component)
    for side in VISUAL_ARM_SIDES
    for component in VISUAL_LEG_COMPONENTS
)


@dataclass(frozen=True)
class SocialEyeConfig:
    eye_x_m: float = 0.01
    eye_y_offset_m: float = 0.04
    eye_z_m: float = -0.06
    eye_radius_m: float = 0.02
    eye_depth_m: float = 0.002
    eye_quat: str = "0.707107 0 0.707107 0"
    debug_frame: bool = False
    eye_rgba: str = "0.015 0.018 0.022 1"
    closed_eye_rgba: str = "0.015 0.018 0.022 0"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualArmConfig:
    """Cosmetic arm styling matched to the Open Duck shell.

    Segments are chamfered, slightly tapering shells in the trunk's grey with
    orange hinge drums like the hip servos, so the arms read as part of the
    same chunky body rather than thin tubes with human hands.
    """

    shoulder_x_m: float = -0.02
    shoulder_y_offset_m: float = 0.092
    shoulder_z_m: float = 0.094
    shoulder_mount_y_offset_m: float = 0.055
    shoulder_housing_size_m: float = 0.044
    upper_arm_length_m: float = 0.058
    forearm_length_m: float = 0.052
    upper_arm_width_m: float = 0.04
    upper_arm_depth_m: float = 0.036
    forearm_width_m: float = 0.036
    forearm_depth_m: float = 0.032
    shell_taper: float = 0.86
    shell_chamfer_m: float = 0.009
    shoulder_joint_radius_m: float = 0.019
    shoulder_joint_depth_m: float = 0.009
    elbow_joint_radius_m: float = 0.0145
    elbow_joint_half_depth_m: float = 0.0195
    joint_axle_radius_m: float = 0.0072
    cuff_radius_m: float = 0.0165
    wrist_radius_m: float = 0.0115
    palm_thickness_m: float = 0.018
    palm_width_m: float = 0.034
    palm_length_m: float = 0.022
    finger_radius_m: float = 0.0054
    finger_length_m: float = 0.018
    arm_rgba: str = "0.917647 0.917647 0.917647 1"
    hand_rgba: str = "0.917647 0.917647 0.917647 1"
    joint_rgba: str = "0.909804 0.572549 0.164706 1"
    joint_axle_rgba: str = "0.223529 0.219608 0.219608 1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualLegConfig:
    link_length_m: float = 0.07865
    shin_radius_m: float = 0.0135
    ankle_radius_m: float = 0.0125
    shell_rgba: str = "0.917647 0.917647 0.917647 1"
    joint_rgba: str = "0.909804 0.572549 0.164706 1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SocialEyeResult:
    output_path: str
    base_path: str
    robot_output_path: str
    robot_base_path: str
    eye_count: int
    config: dict[str, Any]
    arm_geom_count: int = 0
    arm_config: dict[str, Any] | None = None
    leg_shell_geom_count: int = 0
    leg_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _format_xyz(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _arm_pose_points(
    side: str,
    pose: str,
    config: VisualArmConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    sign = 1.0 if side == "left" else -1.0
    shoulder = (
        config.shoulder_x_m,
        sign * config.shoulder_y_offset_m,
        config.shoulder_z_m,
    )
    points = {
        "rest": (
            (0.03, sign * 0.115, 0.07),
            (0.07, sign * 0.115, 0.03),
        ),
        "reach": (
            (0.04, sign * 0.125, 0.08),
            (0.11, sign * 0.105, 0.06),
        ),
        "hold": (
            (0.035, sign * 0.105, 0.08),
            (0.09, sign * 0.01, 0.07),
        ),
        "place": (
            (0.035, sign * 0.135, 0.075),
            (0.12, sign * 0.08, 0.055),
        ),
        "wave_up": (
            (0.002, sign * 0.145, 0.148),
            (0.018, sign * 0.19, 0.205),
        ),
        "wave_out": (
            (0.002, sign * 0.145, 0.148),
            (0.018, sign * 0.205, 0.18),
        ),
        "celebrate": (
            (-0.004, sign * 0.145, 0.16),
            (-0.004, sign * 0.188, 0.216),
        ),
        "welcome_open": (
            (0.025, sign * 0.14, 0.1),
            (0.1, sign * 0.175, 0.112),
        ),
        "welcome_close": (
            (0.04, sign * 0.106, 0.1),
            (0.105, sign * 0.02, 0.1),
        ),
    }
    elbow_hint, wrist_hint = points[pose]
    upper_direction = _unit_vector(_vector_subtract(elbow_hint, shoulder))
    forearm_direction = _unit_vector(_vector_subtract(wrist_hint, elbow_hint))
    elbow = _vector_add(
        shoulder,
        _vector_scale(upper_direction, config.upper_arm_length_m),
    )
    wrist = _vector_add(
        elbow,
        _vector_scale(forearm_direction, config.forearm_length_m),
    )
    return shoulder, elbow, wrist


def _vector_add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _vector_scale(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(value * scale for value in vector)  # type: ignore[return-value]


def _vector_cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _quat_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a MuJoCo wxyz quaternion for orthonormal local axis columns."""
    matrix = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        quat = (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    else:
        dominant = max(range(3), key=lambda index: matrix[index][index])
        if dominant == 0:
            scale = (1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) ** 0.5 * 2.0
            quat = (
                (matrix[2][1] - matrix[1][2]) / scale,
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
            )
        elif dominant == 1:
            scale = (1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) ** 0.5 * 2.0
            quat = (
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
            )
        else:
            scale = (1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) ** 0.5 * 2.0
            quat = (
                (matrix[1][0] - matrix[0][1]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
            )
    length = sum(value * value for value in quat) ** 0.5
    return tuple(value / length for value in quat)  # type: ignore[return-value]


def _unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sum(value * value for value in vector) ** 0.5
    if length <= 1e-9:
        raise ValueError("visual arm segment has zero length")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _perpendicular_unit(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    projection = sum(value * direction for value, direction in zip(vector, axis))
    perpendicular = _vector_subtract(vector, _vector_scale(axis, projection))
    try:
        return _unit_vector(perpendicular)
    except ValueError:
        fallback = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 0.0, 1.0)
        fallback_projection = sum(value * direction for value, direction in zip(fallback, axis))
        return _unit_vector(_vector_subtract(fallback, _vector_scale(axis, fallback_projection)))


def _arm_hinge_axis(
    upper_direction: tuple[float, float, float],
    forearm_direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the mechanical elbow axis perpendicular to both arm segments."""
    try:
        return _unit_vector(_vector_cross(upper_direction, forearm_direction))
    except ValueError:
        return _perpendicular_unit((1.0, 0.0, 0.0), upper_direction)


def _arm_hand_direction(
    pose: str,
    elbow: tuple[float, float, float],
    wrist: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the intentional finger direction for a fixed display pose."""
    if pose in {
        "wave_up",
        "wave_out",
        "celebrate",
        "welcome_open",
        "welcome_close",
        "hold",
    }:
        return (0.0, 0.0, 1.0)
    if pose == "reach":
        return _unit_vector((1.0, 0.0, -0.08))
    if pose == "place":
        return _unit_vector((1.0, 0.0, -0.15))
    return _unit_vector(_vector_subtract(wrist, elbow))


def _finger_bend_weights(pose: str) -> tuple[float, float]:
    """Return modest PIP/DIP curl while keeping social-display hands readable."""
    if pose == "hold":
        return (0.35, 0.7)
    if pose in {"rest", "welcome_close"}:
        return (0.16, 0.34)
    return (0.05, 0.12)


def _rgba_with_alpha(rgba: str, alpha: str) -> str:
    return " ".join(rgba.split()[:3] + [alpha])


def _visual_geom_xml(name: str, geom_type: str, placement: str, rgba: str) -> str:
    return (
        f'        <geom name="{name}" type="{geom_type}" class="visual" contype="0" '
        f'conaffinity="0" {placement} rgba="{rgba}"/>\n'
    )


def _chamfered_shell_vertices(
    length: float,
    width: float,
    depth: float,
    chamfer: float,
    taper: float,
) -> list[tuple[float, float, float]]:
    """Return hull points of an octagonal shell along +z that tapers toward +z.

    The cross-section is a box with chamfered edges and both ends are bevelled,
    matching the faceted printed shells of the Open Duck trunk and thighs.
    """

    def ring(
        half_x: float, half_y: float, cut: float, z: float
    ) -> list[tuple[float, float, float]]:
        points = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                points.append((sx * half_x, sy * (half_y - cut), z))
                points.append((sx * (half_x - cut), sy * half_y, z))
        return points

    half_length = 0.5 * length - chamfer
    vertices: list[tuple[float, float, float]] = []
    for z, scale, inset in (
        (-half_length - chamfer, 1.0, chamfer),
        (-half_length, 1.0, 0.0),
        (half_length, taper, 0.0),
        (half_length + chamfer, taper, chamfer),
    ):
        half_x = 0.5 * depth * scale - inset
        half_y = 0.5 * width * scale - inset
        vertices.extend(ring(half_x, half_y, min(chamfer, half_x, half_y) * 0.9, z))
    return vertices


def visual_arm_mesh_name(part: str) -> str:
    return f"soridormi_arm_{part}_shell_mesh"


VISUAL_ARM_MESH_PARTS = ("shoulder", "upper", "forearm", "palm")
VISUAL_ARM_MESH_NAMES = tuple(visual_arm_mesh_name(part) for part in VISUAL_ARM_MESH_PARTS)


def build_visual_arm_mesh_assets(config: VisualArmConfig | None = None) -> str:
    cfg = config or VisualArmConfig()
    housing = cfg.shoulder_housing_size_m
    housing_depth = (
        cfg.shoulder_y_offset_m - cfg.shoulder_joint_depth_m - cfg.shoulder_mount_y_offset_m
    )
    shells = {
        "shoulder": _chamfered_shell_vertices(
            housing_depth + cfg.shell_chamfer_m, housing, housing, cfg.shell_chamfer_m, 0.92
        ),
        "upper": _chamfered_shell_vertices(
            cfg.upper_arm_length_m - 0.004,
            cfg.upper_arm_width_m,
            cfg.upper_arm_depth_m,
            cfg.shell_chamfer_m,
            cfg.shell_taper,
        ),
        "forearm": _chamfered_shell_vertices(
            cfg.forearm_length_m - 0.012,
            cfg.forearm_width_m,
            cfg.forearm_depth_m,
            cfg.shell_chamfer_m,
            cfg.shell_taper,
        ),
        "palm": _chamfered_shell_vertices(
            cfg.palm_length_m,
            cfg.palm_width_m,
            cfg.palm_thickness_m,
            cfg.shell_chamfer_m * 0.7,
            0.94,
        ),
    }
    chunks = [VISUAL_ARM_ASSET_OPEN]
    for part, vertices in shells.items():
        flat = " ".join(_format_xyz(vertex) for vertex in vertices)
        chunks.append(f'    <mesh name="{visual_arm_mesh_name(part)}" vertex="{flat}"/>\n')
    chunks.append(VISUAL_ARM_ASSET_CLOSE)
    return "".join(chunks)


def _shell_xml(
    name: str,
    part: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    hinge: tuple[float, float, float],
    rgba: str,
) -> str:
    """Place a centred shell mesh between two points with its width on ``hinge``."""
    axis = _unit_vector(_vector_subtract(end, start))
    width_axis = _perpendicular_unit(hinge, axis)
    quat = _quat_from_axes(_vector_cross(width_axis, axis), width_axis, axis)
    center = _vector_scale(_vector_add(start, end), 0.5)
    return _visual_geom_xml(
        name,
        "mesh",
        f'mesh="{visual_arm_mesh_name(part)}" pos="{_format_xyz(center)}" '
        f'quat="{_format_xyz(quat)}"',
        rgba,
    )


def _cylinder_xml(
    name: str,
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    half_depth: float,
    radius: float,
    rgba: str,
    geom_type: str = "cylinder",
) -> str:
    offset = _vector_scale(axis, half_depth)
    return _visual_geom_xml(
        name,
        geom_type,
        f'fromto="{_format_xyz(_vector_subtract(center, offset))} '
        f'{_format_xyz(_vector_add(center, offset))}" size="{_format_float(radius)}"',
        rgba,
    )


def _visual_arm_pose_xml(side: str, pose: str, config: VisualArmConfig) -> str:
    shoulder, elbow, wrist = _arm_pose_points(side, pose, config)
    alpha = "1" if pose == "rest" else "0"
    arm_rgba = _rgba_with_alpha(config.arm_rgba, alpha)
    hand_rgba = _rgba_with_alpha(config.hand_rgba, alpha)
    joint_rgba = _rgba_with_alpha(config.joint_rgba, alpha)
    joint_axle_rgba = _rgba_with_alpha(config.joint_axle_rgba, alpha)

    def name(component: str) -> str:
        return visual_arm_geom_name(side, pose, component)

    hand_forward = _arm_hand_direction(pose, elbow, wrist)
    sign = 1.0 if side == "left" else -1.0
    inward = _perpendicular_unit((0.0, -sign, 0.0), hand_forward)
    if pose == "rest":
        # Relaxed hands turn their palms toward the body, thumbs up.
        medial = _perpendicular_unit((0.0, 0.0, 1.0), hand_forward)
        curl_direction = inward
    else:
        medial = inward
        curl_direction = _unit_vector(_vector_scale(_vector_cross(medial, hand_forward), sign))
    forearm_direction = _unit_vector(_vector_subtract(wrist, elbow))
    upper_direction = _unit_vector(_vector_subtract(elbow, shoulder))
    elbow_axis = _arm_hinge_axis(upper_direction, forearm_direction)
    cuff_center = _vector_subtract(wrist, _vector_scale(forearm_direction, 0.0075))
    forearm_shell_end = _vector_subtract(wrist, _vector_scale(forearm_direction, 0.012))
    wrist_center = _vector_add(wrist, _vector_scale(hand_forward, 0.0005))
    palm_start = _vector_add(wrist, _vector_scale(hand_forward, 0.003))
    palm_end = _vector_add(palm_start, _vector_scale(hand_forward, config.palm_length_m))

    xml = (
        _shell_xml(name("upper"), "upper", shoulder, elbow, elbow_axis, arm_rgba)
        + _cylinder_xml(
            name("elbow"),
            elbow,
            elbow_axis,
            config.elbow_joint_half_depth_m,
            config.elbow_joint_radius_m,
            joint_rgba,
        )
        + _cylinder_xml(
            name("elbow_axle"),
            elbow,
            elbow_axis,
            config.elbow_joint_half_depth_m + 0.0012,
            config.joint_axle_radius_m,
            joint_axle_rgba,
        )
        + _shell_xml(name("forearm"), "forearm", elbow, forearm_shell_end, elbow_axis, arm_rgba)
        + _cylinder_xml(
            name("cuff"),
            cuff_center,
            forearm_direction,
            0.0028,
            config.cuff_radius_m,
            joint_axle_rgba,
        )
        + _cylinder_xml(
            name("wrist"),
            wrist_center,
            hand_forward,
            0.0035,
            config.wrist_radius_m,
            joint_rgba,
        )
        + _shell_xml(name("palm"), "palm", palm_start, palm_end, medial, hand_rgba)
    )

    # Short, chunky, nearly parallel fingers keep the hand in the same toy
    # scale as the body. Knuckle spheres share the finger colour and radius so
    # each digit reads as one rounded piece rather than a string of beads.
    segment_scales = (1.0, 0.94, 0.88)
    finger_offsets = {"index": 0.0114, "middle": 0.0038, "ring": -0.0038, "little": -0.0114}
    finger_lengths = {"index": 0.95, "middle": 1.0, "ring": 0.95, "little": 0.82}
    middle_bend, distal_bend = _finger_bend_weights(pose)
    for finger, offset in finger_offsets.items():
        length = config.finger_length_m * finger_lengths[finger]
        mcp = _vector_add(
            _vector_subtract(palm_end, _vector_scale(hand_forward, config.finger_radius_m)),
            _vector_scale(medial, offset),
        )
        middle_direction = _unit_vector(
            _vector_add(hand_forward, _vector_scale(curl_direction, middle_bend))
        )
        distal_direction = _unit_vector(
            _vector_add(hand_forward, _vector_scale(curl_direction, distal_bend))
        )
        pip = _vector_add(mcp, _vector_scale(hand_forward, length * 0.45))
        dip = _vector_add(pip, _vector_scale(middle_direction, length * 0.32))
        tip = _vector_add(dip, _vector_scale(distal_direction, length * 0.23))
        for scale, (joint, position) in zip(
            segment_scales, (("mcp", mcp), ("pip", pip), ("dip", dip))
        ):
            xml += _visual_geom_xml(
                name(f"{finger}_{joint}"),
                "sphere",
                f'pos="{_format_xyz(position)}" '
                f'size="{_format_float(config.finger_radius_m * scale)}"',
                hand_rgba,
            )
        for scale, (segment, start, end) in zip(
            segment_scales,
            (("proximal", mcp, pip), ("middle", pip, dip), ("distal", dip, tip)),
        ):
            xml += _visual_geom_xml(
                name(f"{finger}_{segment}"),
                "capsule",
                f'fromto="{_format_xyz(start)} {_format_xyz(end)}" '
                f'size="{_format_float(config.finger_radius_m * scale)}"',
                hand_rgba,
            )

    thumb_cmc = _vector_add(
        _vector_add(palm_start, _vector_scale(hand_forward, config.palm_length_m * 0.35)),
        _vector_scale(medial, config.palm_width_m * 0.42),
    )
    thumb_directions = (
        _unit_vector(_vector_add(_vector_scale(hand_forward, 0.5), medial)),
        _unit_vector(
            _vector_add(
                _vector_add(hand_forward, _vector_scale(medial, 0.55)),
                _vector_scale(curl_direction, 0.15 + middle_bend * 0.2),
            )
        ),
        _unit_vector(
            _vector_add(
                _vector_add(hand_forward, _vector_scale(medial, 0.3)),
                _vector_scale(curl_direction, 0.25 + distal_bend * 0.25),
            )
        ),
    )
    thumb_mcp = _vector_add(thumb_cmc, _vector_scale(thumb_directions[0], 0.006))
    thumb_ip = _vector_add(thumb_mcp, _vector_scale(thumb_directions[1], 0.0065))
    thumb_tip = _vector_add(thumb_ip, _vector_scale(thumb_directions[2], 0.0055))
    thumb_radius = config.finger_radius_m * 1.05
    for scale, (joint, position) in zip(
        segment_scales, (("cmc", thumb_cmc), ("mcp", thumb_mcp), ("ip", thumb_ip))
    ):
        xml += _visual_geom_xml(
            name(f"thumb_{joint}"),
            "sphere",
            f'pos="{_format_xyz(position)}" size="{_format_float(thumb_radius * scale)}"',
            hand_rgba,
        )
    for scale, (segment, start, end) in zip(
        segment_scales,
        (
            ("metacarpal", thumb_cmc, thumb_mcp),
            ("proximal", thumb_mcp, thumb_ip),
            ("distal", thumb_ip, thumb_tip),
        ),
    ):
        xml += _visual_geom_xml(
            name(f"thumb_{segment}"),
            "capsule",
            f'fromto="{_format_xyz(start)} {_format_xyz(end)}" '
            f'size="{_format_float(thumb_radius * scale)}"',
            hand_rgba,
        )
    return xml


def build_visual_arm_geoms(config: VisualArmConfig | None = None) -> str:
    cfg = config or VisualArmConfig()
    chunks = [VISUAL_ARM_COMMENT]
    for side, mount_name, shoulder_name, shoulder_axle_name in zip(
        VISUAL_ARM_SIDES,
        VISUAL_ARM_SHOULDER_MOUNT_NAMES,
        VISUAL_ARM_SHOULDER_NAMES,
        VISUAL_ARM_SHOULDER_AXLE_NAMES,
    ):
        sign = 1.0 if side == "left" else -1.0
        shoulder, _elbow, _hand = _arm_pose_points(side, "rest", cfg)
        outward = (0.0, sign, 0.0)
        # A chamfered servo housing sits flush on the trunk side; the orange
        # shoulder drum on its outer face echoes the hip servos below it.
        drum_outer = cfg.shoulder_y_offset_m
        drum_inner = drum_outer - cfg.shoulder_joint_depth_m
        housing_start = (cfg.shoulder_x_m, sign * cfg.shoulder_mount_y_offset_m, cfg.shoulder_z_m)
        housing_end = (
            cfg.shoulder_x_m,
            sign * (drum_inner + cfg.shell_chamfer_m),
            cfg.shoulder_z_m,
        )
        drum_center = (cfg.shoulder_x_m, sign * 0.5 * (drum_inner + drum_outer), cfg.shoulder_z_m)
        chunks.append(
            _shell_xml(
                mount_name, "shoulder", housing_start, housing_end, (1.0, 0.0, 0.0), cfg.arm_rgba
            )
        )
        chunks.append(
            _cylinder_xml(
                shoulder_name,
                drum_center,
                outward,
                0.5 * cfg.shoulder_joint_depth_m,
                cfg.shoulder_joint_radius_m,
                cfg.joint_rgba,
            )
        )
        chunks.append(
            _cylinder_xml(
                shoulder_axle_name,
                shoulder,
                outward,
                0.0018,
                cfg.joint_axle_radius_m,
                cfg.joint_axle_rgba,
            )
        )
        for pose in VISUAL_ARM_POSES:
            chunks.append(_visual_arm_pose_xml(side, pose, cfg))
    return "".join(chunks)


def _visual_leg_body_geoms(
    side: str,
    config: VisualLegConfig,
) -> str:
    shin_start = (0.0, -0.009, 0.0)
    shin_end = (0.0, -(config.link_length_m - 0.014), 0.0)
    ankle = (0.0, -config.link_length_m, 0.0001)
    return (
        f'                <geom name="{visual_leg_geom_name(side, "shin_shell")}" '
        f'type="capsule" class="visual" contype="0" conaffinity="0" '
        f'fromto="{_format_xyz(shin_start)} {_format_xyz(shin_end)}" '
        f'size="{_format_float(config.shin_radius_m)}" rgba="{config.shell_rgba}"/>\n'
        f'                <geom name="{visual_leg_geom_name(side, "ankle")}" '
        f'type="sphere" class="visual" contype="0" conaffinity="0" '
        f'pos="{_format_xyz(ankle)}" size="{_format_float(config.ankle_radius_m)}" '
        f'rgba="{config.joint_rgba}"/>\n'
    )


def _eye_geom_xml(name: str, *, y: float, config: SocialEyeConfig) -> str:
    pos = " ".join(
        [
            _format_float(config.eye_x_m),
            _format_float(y),
            _format_float(config.eye_z_m),
        ]
    )
    size = " ".join(
        [
            _format_float(max(config.eye_depth_m, 0.001)),
            _format_float(config.eye_radius_m),
            _format_float(config.eye_radius_m),
        ]
    )
    return (
        f'                <geom name="{name}" type="ellipsoid" class="visual" '
        f'pos="{pos}" quat="{config.eye_quat}" size="{size}" rgba="{config.eye_rgba}"/>\n'
    )


def _closed_eye_geom_xml(name: str, *, y: float, config: SocialEyeConfig) -> str:
    pos = " ".join(
        [
            _format_float(config.eye_x_m + 0.001),
            _format_float(y),
            _format_float(config.eye_z_m),
        ]
    )
    size = " ".join(
        [
            _format_float(max(config.eye_radius_m * 0.12, 0.001)),
            _format_float(max(config.eye_radius_m * 0.85, 0.002)),
            _format_float(max(config.eye_radius_m * 0.1, 0.001)),
        ]
    )
    return (
        f'                <geom name="{name}" type="ellipsoid" class="visual" '
        f'pos="{pos}" quat="{config.eye_quat}" size="{size}" rgba="{config.closed_eye_rgba}"/>\n'
    )


def _social_eye_frame_xml(config: SocialEyeConfig) -> str:
    pos = " ".join(
        [
            _format_float(config.eye_x_m),
            "0",
            _format_float(config.eye_z_m),
        ]
    )
    return (
        f'                <body name="{SOCIAL_EYE_FRAME_BODY_NAME}" pos="{pos}" quat="{config.eye_quat}">\n'
        f'                  <site name="{SOCIAL_EYE_FRAME_ORIGIN_NAME}" type="sphere" '
        'size="0.005" rgba="1 1 1 1"/>\n'
        f'                  <geom name="{SOCIAL_EYE_FRAME_X_AXIS_NAME}" type="capsule" class="visual" '
        'fromto="0 0 0 0.06 0 0" size="0.002" rgba="1 0 0 1"/>\n'
        f'                  <geom name="{SOCIAL_EYE_FRAME_Y_AXIS_NAME}" type="capsule" class="visual" '
        'fromto="0 0 0 0 0.06 0" size="0.002" rgba="0 1 0 1"/>\n'
        f'                  <geom name="{SOCIAL_EYE_FRAME_Z_AXIS_NAME}" type="capsule" class="visual" '
        'fromto="0 0 0 0 0 0.06" size="0.002" rgba="0 0.25 1 1"/>\n'
        "                </body>\n"
    )


def build_social_eye_geoms(
    config: SocialEyeConfig | None = None,
    *,
    names: tuple[str, ...] = SOCIAL_EYE_GEOM_NAMES,
) -> str:
    cfg = config or SocialEyeConfig()
    chunks = [SOCIAL_EYE_COMMENT]
    if LEFT_EYE_NAME in names:
        chunks.append(_eye_geom_xml(LEFT_EYE_NAME, y=cfg.eye_y_offset_m, config=cfg))
    if RIGHT_EYE_NAME in names:
        chunks.append(_eye_geom_xml(RIGHT_EYE_NAME, y=-cfg.eye_y_offset_m, config=cfg))
    if LEFT_EYE_CLOSED_NAME in names:
        chunks.append(_closed_eye_geom_xml(LEFT_EYE_CLOSED_NAME, y=cfg.eye_y_offset_m, config=cfg))
    if RIGHT_EYE_CLOSED_NAME in names:
        chunks.append(
            _closed_eye_geom_xml(RIGHT_EYE_CLOSED_NAME, y=-cfg.eye_y_offset_m, config=cfg)
        )
    if cfg.debug_frame:
        chunks.append(_social_eye_frame_xml(cfg))
    return "".join(chunks)


def _remove_existing_social_eye_geoms(robot_xml: str) -> str:
    xml = robot_xml.replace(SOCIAL_EYE_COMMENT, "")
    xml = re.sub(
        rf"^[ \t]*<body\b[^>]*\bname=\"{re.escape(SOCIAL_EYE_FRAME_BODY_NAME)}\"[^>]*>.*?^[ \t]*</body>\n?",
        "",
        xml,
        flags=re.MULTILINE | re.DOTALL,
    )
    for name in SOCIAL_EYE_GEOM_NAMES:
        xml = re.sub(
            rf"^[ \t]*<geom\b[^>]*\bname=\"{re.escape(name)}\"[^>]*/>\n?",
            "",
            xml,
            flags=re.MULTILINE,
        )
    return xml


def _remove_existing_visual_arm_geoms(robot_xml: str) -> str:
    xml = robot_xml.replace(VISUAL_ARM_COMMENT, "")
    xml = re.sub(
        rf"{re.escape(VISUAL_ARM_ASSET_OPEN)}.*?{re.escape(VISUAL_ARM_ASSET_CLOSE)}",
        "",
        xml,
        flags=re.MULTILINE | re.DOTALL,
    )
    legacy_names = tuple(
        visual_arm_geom_name(side, pose, component)
        for side in VISUAL_ARM_SIDES
        for pose in VISUAL_ARM_POSES
        for component in LEGACY_VISUAL_ARM_COMPONENTS
    )
    for name in (*VISUAL_ARM_GEOM_NAMES, *legacy_names):
        xml = re.sub(
            rf"^[ \t]*<geom\b[^>]*\bname=\"{re.escape(name)}\"[^>]*/>\n?",
            "",
            xml,
            flags=re.MULTILINE,
        )
    return xml


def _remove_existing_visual_leg_geoms(robot_xml: str) -> str:
    xml = robot_xml
    legacy_names = tuple(
        visual_leg_geom_name(side, component)
        for side in VISUAL_ARM_SIDES
        for component in LEGACY_VISUAL_LEG_COMPONENTS
    )
    for name in (*VISUAL_LEG_GEOM_NAMES, *legacy_names):
        xml = re.sub(
            rf"^[ \t]*<geom\b[^>]*\bname=\"{re.escape(name)}\"[^>]*/>\n?",
            "",
            xml,
            flags=re.MULTILINE,
        )
    return xml


def build_visual_leg_shell_robot_xml(
    robot_xml: str,
    config: VisualLegConfig | None = None,
) -> str:
    cfg = config or VisualLegConfig()
    xml = _remove_existing_visual_leg_geoms(robot_xml)
    for side, shin_body in VISUAL_LEG_BODY_NAMES.items():
        insertion = _visual_leg_body_geoms(side, cfg)
        pattern = re.compile(rf'(<body\b[^>]*\bname=["\']{re.escape(shin_body)}["\'][^>]*>\n?)')
        xml, count = pattern.subn(
            lambda match, insertion=insertion: match.group(1) + insertion,
            xml,
            count=1,
        )
        if count != 1:
            raise ValueError(f"robot XML does not contain visual leg target body {shin_body!r}")
    return xml


def build_social_eye_robot_xml(robot_xml: str, config: SocialEyeConfig | None = None) -> str:
    robot_xml = _remove_existing_social_eye_geoms(robot_xml)
    insertion = build_social_eye_geoms(config)
    markers = [
        "                <!-- Part left_eye -->\n",
        "                <!-- Frame head -->\n",
    ]
    insert_at = -1
    for marker in markers:
        marker_at = robot_xml.find(marker)
        if marker_at >= 0:
            insert_at = marker_at + len(marker)
            break
    if insert_at < 0:
        raise ValueError("robot MuJoCo XML does not contain a head eye insertion marker")

    xml = robot_xml[:insert_at] + insertion + robot_xml[insert_at:]
    ElementTree.fromstring(xml)
    return xml


def build_visual_arm_robot_xml(
    robot_xml: str,
    config: VisualArmConfig | None = None,
) -> str:
    robot_xml = _remove_existing_visual_arm_geoms(robot_xml)
    marker = "        <!-- Frame trunk -->\n"
    insert_at = robot_xml.find(marker)
    if insert_at < 0:
        raise ValueError("robot MuJoCo XML does not contain a trunk visual insertion marker")
    xml = robot_xml[:insert_at] + build_visual_arm_geoms(config) + robot_xml[insert_at:]
    worldbody_at = xml.find("<worldbody")
    if worldbody_at < 0:
        raise ValueError("robot MuJoCo XML does not contain a worldbody")
    line_start = xml.rfind("\n", 0, worldbody_at) + 1
    if xml[line_start:worldbody_at].strip():
        line_start = worldbody_at
    xml = xml[:line_start] + build_visual_arm_mesh_assets(config) + xml[line_start:]
    ElementTree.fromstring(xml)
    return xml


def _find_open_duck_robot_include(scene_xml: str) -> tuple[str, str, str]:
    for match in re.finditer(r'(<include\b[^>]*\bfile=")([^"]+)(")', scene_xml):
        include_path = match.group(2)
        name = Path(include_path).name
        if name in {"open_duck_mini_v2.xml", "open_duck_mini_v2_backlash.xml"} or name.startswith(
            DEFAULT_ROBOT_PREFIX
        ):
            return match.group(1), include_path, match.group(3)
    raise ValueError("base MuJoCo scene does not include an Open Duck Mini v2 robot XML")


def _resolve_include_path(scene_path: Path, include_path: str) -> Path:
    include = Path(include_path)
    if include.is_absolute():
        return include
    return scene_path.parent / include


def _include_reference(scene_output_path: Path, robot_output_path: Path) -> str:
    if scene_output_path.parent.resolve() == robot_output_path.parent.resolve():
        return robot_output_path.name
    return robot_output_path.resolve().as_posix()


def build_social_eye_scene_xml(
    scene_xml: str,
    *,
    scene_path: str | Path,
    robot_output_path: str | Path,
) -> str:
    prefix, include_path, suffix = _find_open_duck_robot_include(scene_xml)
    output = Path(robot_output_path)
    include_ref = _include_reference(Path(scene_path), output)
    return scene_xml.replace(f"{prefix}{include_path}{suffix}", f"{prefix}{include_ref}{suffix}", 1)


def generate_social_eye_scene(
    base_path: str | Path,
    output_path: str | Path,
    *,
    robot_output_path: str | Path | None = None,
    config: SocialEyeConfig | None = None,
    arm_config: VisualArmConfig | None = None,
    leg_config: VisualLegConfig | None = None,
    include_eyes: bool = True,
) -> SocialEyeResult:
    cfg = config or SocialEyeConfig()
    base = Path(base_path)
    output = Path(output_path)
    if not base.exists():
        raise FileNotFoundError(base)

    scene_xml = base.read_text(encoding="utf-8")
    _, include_path, _ = _find_open_duck_robot_include(scene_xml)
    robot_base = _resolve_include_path(base, include_path)
    if not robot_base.exists():
        raise FileNotFoundError(robot_base)

    if robot_output_path is not None:
        robot_output = Path(robot_output_path)
    elif robot_base.name.startswith(DEFAULT_ROBOT_PREFIX):
        robot_output = robot_base
    else:
        robot_output = robot_base.with_name(f"{DEFAULT_ROBOT_PREFIX}{robot_base.name}")
    robot_xml = robot_base.read_text(encoding="utf-8")
    generated_robot_xml = (
        build_social_eye_robot_xml(robot_xml, cfg)
        if include_eyes
        else _remove_existing_social_eye_geoms(robot_xml)
    )
    if arm_config is not None:
        generated_robot_xml = build_visual_arm_robot_xml(generated_robot_xml, arm_config)
    if leg_config is not None:
        generated_robot_xml = build_visual_leg_shell_robot_xml(generated_robot_xml, leg_config)
    robot_output.parent.mkdir(parents=True, exist_ok=True)
    robot_output.write_text(generated_robot_xml, encoding="utf-8")

    generated_scene_xml = build_social_eye_scene_xml(
        scene_xml, scene_path=output, robot_output_path=robot_output
    )
    ElementTree.fromstring(generated_scene_xml)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated_scene_xml, encoding="utf-8")
    return SocialEyeResult(
        output_path=str(output),
        base_path=str(base),
        robot_output_path=str(robot_output),
        robot_base_path=str(robot_base),
        eye_count=2 if include_eyes else 0,
        config=cfg.as_dict(),
        arm_geom_count=len(VISUAL_ARM_GEOM_NAMES) if arm_config is not None else 0,
        arm_config=arm_config.as_dict() if arm_config is not None else None,
        leg_shell_geom_count=(len(VISUAL_LEG_GEOM_NAMES) if leg_config is not None else 0),
        leg_config=leg_config.as_dict() if leg_config is not None else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a MuJoCo scene with visual-only Soridormi social eyes "
            "and optional cosmetic limbs."
        )
    )
    parser.add_argument("--base", type=Path, required=True, help="Base MuJoCo scene XML path")
    parser.add_argument(
        "--output", type=Path, required=True, help="Generated MuJoCo scene XML output path"
    )
    parser.add_argument(
        "--robot-output", type=Path, default=None, help="Generated robot XML output path"
    )
    parser.add_argument("--eye-x", type=float, default=SocialEyeConfig.eye_x_m)
    parser.add_argument("--eye-y-offset", type=float, default=SocialEyeConfig.eye_y_offset_m)
    parser.add_argument("--eye-z", type=float, default=SocialEyeConfig.eye_z_m)
    parser.add_argument("--eye-radius", type=float, default=SocialEyeConfig.eye_radius_m)
    parser.add_argument("--eye-depth", type=float, default=SocialEyeConfig.eye_depth_m)
    parser.add_argument("--eye-quat", default=SocialEyeConfig.eye_quat)
    parser.add_argument("--eye-rgba", default=SocialEyeConfig.eye_rgba)
    parser.add_argument(
        "--no-eyes",
        action="store_true",
        help="Do not add social-eye geoms while generating another visual overlay.",
    )
    parser.add_argument(
        "--debug-frame", action="store_true", help="Add RGB axes at the generated eye anchor"
    )
    parser.add_argument(
        "--visual-arms",
        action="store_true",
        help=("Add non-colliding visual arms with fixed display poses and rounded leg shells."),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = generate_social_eye_scene(
        args.base,
        args.output,
        robot_output_path=args.robot_output,
        config=SocialEyeConfig(
            eye_x_m=args.eye_x,
            eye_y_offset_m=args.eye_y_offset,
            eye_z_m=args.eye_z,
            eye_radius_m=args.eye_radius,
            eye_depth_m=args.eye_depth,
            eye_quat=args.eye_quat,
            debug_frame=args.debug_frame,
            eye_rgba=args.eye_rgba,
        ),
        arm_config=VisualArmConfig() if args.visual_arms else None,
        leg_config=VisualLegConfig() if args.visual_arms else None,
        include_eyes=not args.no_eyes,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Soridormi social-eye scene")
        print("===========================")
        print(f"Base: {result.base_path}")
        print(f"Output: {result.output_path}")
        print(f"Robot XML: {result.robot_output_path}")
        print(f"Eyes: {result.eye_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
