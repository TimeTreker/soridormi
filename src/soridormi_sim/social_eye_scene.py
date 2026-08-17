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
LEGACY_VISUAL_ARM_COMPONENTS = ("hand", "thumb", *VISUAL_ARM_FINGERS)


def visual_arm_geom_name(side: str, pose: str, component: str) -> str:
    return f"soridormi_{side}_arm_{pose}_{component}_visual"


VISUAL_ARM_SHOULDER_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_POSE_GEOM_NAMES = tuple(
    visual_arm_geom_name(side, pose, component)
    for side in VISUAL_ARM_SIDES
    for pose in VISUAL_ARM_POSES
    for component in VISUAL_ARM_COMPONENTS
)
VISUAL_ARM_GEOM_NAMES = VISUAL_ARM_SHOULDER_NAMES + VISUAL_ARM_POSE_GEOM_NAMES

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
    shoulder_x_m: float = -0.02
    shoulder_y_offset_m: float = 0.09
    shoulder_z_m: float = 0.105
    upper_arm_radius_m: float = 0.017
    forearm_radius_m: float = 0.014
    joint_radius_m: float = 0.0185
    cuff_radius_m: float = 0.0125
    wrist_radius_m: float = 0.008
    palm_thickness_m: float = 0.0065
    palm_width_m: float = 0.0105
    palm_length_m: float = 0.026
    finger_radius_m: float = 0.0034
    finger_joint_radius_m: float = 0.0038
    arm_rgba: str = "0.917647 0.917647 0.917647 1"
    hand_rgba: str = "0.917647 0.917647 0.917647 1"
    joint_rgba: str = "0.909804 0.572549 0.164706 1"
    finger_joint_rgba: str = "0.82 0.82 0.80 1"
    finger_root_joint_rgba: str = "0.93 0.66 0.28 1"

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
            (-0.028, sign * 0.157, 0.045),
            (-0.005, sign * 0.184, 0.0),
        ),
        "reach": (
            (0.04, sign * 0.125, 0.08),
            (0.11, sign * 0.105, 0.06),
        ),
        "hold": (
            (0.035, sign * 0.12, 0.08),
            (0.09, sign * 0.038, 0.07),
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
            (0.04, sign * 0.12, 0.1),
            (0.105, sign * 0.035, 0.1),
        ),
    }
    elbow, hand = points[pose]
    return shoulder, elbow, hand


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


def _visual_arm_pose_xml(side: str, pose: str, config: VisualArmConfig) -> str:
    shoulder, elbow, wrist = _arm_pose_points(side, pose, config)
    alpha = "1" if pose == "rest" else "0"
    arm_rgba = " ".join(config.arm_rgba.split()[:3] + [alpha])
    hand_rgba = " ".join(config.hand_rgba.split()[:3] + [alpha])
    joint_rgba = " ".join(config.joint_rgba.split()[:3] + [alpha])
    upper = visual_arm_geom_name(side, pose, "upper")
    elbow_name = visual_arm_geom_name(side, pose, "elbow")
    forearm = visual_arm_geom_name(side, pose, "forearm")
    cuff = visual_arm_geom_name(side, pose, "cuff")
    wrist_name = visual_arm_geom_name(side, pose, "wrist")
    palm = visual_arm_geom_name(side, pose, "palm")

    hand_forward = _arm_hand_direction(pose, elbow, wrist)
    sign = 1.0 if side == "left" else -1.0
    medial = _perpendicular_unit((0.0, -sign, 0.0), hand_forward)
    palm_normal = _unit_vector(_vector_cross(medial, hand_forward))
    palm_quat = _quat_from_axes(palm_normal, medial, hand_forward)
    curl_direction = _unit_vector(_vector_scale(_vector_cross(medial, hand_forward), sign))
    forearm_direction = _unit_vector(_vector_subtract(wrist, elbow))
    cuff_start = _vector_subtract(wrist, _vector_scale(forearm_direction, 0.02))
    cuff_end = _vector_subtract(wrist, _vector_scale(forearm_direction, 0.008))
    wrist_start = _vector_subtract(wrist, _vector_scale(forearm_direction, 0.004))
    wrist_end = _vector_add(wrist, _vector_scale(hand_forward, 0.006))
    palm_start = _vector_add(wrist, _vector_scale(hand_forward, 0.005))
    palm_end = _vector_add(palm_start, _vector_scale(hand_forward, config.palm_length_m))
    palm_center = _vector_add(palm_start, _vector_scale(hand_forward, config.palm_length_m * 0.5))
    finger_joint_rgba = " ".join(config.finger_joint_rgba.split()[:3] + [alpha])
    finger_root_joint_rgba = " ".join(config.finger_root_joint_rgba.split()[:3] + [alpha])

    xml = (
        f'        <geom name="{upper}" type="capsule" class="visual" contype="0" conaffinity="0" '
        f'fromto="{_format_xyz(shoulder)} {_format_xyz(elbow)}" '
        f'size="{_format_float(config.upper_arm_radius_m)}" rgba="{arm_rgba}"/>\n'
        f'        <geom name="{elbow_name}" type="sphere" class="visual" contype="0" conaffinity="0" '
        f'pos="{_format_xyz(elbow)}" size="{_format_float(config.joint_radius_m)}" rgba="{joint_rgba}"/>\n'
        f'        <geom name="{forearm}" type="capsule" class="visual" contype="0" conaffinity="0" '
        f'fromto="{_format_xyz(elbow)} {_format_xyz(cuff_start)}" '
        f'size="{_format_float(config.forearm_radius_m)}" rgba="{arm_rgba}"/>\n'
        f'        <geom name="{cuff}" type="cylinder" class="visual" contype="0" conaffinity="0" '
        f'fromto="{_format_xyz(cuff_start)} {_format_xyz(cuff_end)}" '
        f'size="{_format_float(config.cuff_radius_m)}" rgba="{arm_rgba}"/>\n'
        f'        <geom name="{wrist_name}" type="cylinder" class="visual" contype="0" conaffinity="0" '
        f'fromto="{_format_xyz(wrist_start)} {_format_xyz(wrist_end)}" '
        f'size="{_format_float(config.wrist_radius_m)}" rgba="{joint_rgba}"/>\n'
        f'        <geom name="{palm}" type="ellipsoid" class="visual" contype="0" conaffinity="0" '
        f'pos="{_format_xyz(palm_center)}" quat="{_format_xyz(palm_quat)}" '
        f'size="{_format_xyz((config.palm_thickness_m, config.palm_width_m, config.palm_length_m * 0.5))}" '
        f'rgba="{hand_rgba}"/>\n'
    )

    finger_specs = {
        "index": (0.009, 0.024),
        "middle": (0.003, 0.026),
        "ring": (-0.003, 0.0245),
        "little": (-0.009, 0.02),
    }
    middle_bend, distal_bend = _finger_bend_weights(pose)
    for finger, (offset, length) in finger_specs.items():
        mcp = _vector_add(palm_end, _vector_scale(medial, offset))
        fan = _vector_scale(medial, offset * 7.5)
        proximal_direction = _unit_vector(_vector_add(hand_forward, fan))
        middle_direction = _unit_vector(
            _vector_add(
                _vector_add(hand_forward, fan),
                _vector_scale(curl_direction, middle_bend),
            )
        )
        distal_direction = _unit_vector(
            _vector_add(
                _vector_add(hand_forward, fan),
                _vector_scale(curl_direction, distal_bend),
            )
        )
        pip = _vector_add(mcp, _vector_scale(proximal_direction, length * 0.42))
        dip = _vector_add(pip, _vector_scale(middle_direction, length * 0.33))
        tip = _vector_add(dip, _vector_scale(distal_direction, length * 0.25))

        for joint_index, (joint, position) in enumerate((("mcp", mcp), ("pip", pip), ("dip", dip))):
            joint_scale = (1.05, 0.95, 0.85)[joint_index]
            knuckle_rgba = finger_root_joint_rgba if joint == "mcp" else finger_joint_rgba
            xml += (
                f'        <geom name="{visual_arm_geom_name(side, pose, f"{finger}_{joint}")}" '
                f'type="sphere" class="visual" contype="0" conaffinity="0" '
                f'pos="{_format_xyz(position)}" '
                f'size="{_format_float(config.finger_joint_radius_m * joint_scale)}" '
                f'rgba="{knuckle_rgba}"/>\n'
            )

        for segment_index, (segment, start, end) in enumerate(
            (
                ("proximal", mcp, pip),
                ("middle", pip, dip),
                ("distal", dip, tip),
            )
        ):
            segment_scale = (1.0, 0.88, 0.76)[segment_index]
            xml += (
                f'        <geom name="{visual_arm_geom_name(side, pose, f"{finger}_{segment}")}" '
                f'type="capsule" class="visual" contype="0" conaffinity="0" '
                f'fromto="{_format_xyz(start)} {_format_xyz(end)}" '
                f'size="{_format_float(config.finger_radius_m * segment_scale)}" '
                f'rgba="{hand_rgba}"/>\n'
            )

    thumb_cmc = _vector_add(
        _vector_add(wrist, _vector_scale(hand_forward, config.palm_length_m * 0.38)),
        _vector_scale(medial, config.palm_width_m * 0.72),
    )
    thumb_metacarpal_direction = _unit_vector(
        _vector_add(
            _vector_add(
                _vector_scale(hand_forward, 0.35),
                _vector_scale(medial, 1.0),
            ),
            _vector_scale(curl_direction, 0.08),
        )
    )
    thumb_proximal_direction = _unit_vector(
        _vector_add(
            _vector_add(
                _vector_scale(hand_forward, 0.6),
                _vector_scale(medial, 0.8),
            ),
            _vector_scale(curl_direction, 0.18 + middle_bend * 0.2),
        )
    )
    thumb_distal_direction = _unit_vector(
        _vector_add(
            _vector_add(
                _vector_scale(hand_forward, 0.72),
                _vector_scale(medial, 0.62),
            ),
            _vector_scale(curl_direction, 0.28 + distal_bend * 0.25),
        )
    )
    thumb_mcp = _vector_add(thumb_cmc, _vector_scale(thumb_metacarpal_direction, 0.007))
    thumb_ip = _vector_add(thumb_mcp, _vector_scale(thumb_proximal_direction, 0.008))
    thumb_tip = _vector_add(thumb_ip, _vector_scale(thumb_distal_direction, 0.007))

    for joint_index, (joint, position) in enumerate(
        (
            ("cmc", thumb_cmc),
            ("mcp", thumb_mcp),
            ("ip", thumb_ip),
        )
    ):
        joint_scale = (1.05, 0.95, 0.85)[joint_index]
        knuckle_rgba = finger_root_joint_rgba if joint == "cmc" else finger_joint_rgba
        xml += (
            f'        <geom name="{visual_arm_geom_name(side, pose, f"thumb_{joint}")}" '
            f'type="sphere" class="visual" contype="0" conaffinity="0" '
            f'pos="{_format_xyz(position)}" '
            f'size="{_format_float(config.finger_joint_radius_m * joint_scale)}" '
            f'rgba="{knuckle_rgba}"/>\n'
        )

    for segment_index, (segment, start, end) in enumerate(
        (
            ("metacarpal", thumb_cmc, thumb_mcp),
            ("proximal", thumb_mcp, thumb_ip),
            ("distal", thumb_ip, thumb_tip),
        )
    ):
        segment_scale = (1.0, 0.9, 0.8)[segment_index]
        xml += (
            f'        <geom name="{visual_arm_geom_name(side, pose, f"thumb_{segment}")}" '
            f'type="capsule" class="visual" contype="0" conaffinity="0" '
            f'fromto="{_format_xyz(start)} {_format_xyz(end)}" '
            f'size="{_format_float(config.finger_radius_m * segment_scale)}" '
            f'rgba="{hand_rgba}"/>\n'
        )
    return xml


def build_visual_arm_geoms(config: VisualArmConfig | None = None) -> str:
    cfg = config or VisualArmConfig()
    chunks = [VISUAL_ARM_COMMENT]
    for side, shoulder_name in zip(VISUAL_ARM_SIDES, VISUAL_ARM_SHOULDER_NAMES):
        shoulder, _elbow, _hand = _arm_pose_points(side, "rest", cfg)
        chunks.append(
            f'        <geom name="{shoulder_name}" type="sphere" class="visual" '
            f'contype="0" conaffinity="0" pos="{_format_xyz(shoulder)}" '
            f'size="{_format_float(cfg.joint_radius_m)}" rgba="{cfg.joint_rgba}"/>\n'
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
