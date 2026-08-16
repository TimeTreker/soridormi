from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import sqrt
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

VISUAL_ARM_POSES = ("rest", "reach", "hold", "place")
VISUAL_ARM_SIDES = ("left", "right")
VISUAL_ARM_MAIN_FINGER_COMPONENTS = (
    "index_finger",
    "middle_finger",
    "ring_finger",
    "pinky_finger",
)
VISUAL_ARM_FINGER_COMPONENTS = VISUAL_ARM_MAIN_FINGER_COMPONENTS + ("thumb",)
VISUAL_ARM_COMPONENTS = (
    "upper",
    "elbow",
    "forearm",
    "hand",
) + VISUAL_ARM_FINGER_COMPONENTS


def visual_arm_geom_name(side: str, pose: str, component: str) -> str:
    return f"soridormi_{side}_arm_{pose}_{component}_visual"


VISUAL_ARM_SHOULDER_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_SHOULDER_MOUNT_NAMES = tuple(
    f"soridormi_{side}_arm_shoulder_mount_visual" for side in VISUAL_ARM_SIDES
)
VISUAL_ARM_POSE_GEOM_NAMES = tuple(
    visual_arm_geom_name(side, pose, component)
    for side in VISUAL_ARM_SIDES
    for pose in VISUAL_ARM_POSES
    for component in VISUAL_ARM_COMPONENTS
)
VISUAL_ARM_GEOM_NAMES = (
    VISUAL_ARM_SHOULDER_MOUNT_NAMES + VISUAL_ARM_SHOULDER_NAMES + VISUAL_ARM_POSE_GEOM_NAMES
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
    shoulder_mount_y_offset_m: float = 0.055
    shoulder_mount_radius_m: float = 0.019
    segment_radius_m: float = 0.017
    joint_radius_m: float = 0.021
    hand_size_xyz_m: tuple[float, float, float] = (0.022, 0.019, 0.024)
    finger_radius_m: float = 0.0042
    finger_spread_m: float = 0.0065
    finger_start_offset_m: float = 0.008
    finger_tip_offsets_m: tuple[float, float, float, float] = (0.04, 0.044, 0.041, 0.034)
    thumb_root_forward_offset_m: float = 0.004
    thumb_root_side_offset_m: float = 0.01
    thumb_tip_forward_offset_m: float = 0.024
    thumb_tip_side_offset_m: float = 0.028
    arm_rgba: str = "0.917647 0.917647 0.917647 1"
    joint_rgba: str = "0.223529 0.219608 0.219608 1"
    hand_rgba: str = "0.980392 0.713725 0.00392157 1"

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _format_xyz(values: tuple[float, float, float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _normalized_xyz(
    values: tuple[float, float, float],
) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-9:
        raise ValueError("visual-arm direction must have non-zero length")
    return tuple(value / norm for value in values)


def _offset_xyz(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    return tuple(value + distance * delta for value, delta in zip(origin, direction))


def _dot_xyz(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _visual_arm_finger_points(
    side: str,
    elbow: tuple[float, float, float],
    hand: tuple[float, float, float],
    config: VisualArmConfig,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    ...,
]:
    finger_direction = _normalized_xyz(tuple(end - start for start, end in zip(elbow, hand)))
    sign = 1.0 if side == "left" else -1.0
    inner_hint = (0.0, -sign, 0.0)
    inner_projection = _dot_xyz(inner_hint, finger_direction)
    inner_direction = _normalized_xyz(
        tuple(
            hint - inner_projection * direction
            for hint, direction in zip(inner_hint, finger_direction)
        )
    )
    spread_candidate = (0.0, -finger_direction[2], finger_direction[1])
    if sqrt(sum(value * value for value in spread_candidate)) <= 1e-9:
        spread_direction = inner_direction
    else:
        spread_direction = _normalized_xyz(spread_candidate)
    if _dot_xyz(spread_direction, inner_direction) < 0.0:
        spread_direction = tuple(-value for value in spread_direction)

    fingers = []
    for spread_multiplier, tip_offset in zip(
        (1.5, 0.5, -0.5, -1.5),
        config.finger_tip_offsets_m,
    ):
        spread = spread_multiplier * config.finger_spread_m
        root = _offset_xyz(hand, spread_direction, spread)
        fingers.append(
            (
                _offset_xyz(root, finger_direction, config.finger_start_offset_m),
                _offset_xyz(root, finger_direction, tip_offset),
            )
        )

    thumb_root = _offset_xyz(
        _offset_xyz(hand, finger_direction, config.thumb_root_forward_offset_m),
        inner_direction,
        config.thumb_root_side_offset_m,
    )
    thumb_tip = _offset_xyz(
        _offset_xyz(hand, finger_direction, config.thumb_tip_forward_offset_m),
        inner_direction,
        config.thumb_tip_side_offset_m,
    )
    fingers.append((thumb_root, thumb_tip))
    return tuple(fingers)


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
            (-0.028, sign * 0.156, 0.045),
            (-0.005, sign * 0.186, 0.0),
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
    }
    elbow, hand = points[pose]
    return shoulder, elbow, hand


def _visual_arm_pose_xml(side: str, pose: str, config: VisualArmConfig) -> str:
    shoulder, elbow, hand = _arm_pose_points(side, pose, config)
    alpha = "1" if pose == "rest" else "0"
    arm_rgba = " ".join(config.arm_rgba.split()[:3] + [alpha])
    joint_rgba = " ".join(config.joint_rgba.split()[:3] + [alpha])
    hand_rgba = " ".join(config.hand_rgba.split()[:3] + [alpha])
    upper = visual_arm_geom_name(side, pose, "upper")
    elbow_name = visual_arm_geom_name(side, pose, "elbow")
    forearm = visual_arm_geom_name(side, pose, "forearm")
    hand_name = visual_arm_geom_name(side, pose, "hand")
    chunks = [
        (
            f'        <geom name="{upper}" type="capsule" class="visual" '
            f'contype="0" conaffinity="0" '
            f'fromto="{_format_xyz(shoulder)} {_format_xyz(elbow)}" '
            f'size="{_format_float(config.segment_radius_m)}" rgba="{arm_rgba}"/>\n'
        ),
        (
            f'        <geom name="{elbow_name}" type="sphere" class="visual" '
            f'contype="0" conaffinity="0" pos="{_format_xyz(elbow)}" '
            f'size="{_format_float(config.joint_radius_m)}" rgba="{joint_rgba}"/>\n'
        ),
        (
            f'        <geom name="{forearm}" type="capsule" class="visual" '
            f'contype="0" conaffinity="0" '
            f'fromto="{_format_xyz(elbow)} {_format_xyz(hand)}" '
            f'size="{_format_float(config.segment_radius_m)}" rgba="{arm_rgba}"/>\n'
        ),
        (
            f'        <geom name="{hand_name}" type="ellipsoid" class="visual" '
            f'contype="0" conaffinity="0" pos="{_format_xyz(hand)}" '
            f'size="{_format_xyz(config.hand_size_xyz_m)}" rgba="{hand_rgba}"/>\n'
        ),
    ]
    for component, (finger_root, finger_tip) in zip(
        VISUAL_ARM_FINGER_COMPONENTS,
        _visual_arm_finger_points(side, elbow, hand, config),
    ):
        finger_name = visual_arm_geom_name(side, pose, component)
        chunks.append(
            f'        <geom name="{finger_name}" type="capsule" class="visual" '
            f'contype="0" conaffinity="0" '
            f'fromto="{_format_xyz(finger_root)} {_format_xyz(finger_tip)}" '
            f'size="{_format_float(config.finger_radius_m)}" rgba="{hand_rgba}"/>\n'
        )
    return "".join(chunks)


def build_visual_arm_geoms(config: VisualArmConfig | None = None) -> str:
    cfg = config or VisualArmConfig()
    chunks = [VISUAL_ARM_COMMENT]
    for side, mount_name, shoulder_name in zip(
        VISUAL_ARM_SIDES,
        VISUAL_ARM_SHOULDER_MOUNT_NAMES,
        VISUAL_ARM_SHOULDER_NAMES,
    ):
        sign = 1.0 if side == "left" else -1.0
        shoulder, _elbow, _hand = _arm_pose_points(side, "rest", cfg)
        mount = (
            cfg.shoulder_x_m,
            sign * cfg.shoulder_mount_y_offset_m,
            cfg.shoulder_z_m,
        )
        chunks.append(
            f'        <geom name="{mount_name}" type="capsule" class="visual" '
            f'contype="0" conaffinity="0" fromto="{_format_xyz(mount)} {_format_xyz(shoulder)}" '
            f'size="{_format_float(cfg.shoulder_mount_radius_m)}" rgba="{cfg.arm_rgba}"/>\n'
        )
        chunks.append(
            f'        <geom name="{shoulder_name}" type="sphere" class="visual" '
            f'contype="0" conaffinity="0" pos="{_format_xyz(shoulder)}" '
            f'size="{_format_float(cfg.joint_radius_m)}" rgba="{cfg.joint_rgba}"/>\n'
        )
        for pose in VISUAL_ARM_POSES:
            chunks.append(_visual_arm_pose_xml(side, pose, cfg))
    return "".join(chunks)


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
    for name in VISUAL_ARM_GEOM_NAMES:
        xml = re.sub(
            rf"^[ \t]*<geom\b[^>]*\bname=\"{re.escape(name)}\"[^>]*/>\n?",
            "",
            xml,
            flags=re.MULTILINE,
        )
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
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo scene with visual-only Soridormi social eyes and optional arms."
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
        help="Add non-colliding, jointless visual arms with fixed display poses.",
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
