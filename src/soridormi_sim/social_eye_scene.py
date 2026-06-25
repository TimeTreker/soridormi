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
DEFAULT_ROBOT_PREFIX = "soridormi_social_eyes_"


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
class SocialEyeResult:
    output_path: str
    base_path: str
    robot_output_path: str
    robot_base_path: str
    eye_count: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


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
        chunks.append(_closed_eye_geom_xml(RIGHT_EYE_CLOSED_NAME, y=-cfg.eye_y_offset_m, config=cfg))
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
        xml = re.sub(rf"^[ \t]*<geom\b[^>]*\bname=\"{re.escape(name)}\"[^>]*/>\n?", "", xml, flags=re.MULTILINE)
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
    generated_robot_xml = build_social_eye_robot_xml(robot_xml, cfg)
    robot_output.parent.mkdir(parents=True, exist_ok=True)
    robot_output.write_text(generated_robot_xml, encoding="utf-8")

    generated_scene_xml = build_social_eye_scene_xml(scene_xml, scene_path=output, robot_output_path=robot_output)
    ElementTree.fromstring(generated_scene_xml)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated_scene_xml, encoding="utf-8")
    return SocialEyeResult(
        output_path=str(output),
        base_path=str(base),
        robot_output_path=str(robot_output),
        robot_base_path=str(robot_base),
        eye_count=2,
        config=cfg.as_dict(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo scene with visual-only Soridormi social eyes.")
    parser.add_argument("--base", type=Path, required=True, help="Base MuJoCo scene XML path")
    parser.add_argument("--output", type=Path, required=True, help="Generated MuJoCo scene XML output path")
    parser.add_argument("--robot-output", type=Path, default=None, help="Generated robot XML output path")
    parser.add_argument("--eye-x", type=float, default=SocialEyeConfig.eye_x_m)
    parser.add_argument("--eye-y-offset", type=float, default=SocialEyeConfig.eye_y_offset_m)
    parser.add_argument("--eye-z", type=float, default=SocialEyeConfig.eye_z_m)
    parser.add_argument("--eye-radius", type=float, default=SocialEyeConfig.eye_radius_m)
    parser.add_argument("--eye-depth", type=float, default=SocialEyeConfig.eye_depth_m)
    parser.add_argument("--eye-quat", default=SocialEyeConfig.eye_quat)
    parser.add_argument("--eye-rgba", default=SocialEyeConfig.eye_rgba)
    parser.add_argument("--debug-frame", action="store_true", help="Add RGB axes at the generated eye anchor")
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
