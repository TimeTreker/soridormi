from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


@dataclass(frozen=True)
class RoughGroundConfig:
    stone_count: int = 8
    stone_height_m: float = 0.008
    stone_radius_m: float = 0.018
    start_x_m: float = 0.18
    spacing_m: float = 0.12
    lateral_span_m: float = 0.10
    seed: int = 123

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoughGroundResult:
    output_path: str
    base_path: str
    stone_count: int
    config: dict[str, Any]
    stones: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stone_geom_xml(name: str, *, x: float, y: float, radius: float, height: float) -> str:
    # MuJoCo boxes use half extents. Place the box so its top is `height` above
    # the ground plane and bottom touches z=0.
    half_height = max(float(height) / 2.0, 0.0005)
    half_radius = max(float(radius), 0.001)
    z = half_height
    return (
        f'    <geom name="{name}" type="box" '
        f'size="{half_radius:.6f} {half_radius:.6f} {half_height:.6f}" '
        f'pos="{x:.6f} {y:.6f} {z:.6f}" '
        'rgba="0.35 0.30 0.24 1" friction="1.0 0.005 0.0001"/>\n'
    )


def _build_stone_insertion(config: RoughGroundConfig) -> tuple[str, list[dict[str, float | str]]]:
    rng = random.Random(int(config.seed))
    stones: list[dict[str, float | str]] = []
    xml_chunks = ["    <!-- Soridormi generated rough-ground stones. -->\n"]
    for index in range(int(config.stone_count)):
        x = float(config.start_x_m) + float(index) * float(config.spacing_m)
        y = rng.uniform(-float(config.lateral_span_m), float(config.lateral_span_m))
        height_jitter = rng.uniform(0.75, 1.25)
        radius_jitter = rng.uniform(0.75, 1.25)
        height = max(0.001, float(config.stone_height_m) * height_jitter)
        radius = max(0.002, float(config.stone_radius_m) * radius_jitter)
        name = f"soridormi_stone_{index:02d}"
        stones.append({"name": name, "x": x, "y": y, "height": height, "radius": radius})
        xml_chunks.append(_stone_geom_xml(name, x=x, y=y, radius=radius, height=height))
    return "".join(xml_chunks), stones


def build_rough_ground_xml(base_xml: str, config: RoughGroundConfig) -> tuple[str, list[dict[str, float | str]]]:
    if "</worldbody>" not in base_xml:
        raise ValueError("base MuJoCo XML does not contain </worldbody>")

    insertion, stones = _build_stone_insertion(config)

    # Open Duck's scene XML contains an older worldbody block inside an XML
    # comment before the real <worldbody>. Inserting at the first
    # </worldbody> corrupts that comment and produces a MuJoCo parse error like
    # XML_ERROR_MISMATCHED_ELEMENT. Insert before the last real closing tag
    # instead. This remains intentionally string-based so MuJoCo <include>
    # statements and formatting are preserved.
    insert_at = base_xml.rfind("</worldbody>")
    if insert_at < 0:
        raise ValueError("base MuJoCo XML does not contain </worldbody>")
    xml = base_xml[:insert_at] + insertion + base_xml[insert_at:]
    ElementTree.fromstring(xml)
    return xml, stones


def rewrite_relative_includes(base_xml: str, base_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, include_path, suffix = match.groups()
        include = Path(include_path)
        if include.is_absolute() or "://" in include_path:
            return match.group(0)
        return f'{prefix}{(base_dir / include).resolve().as_posix()}{suffix}'

    return re.sub(r'(<include\b[^>]*\bfile=")([^"/][^"]*)(")', replace, base_xml)


def generate_rough_ground_scene(
    base_path: str | Path,
    output_path: str | Path,
    *,
    config: RoughGroundConfig | None = None,
) -> RoughGroundResult:
    cfg = config or RoughGroundConfig()
    base = Path(base_path)
    output = Path(output_path)
    if not base.exists():
        raise FileNotFoundError(base)
    base_xml = base.read_text(encoding="utf-8")
    if output.parent.resolve() != base.parent.resolve():
        base_xml = rewrite_relative_includes(base_xml, base.parent)
    xml, stones = build_rough_ground_xml(base_xml, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml, encoding="utf-8")
    return RoughGroundResult(
        output_path=str(output),
        base_path=str(base),
        stone_count=len(stones),
        config=cfg.as_dict(),
        stones=stones,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo XML with small rough-ground stones.")
    parser.add_argument("--base", type=Path, required=True, help="Base MuJoCo XML path")
    parser.add_argument("--output", type=Path, required=True, help="Generated XML output path")
    parser.add_argument("--stone-count", type=int, default=RoughGroundConfig.stone_count)
    parser.add_argument("--stone-height", type=float, default=RoughGroundConfig.stone_height_m)
    parser.add_argument("--stone-radius", type=float, default=RoughGroundConfig.stone_radius_m)
    parser.add_argument("--start-x", type=float, default=RoughGroundConfig.start_x_m)
    parser.add_argument("--spacing", type=float, default=RoughGroundConfig.spacing_m)
    parser.add_argument("--lateral-span", type=float, default=RoughGroundConfig.lateral_span_m)
    parser.add_argument("--seed", type=int, default=RoughGroundConfig.seed)
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = generate_rough_ground_scene(
        args.base,
        args.output,
        config=RoughGroundConfig(
            stone_count=args.stone_count,
            stone_height_m=args.stone_height,
            stone_radius_m=args.stone_radius,
            start_x_m=args.start_x,
            spacing_m=args.spacing,
            lateral_span_m=args.lateral_span,
            seed=args.seed,
        ),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Soridormi rough-ground scene")
        print("============================")
        print(f"Base: {result.base_path}")
        print(f"Output: {result.output_path}")
        print(f"Stones: {result.stone_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
