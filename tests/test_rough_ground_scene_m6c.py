from __future__ import annotations

from pathlib import Path

from xml.etree import ElementTree

from soridormi_sim.rough_ground_scene import (
    RoughGroundConfig,
    build_rough_ground_xml,
    generate_rough_ground_scene,
    rewrite_relative_includes,
)


def test_build_rough_ground_xml_inserts_stones() -> None:
    base = "<mujoco><worldbody><geom name='floor'/></worldbody></mujoco>"
    xml, stones = build_rough_ground_xml(base, RoughGroundConfig(stone_count=3, seed=1))
    assert len(stones) == 3
    assert "soridormi_stone_00" in xml
    assert "soridormi_stone_02" in xml
    assert xml.count("soridormi_stone_") == 3


def test_generate_rough_ground_scene_writes_file(tmp_path: Path) -> None:
    base = tmp_path / "base.xml"
    out = tmp_path / "rough.xml"
    base.write_text("<mujoco><worldbody></worldbody></mujoco>", encoding="utf-8")
    result = generate_rough_ground_scene(base, out, config=RoughGroundConfig(stone_count=2))
    assert out.exists()
    assert result.stone_count == 2
    assert "soridormi_stone_01" in out.read_text(encoding="utf-8")


def test_build_rough_ground_xml_ignores_commented_worldbody() -> None:
    base = (
        '<mujoco>'
        '<!-- <worldbody><geom name="old"/></worldbody> -->'
        '<worldbody><geom name="floor"/></worldbody>'
        '</mujoco>'
    )
    xml, stones = build_rough_ground_xml(base, RoughGroundConfig(stone_count=1, seed=1))

    assert len(stones) == 1
    assert '<!-- <worldbody><geom name="old"/></worldbody> -->' in xml
    assert xml.find('soridormi_stone_00') > xml.find('<geom name="floor"')
    ElementTree.fromstring(xml)


def test_rewrite_relative_includes_for_tmp_output(tmp_path: Path) -> None:
    base_dir = tmp_path / "xmls"
    base_dir.mkdir()
    xml = '<mujoco><include file="open_duck_mini_v2.xml"/><worldbody/></mujoco>'

    rewritten = rewrite_relative_includes(xml, base_dir)

    assert f'file="{(base_dir / "open_duck_mini_v2.xml").resolve().as_posix()}"' in rewritten
