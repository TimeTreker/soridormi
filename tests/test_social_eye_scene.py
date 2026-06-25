from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from soridormi_sim.social_eye_scene import (
    LEFT_EYE_CLOSED_NAME,
    LEFT_EYE_NAME,
    RIGHT_EYE_CLOSED_NAME,
    RIGHT_EYE_NAME,
    SOCIAL_EYE_FRAME_BODY_NAME,
    SOCIAL_EYE_FRAME_ORIGIN_NAME,
    SOCIAL_EYE_FRAME_X_AXIS_NAME,
    SOCIAL_EYE_FRAME_Y_AXIS_NAME,
    SOCIAL_EYE_FRAME_Z_AXIS_NAME,
    SocialEyeConfig,
    build_social_eye_robot_xml,
    generate_social_eye_scene,
)


def test_build_social_eye_robot_xml_inserts_visual_only_eye_geoms() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Part left_eye -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    xml = build_social_eye_robot_xml(robot, SocialEyeConfig(eye_radius_m=0.01))

    root = ElementTree.fromstring(xml)
    geoms = {geom.attrib["name"]: geom.attrib for geom in root.findall(".//geom")}
    assert set(geoms) == {LEFT_EYE_NAME, RIGHT_EYE_NAME, LEFT_EYE_CLOSED_NAME, RIGHT_EYE_CLOSED_NAME}
    assert geoms[LEFT_EYE_NAME]["class"] == "visual"
    assert geoms[RIGHT_EYE_NAME]["class"] == "visual"
    assert geoms[LEFT_EYE_CLOSED_NAME]["class"] == "visual"
    assert geoms[LEFT_EYE_NAME]["type"] == "ellipsoid"
    assert geoms[LEFT_EYE_CLOSED_NAME]["type"] == "ellipsoid"
    assert geoms[LEFT_EYE_NAME]["size"] == "0.002 0.01 0.01"
    assert geoms[LEFT_EYE_NAME]["quat"] == "0.707107 0 0.707107 0"
    assert geoms[LEFT_EYE_CLOSED_NAME]["rgba"].endswith(" 0")
    assert geoms[LEFT_EYE_NAME]["pos"].split()[0] == "0.01"
    assert geoms[LEFT_EYE_NAME]["pos"].split()[1] == "0.04"
    assert geoms[RIGHT_EYE_NAME]["pos"].split()[1] == "-0.04"
    assert geoms[LEFT_EYE_NAME]["pos"].split()[2] == "-0.06"


def test_build_social_eye_robot_xml_is_idempotent() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Part left_eye -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    once = build_social_eye_robot_xml(robot)
    twice = build_social_eye_robot_xml(once)

    assert twice == once
    assert twice.count(LEFT_EYE_NAME) == 1
    assert twice.count(RIGHT_EYE_NAME) == 1
    assert twice.count(LEFT_EYE_CLOSED_NAME) == 1
    assert twice.count(RIGHT_EYE_CLOSED_NAME) == 1


def test_build_social_eye_robot_xml_adds_closed_eye_geoms_to_existing_open_overlay() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        f'<geom name="{LEFT_EYE_NAME}" type="sphere"/>\n'
        f'<geom name="{RIGHT_EYE_NAME}" type="sphere"/>\n'
        "                <!-- Frame head -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    xml = build_social_eye_robot_xml(robot)

    assert xml.count(LEFT_EYE_NAME) == 1
    assert xml.count(RIGHT_EYE_NAME) == 1
    assert xml.count(LEFT_EYE_CLOSED_NAME) == 1
    assert xml.count(RIGHT_EYE_CLOSED_NAME) == 1


def test_build_social_eye_robot_xml_replaces_stale_eye_positions() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Part left_eye -->\n"
        "                <!-- Soridormi generated social eye visuals. -->\n"
        f'                <geom name="{LEFT_EYE_NAME}" type="sphere" class="visual" '
        'pos="0.043 0.021 0.048" size="0.008" rgba="0.015 0.018 0.022 1"/>\n'
        f'                <geom name="{RIGHT_EYE_NAME}" type="sphere" class="visual" '
        'pos="0.043 -0.021 0.048" size="0.008" rgba="0.015 0.018 0.022 1"/>\n'
        "                <!-- Frame head -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    xml = build_social_eye_robot_xml(robot)

    root = ElementTree.fromstring(xml)
    left_eye = root.find(f".//geom[@name='{LEFT_EYE_NAME}']")
    right_eye = root.find(f".//geom[@name='{RIGHT_EYE_NAME}']")
    assert left_eye is not None
    assert right_eye is not None
    assert left_eye.attrib["pos"] == "0.01 0.04 -0.06"
    assert right_eye.attrib["pos"] == "0.01 -0.04 -0.06"
    assert left_eye.attrib["quat"] == "0.707107 0 0.707107 0"
    assert right_eye.attrib["quat"] == "0.707107 0 0.707107 0"
    assert left_eye.attrib["type"] == "ellipsoid"
    assert right_eye.attrib["type"] == "ellipsoid"
    assert left_eye.attrib["size"] == "0.002 0.02 0.02"
    assert right_eye.attrib["size"] == "0.002 0.02 0.02"
    assert xml.count(LEFT_EYE_NAME) == 1
    assert xml.count(RIGHT_EYE_NAME) == 1


def test_build_social_eye_robot_xml_can_add_debug_eye_frame() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Part left_eye -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    xml = build_social_eye_robot_xml(robot, SocialEyeConfig(debug_frame=True))

    root = ElementTree.fromstring(xml)
    frame = root.find(f".//body[@name='{SOCIAL_EYE_FRAME_BODY_NAME}']")
    assert frame is not None
    assert frame.attrib["pos"] == "0.01 0 -0.06"
    assert frame.attrib["quat"] == "0.707107 0 0.707107 0"
    assert frame.find(f".//site[@name='{SOCIAL_EYE_FRAME_ORIGIN_NAME}']") is not None
    axis_geoms = {geom.attrib["name"]: geom.attrib for geom in frame.findall(".//geom")}
    assert axis_geoms[SOCIAL_EYE_FRAME_X_AXIS_NAME]["rgba"] == "1 0 0 1"
    assert axis_geoms[SOCIAL_EYE_FRAME_Y_AXIS_NAME]["rgba"] == "0 1 0 1"
    assert axis_geoms[SOCIAL_EYE_FRAME_Z_AXIS_NAME]["rgba"] == "0 0.25 1 1"


def test_build_social_eye_robot_xml_removes_debug_eye_frame_when_disabled() -> None:
    robot = (
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Part left_eye -->\n"
        "<site name='head'/></body></worldbody></mujoco>"
    )

    with_frame = build_social_eye_robot_xml(robot, SocialEyeConfig(debug_frame=True))
    without_frame = build_social_eye_robot_xml(with_frame)

    root = ElementTree.fromstring(without_frame)
    assert root.find(f".//body[@name='{SOCIAL_EYE_FRAME_BODY_NAME}']") is None
    assert without_frame.count(LEFT_EYE_NAME) == 1
    assert without_frame.count(RIGHT_EYE_NAME) == 1


def test_generate_social_eye_scene_writes_scene_and_robot_overlay(tmp_path: Path) -> None:
    base_scene = tmp_path / "scene_flat_terrain.xml"
    base_robot = tmp_path / "open_duck_mini_v2.xml"
    out_scene = tmp_path / "soridormi_social_eyes_scene.xml"
    base_scene.write_text(
        '<mujoco><include file="open_duck_mini_v2.xml"/><worldbody/></mujoco>',
        encoding="utf-8",
    )
    base_robot.write_text(
        "<mujoco><worldbody><body name='head_assembly'>\n"
        "                <!-- Frame head -->\n"
        "<site name='head'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )

    result = generate_social_eye_scene(base_scene, out_scene)

    generated_robot = tmp_path / "soridormi_social_eyes_open_duck_mini_v2.xml"
    assert out_scene.exists()
    assert generated_robot.exists()
    assert result.eye_count == 2
    assert f'file="{generated_robot.name}"' in out_scene.read_text(encoding="utf-8")
    assert LEFT_EYE_NAME in generated_robot.read_text(encoding="utf-8")


def test_generate_social_eye_scene_reuses_existing_generated_robot_overlay(tmp_path: Path) -> None:
    base_scene = tmp_path / "soridormi_social_eyes_scene.xml"
    generated_robot = tmp_path / "soridormi_social_eyes_open_duck_mini_v2.xml"
    out_scene = tmp_path / "again.xml"
    base_scene.write_text(
        f'<mujoco><include file="{generated_robot.name}"/><worldbody/></mujoco>',
        encoding="utf-8",
    )
    generated_robot.write_text(
        build_social_eye_robot_xml(
            "<mujoco><worldbody><body name='head_assembly'>\n"
            "                <!-- Frame head -->\n"
            "<site name='head'/></body></worldbody></mujoco>"
        ),
        encoding="utf-8",
    )

    result = generate_social_eye_scene(base_scene, out_scene)

    assert result.robot_output_path == str(generated_robot)
    assert not (tmp_path / "soridormi_social_eyes_soridormi_social_eyes_open_duck_mini_v2.xml").exists()
    assert generated_robot.read_text(encoding="utf-8").count(LEFT_EYE_NAME) == 1
