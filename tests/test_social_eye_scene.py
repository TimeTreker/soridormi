from __future__ import annotations

from math import dist, sqrt
from pathlib import Path
from xml.etree import ElementTree

import pytest

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
    VISUAL_ARM_COMPONENTS,
    VISUAL_ARM_FINGER_COMPONENTS,
    VISUAL_ARM_GEOM_NAMES,
    VISUAL_ARM_MAIN_FINGER_COMPONENTS,
    VISUAL_ARM_POSES,
    VISUAL_ARM_SHOULDER_MOUNT_NAMES,
    VISUAL_ARM_SHOULDER_NAMES,
    VISUAL_ARM_SIDES,
    SocialEyeConfig,
    VisualArmConfig,
    build_social_eye_robot_xml,
    build_visual_arm_robot_xml,
    generate_social_eye_scene,
    visual_arm_geom_name,
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
    assert set(geoms) == {
        LEFT_EYE_NAME,
        RIGHT_EYE_NAME,
        LEFT_EYE_CLOSED_NAME,
        RIGHT_EYE_CLOSED_NAME,
    }
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


def test_build_visual_arm_robot_xml_adds_only_non_contact_geoms() -> None:
    robot = (
        "<mujoco><worldbody><body name='trunk_assembly'>\n"
        "        <!-- Frame trunk -->\n"
        "<site name='trunk'/></body></worldbody><actuator>"
        "<position name='leg' joint='leg'/></actuator></mujoco>"
    )

    xml = build_visual_arm_robot_xml(robot, VisualArmConfig())

    root = ElementTree.fromstring(xml)
    geoms = {geom.attrib["name"]: geom.attrib for geom in root.findall(".//geom")}
    assert set(geoms) == set(VISUAL_ARM_GEOM_NAMES)
    assert root.findall(".//joint") == []
    assert len(root.findall(".//actuator/position")) == 1
    assert root.findall(".//inertial") == []
    for geom in geoms.values():
        assert geom["class"] == "visual"
        assert geom["contype"] == "0"
        assert geom["conaffinity"] == "0"

    config = VisualArmConfig()
    for side, mount_name, shoulder_name in zip(
        VISUAL_ARM_SIDES,
        VISUAL_ARM_SHOULDER_MOUNT_NAMES,
        VISUAL_ARM_SHOULDER_NAMES,
    ):
        sign = 1.0 if side == "left" else -1.0
        mount = geoms[mount_name]
        assert mount["type"] == "capsule"
        assert mount["rgba"] == config.arm_rgba
        assert [float(value) for value in mount["fromto"].split()] == pytest.approx(
            [
                config.shoulder_x_m,
                sign * config.shoulder_mount_y_offset_m,
                config.shoulder_z_m,
                config.shoulder_x_m,
                sign * config.shoulder_y_offset_m,
                config.shoulder_z_m,
            ]
        )
        assert geoms[shoulder_name]["rgba"] == config.joint_rgba

    for side in VISUAL_ARM_SIDES:
        for pose in VISUAL_ARM_POSES:
            expected_alpha = "1" if pose == "rest" else "0"
            for component in VISUAL_ARM_COMPONENTS:
                rgba = geoms[visual_arm_geom_name(side, pose, component)]["rgba"]
                assert rgba.split()[-1] == expected_alpha
            hand_rgba = geoms[visual_arm_geom_name(side, pose, "hand")]["rgba"]
            assert hand_rgba.split()[:3] == config.hand_rgba.split()[:3]
            hand_position = tuple(
                float(value)
                for value in geoms[visual_arm_geom_name(side, pose, "hand")]["pos"].split()
            )
            finger_vectors: dict[str, tuple[float, float, float]] = {}
            for component in VISUAL_ARM_FINGER_COMPONENTS:
                finger = geoms[visual_arm_geom_name(side, pose, component)]
                fromto = [float(value) for value in finger["fromto"].split()]
                finger_root = tuple(fromto[:3])
                finger_tip = tuple(fromto[3:])
                assert finger["type"] == "capsule"
                assert float(finger["size"]) == pytest.approx(config.finger_radius_m)
                assert finger["rgba"].split()[:3] == config.hand_rgba.split()[:3]
                assert dist(hand_position, finger_root) < min(config.hand_size_xyz_m)
                finger_vectors[component] = tuple(
                    tip_value - root_value for root_value, tip_value in zip(finger_root, finger_tip)
                )

            for component, tip_offset in zip(
                VISUAL_ARM_MAIN_FINGER_COMPONENTS,
                config.finger_tip_offsets_m,
            ):
                assert sqrt(sum(value * value for value in finger_vectors[component])) == (
                    pytest.approx(tip_offset - config.finger_start_offset_m, abs=2e-6)
                )
            thumb_length = sqrt(
                (config.thumb_tip_forward_offset_m - config.thumb_root_forward_offset_m) ** 2
                + (config.thumb_tip_side_offset_m - config.thumb_root_side_offset_m) ** 2
            )
            assert sqrt(sum(value * value for value in finger_vectors["thumb"])) == pytest.approx(
                thumb_length,
                abs=2e-6,
            )
            thumb_vector = finger_vectors["thumb"]
            middle_vector = finger_vectors["middle_finger"]
            alignment = sum(
                thumb_value * middle_value
                for thumb_value, middle_value in zip(thumb_vector, middle_vector)
            ) / (
                sqrt(sum(value * value for value in thumb_vector))
                * sqrt(sum(value * value for value in middle_vector))
            )
            assert alignment < 0.9


def test_build_visual_arm_robot_xml_is_idempotent() -> None:
    robot = (
        "<mujoco><worldbody><body name='trunk_assembly'>\n"
        "        <!-- Frame trunk -->\n"
        "<site name='trunk'/></body></worldbody></mujoco>"
    )

    once = build_visual_arm_robot_xml(robot)
    twice = build_visual_arm_robot_xml(once)

    assert twice == once
    for name in VISUAL_ARM_GEOM_NAMES:
        assert twice.count(name) == 1


def test_visual_arm_overlay_preserves_official_dynamic_contract() -> None:
    robot_path = Path(
        "workspace/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml"
    )
    original = robot_path.read_text(encoding="utf-8")
    overlaid = build_visual_arm_robot_xml(original)
    original_root = ElementTree.fromstring(original)
    overlaid_root = ElementTree.fromstring(overlaid)

    def dynamics_snapshot(root: ElementTree.Element) -> list[tuple[str, dict[str, str]]]:
        return [
            (element.tag, dict(element.attrib))
            for tag in ("joint", "inertial", "actuator", "position")
            for element in root.findall(f".//{tag}")
        ]

    assert dynamics_snapshot(overlaid_root) == dynamics_snapshot(original_root)
    original_geoms = {geom.attrib.get("name") for geom in original_root.findall(".//geom")}
    added_geoms = [
        geom
        for geom in overlaid_root.findall(".//geom")
        if geom.attrib.get("name") not in original_geoms
    ]
    assert {geom.attrib["name"] for geom in added_geoms} == set(VISUAL_ARM_GEOM_NAMES)
    assert all(geom.attrib["contype"] == "0" for geom in added_geoms)
    assert all(geom.attrib["conaffinity"] == "0" for geom in added_geoms)


def test_visual_arm_poses_clear_official_leg_geometry_at_home(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_dir = Path("workspace/Open_Duck_Playground/playground/open_duck_mini_v2/xmls").resolve()
    robot_name = "open_duck_mini_v2.xml"
    (tmp_path / "assets").symlink_to(xml_dir / "assets", target_is_directory=True)
    (tmp_path / robot_name).write_text(
        build_visual_arm_robot_xml((xml_dir / robot_name).read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    scene_path = tmp_path / "scene_flat_terrain.xml"
    scene_path.write_text(
        (xml_dir / "scene_flat_terrain.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly")
    official_trunk_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == trunk_id
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) not in VISUAL_ARM_GEOM_NAMES
    ]
    assert official_trunk_geom_ids
    for mount_name in VISUAL_ARM_SHOULDER_MOUNT_NAMES:
        mount_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            mount_name,
        )
        assert (
            min(
                mujoco.mj_geomDistance(model, data, mount_geom_id, trunk_geom_id, 1.0, None)
                for trunk_geom_id in official_trunk_geom_ids
            )
            <= 0.0
        )

    leg_bodies = {
        "left": {
            "hip_roll_assembly",
            "left_roll_to_pitch_assembly",
            "knee_and_ankle_assembly",
            "knee_and_ankle_assembly_2",
            "foot_assembly",
        },
        "right": {
            "hip_roll_assembly_2",
            "right_roll_to_pitch_assembly",
            "knee_and_ankle_assembly_3",
            "knee_and_ankle_assembly_4",
            "foot_assembly_2",
        },
    }
    minimum_clearance_m = float("inf")
    for side, body_names in leg_bodies.items():
        leg_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            )
            in body_names
        ]
        assert leg_geom_ids
        for pose in VISUAL_ARM_POSES:
            for component in VISUAL_ARM_COMPONENTS:
                arm_geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    visual_arm_geom_name(side, pose, component),
                )
                for leg_geom_id in leg_geom_ids:
                    minimum_clearance_m = min(
                        minimum_clearance_m,
                        mujoco.mj_geomDistance(
                            model,
                            data,
                            arm_geom_id,
                            leg_geom_id,
                            1.0,
                            None,
                        ),
                    )

    assert minimum_clearance_m >= 0.015


def test_generate_social_eye_scene_writes_scene_and_robot_overlay(tmp_path: Path) -> None:
    base_scene = tmp_path / "scene_flat_terrain.xml"
    base_robot = tmp_path / "open_duck_mini_v2.xml"
    out_scene = tmp_path / "soridormi_social_eyes_scene.xml"
    base_scene.write_text(
        '<mujoco><include file="open_duck_mini_v2.xml"/><worldbody/></mujoco>',
        encoding="utf-8",
    )
    base_robot.write_text(
        "<mujoco><worldbody><body name='trunk_assembly'>\n"
        "        <!-- Frame trunk -->\n"
        "<body name='head_assembly'>\n"
        "                <!-- Frame head -->\n"
        "<site name='head'/></body><site name='trunk'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )

    result = generate_social_eye_scene(
        base_scene,
        out_scene,
        arm_config=VisualArmConfig(),
    )

    generated_robot = tmp_path / "soridormi_social_eyes_open_duck_mini_v2.xml"
    assert out_scene.exists()
    assert generated_robot.exists()
    assert result.eye_count == 2
    assert result.arm_geom_count == len(VISUAL_ARM_GEOM_NAMES)
    assert f'file="{generated_robot.name}"' in out_scene.read_text(encoding="utf-8")
    assert LEFT_EYE_NAME in generated_robot.read_text(encoding="utf-8")
    assert VISUAL_ARM_GEOM_NAMES[0] in generated_robot.read_text(encoding="utf-8")


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
    assert not (
        tmp_path / "soridormi_social_eyes_soridormi_social_eyes_open_duck_mini_v2.xml"
    ).exists()
    assert generated_robot.read_text(encoding="utf-8").count(LEFT_EYE_NAME) == 1
