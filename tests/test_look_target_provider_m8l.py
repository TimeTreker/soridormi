from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from soridormi_runtime.look_target_provider import (
    DEFAULT_HORIZONTAL_FOV_RAD,
    DEFAULT_VERTICAL_FOV_RAD,
    LOOK_AT_PERSON_PITCH_MAX_RAD,
    LOOK_AT_PERSON_PITCH_MIN_RAD,
    LOOK_AT_PERSON_YAW_MAX_RAD,
    LOOK_AT_PERSON_YAW_MIN_RAD,
    image_point_to_yaw_pitch,
    make_image_point_target,
    make_manual_target,
    make_target_from_json,
    resolve_target_from_mapping,
)
from soridormi_runtime.skill_execution import SkillExecutionError


def test_manual_target_clamps_to_look_at_person_bounds() -> None:
    target = make_manual_target(target_yaw_rad=2.0, target_pitch_rad=-2.0, target_ref="person_left", confidence=1.5)

    assert target.source == "manual_yaw_pitch"
    assert target.target_ref == "person_left"
    assert target.yaw_rad == pytest.approx(LOOK_AT_PERSON_YAW_MAX_RAD)
    assert target.pitch_rad == pytest.approx(LOOK_AT_PERSON_PITCH_MIN_RAD)
    assert target.confidence == pytest.approx(1.0)
    skill_args = target.to_skill_args(duration_s=4.0, hold_fraction=0.5)
    assert skill_args["target_ref"] == "person_left"
    assert skill_args["target_yaw_rad"] == pytest.approx(LOOK_AT_PERSON_YAW_MAX_RAD)
    assert skill_args["target_pitch_rad"] == pytest.approx(LOOK_AT_PERSON_PITCH_MIN_RAD)
    assert skill_args["duration_s"] == pytest.approx(4.0)
    assert skill_args["hold_fraction"] == pytest.approx(0.5)
    assert skill_args["end_mode"] == "hold_target"


def test_target_skill_args_can_request_return_to_neutral() -> None:
    target = make_manual_target(target_yaw_rad=0.2, target_pitch_rad=-0.04)

    skill_args = target.to_skill_args(duration_s=5.0, hold_fraction=0.4, end_mode="return_neutral")

    assert skill_args["target_yaw_rad"] == pytest.approx(0.2)
    assert skill_args["target_pitch_rad"] == pytest.approx(-0.04)
    assert skill_args["end_mode"] == "return_neutral"


def test_image_center_maps_to_zero_yaw_and_pitch() -> None:
    yaw, pitch = image_point_to_yaw_pitch(image_x_norm=0.5, image_y_norm=0.5)

    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_image_point_maps_x_to_yaw_and_y_to_pitch() -> None:
    yaw, pitch = image_point_to_yaw_pitch(
        image_x_norm=0.75,
        image_y_norm=0.25,
        horizontal_fov_rad=math.radians(60.0),
        vertical_fov_rad=math.radians(40.0),
    )

    assert yaw == pytest.approx(math.radians(15.0))
    assert pitch == pytest.approx(math.radians(10.0))


def test_image_point_is_clamped_to_skill_bounds() -> None:
    yaw, pitch = image_point_to_yaw_pitch(
        image_x_norm=1.0,
        image_y_norm=0.0,
        horizontal_fov_rad=math.pi,
        vertical_fov_rad=math.pi,
    )

    assert yaw == pytest.approx(LOOK_AT_PERSON_YAW_MAX_RAD)
    assert pitch == pytest.approx(LOOK_AT_PERSON_PITCH_MAX_RAD)


def test_make_image_point_target_keeps_source_payload() -> None:
    target = make_image_point_target(image_x_norm=0.25, image_y_norm=0.75, confidence=0.7)

    assert target.source == "image_point_stub"
    assert target.target_ref == "person"
    assert target.yaw_rad < 0.0
    assert target.pitch_rad < 0.0
    assert target.confidence == pytest.approx(0.7)
    assert target.source_payload["horizontal_fov_rad"] == pytest.approx(DEFAULT_HORIZONTAL_FOV_RAD)
    assert target.source_payload["vertical_fov_rad"] == pytest.approx(DEFAULT_VERTICAL_FOV_RAD)


def test_target_from_json_accepts_manual_and_image_payloads(tmp_path: Path) -> None:
    manual = make_target_from_json(json.dumps({"target_ref": "speaker", "target_yaw_rad": 0.2, "target_pitch_rad": -0.04}))
    assert manual.target_ref == "speaker"
    assert manual.yaw_rad == pytest.approx(0.2)
    assert manual.pitch_rad == pytest.approx(-0.04)

    fixture = tmp_path / "target.json"
    fixture.write_text(json.dumps({"image_x_norm": 0.75, "image_y_norm": 0.5, "confidence": 0.8}), encoding="utf-8")
    image_target = make_target_from_json(str(fixture))
    assert image_target.source == "image_point_stub"
    assert image_target.confidence == pytest.approx(0.8)
    assert image_target.yaw_rad > 0.0


def test_resolve_target_requires_exactly_one_source() -> None:
    with pytest.raises(SkillExecutionError, match="exactly one target source"):
        resolve_target_from_mapping({})
    with pytest.raises(SkillExecutionError, match="exactly one target source"):
        resolve_target_from_mapping({"target_json": "{}", "target_yaw_rad": 0.1})


def test_resolve_target_from_mapping_accepts_manual_zero_values() -> None:
    target = resolve_target_from_mapping({"target_yaw_rad": 0.0, "target_pitch_rad": 0.0, "target_ref": "person"})

    assert target.source == "manual_yaw_pitch"
    assert target.yaw_rad == pytest.approx(0.0)
    assert target.pitch_rad == pytest.approx(0.0)


def test_image_norm_validation_rejects_out_of_range() -> None:
    with pytest.raises(SkillExecutionError, match="image_x_norm must be in"):
        image_point_to_yaw_pitch(image_x_norm=1.2, image_y_norm=0.5)


def test_look_at_person_target_cli_resolve_only_json_manual() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.look_at_person_target",
            "--target-yaw-rad",
            "0.30",
            "--target-pitch-rad",
            "-0.06",
            "--duration-s",
            "4.0",
            "--resolve-only",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["target"]["source"] == "manual_yaw_pitch"
    assert payload["skill_args"]["target_yaw_rad"] == pytest.approx(0.30)
    assert payload["skill_args"]["target_pitch_rad"] == pytest.approx(-0.06)
    assert payload["skill_args"]["end_mode"] == "hold_target"
    assert payload["plan"]["skill_id"] == "look_at_person"


def test_look_at_person_target_cli_dry_run_image_point() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.look_at_person_target",
            "--image-x-norm",
            "0.75",
            "--image-y-norm",
            "0.45",
            "--duration-s",
            "4.0",
            "--end-mode",
            "return_neutral",
            "--backend",
            "mujoco",
            "--control-hz",
            "20",
            "--dry-run",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["target"]["source"] == "image_point_stub"
    assert payload["target"]["yaw_rad"] > 0.0
    assert payload["target"]["pitch_rad"] > 0.0
    assert payload["plan"]["skill_id"] == "look_at_person"
    assert payload["skill_args"]["end_mode"] == "return_neutral"
    assert payload["result"]["executed"] is False
    assert payload["result"]["target_max_positions_by_name"]["head_yaw"] > 0.0
    assert payload["result"]["target_positions_by_name"]["head_yaw"] == pytest.approx(0.0)
