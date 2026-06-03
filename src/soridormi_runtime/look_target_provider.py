"""Structured look-target provider utilities for scripted social skills.

This module is deliberately not a detector.  It converts already-structured
upstream target hints (manual yaw/pitch, image-space point, or JSON fixture) into
bounded yaw/pitch offsets that the ``look_at_person`` scripted skill can execute.
Future camera/person detection should plug in *above* this boundary and provide
structured target measurements instead of calling the low-level head executor
with raw images or language.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .skill_execution import SkillExecutionError


LOOK_AT_PERSON_YAW_MIN_RAD = -0.55
LOOK_AT_PERSON_YAW_MAX_RAD = 0.55
LOOK_AT_PERSON_PITCH_MIN_RAD = -0.25
LOOK_AT_PERSON_PITCH_MAX_RAD = 0.20
DEFAULT_HORIZONTAL_FOV_RAD = math.radians(60.0)
DEFAULT_VERTICAL_FOV_RAD = math.radians(45.0)


@dataclass(frozen=True)
class LookTarget:
    """Structured target direction for ``look_at_person``.

    ``yaw_rad`` and ``pitch_rad`` are offsets in the robot/head camera frame.
    Positive yaw means look to the robot's left/right according to the existing
    head actuator convention used by ``look_direction``; positive pitch means
    look upward.  ``source`` documents where the target came from, but the
    scripted executor only receives bounded numeric offsets.
    """

    target_ref: str
    yaw_rad: float
    pitch_rad: float
    confidence: float
    source: str
    source_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_skill_args(
        self,
        *,
        duration_s: float = 4.0,
        hold_fraction: float = 0.5,
        end_mode: str = "hold_target",
    ) -> dict[str, Any]:
        return {
            "target_ref": self.target_ref,
            "target_yaw_rad": self.yaw_rad,
            "target_pitch_rad": self.pitch_rad,
            "duration_s": float(duration_s),
            "hold_fraction": float(hold_fraction),
            "end_mode": str(end_mode),
        }


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SkillExecutionError(f"{name} must be numeric, got boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SkillExecutionError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SkillExecutionError(f"{name} must be finite")
    return result


def clamp_target_yaw(yaw_rad: float) -> float:
    return max(LOOK_AT_PERSON_YAW_MIN_RAD, min(LOOK_AT_PERSON_YAW_MAX_RAD, float(yaw_rad)))


def clamp_target_pitch(pitch_rad: float) -> float:
    return max(LOOK_AT_PERSON_PITCH_MIN_RAD, min(LOOK_AT_PERSON_PITCH_MAX_RAD, float(pitch_rad)))


def clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, float(confidence)))


def validate_image_norm(name: str, value: Any) -> float:
    number = _coerce_float(name, value)
    if not 0.0 <= number <= 1.0:
        raise SkillExecutionError(f"{name} must be in [0.0, 1.0]")
    return number


def image_point_to_yaw_pitch(
    *,
    image_x_norm: float,
    image_y_norm: float,
    horizontal_fov_rad: float = DEFAULT_HORIZONTAL_FOV_RAD,
    vertical_fov_rad: float = DEFAULT_VERTICAL_FOV_RAD,
) -> tuple[float, float]:
    """Convert a normalized image point to bounded yaw/pitch offsets.

    ``image_x_norm`` and ``image_y_norm`` are in image coordinates where
    ``(0.5, 0.5)`` is the center and image y increases downward.  A target above
    the center therefore produces positive pitch (look up), while a target below
    center produces negative pitch (look down).
    """

    x = validate_image_norm("image_x_norm", image_x_norm)
    y = validate_image_norm("image_y_norm", image_y_norm)
    hfov = _coerce_float("horizontal_fov_rad", horizontal_fov_rad)
    vfov = _coerce_float("vertical_fov_rad", vertical_fov_rad)
    if hfov <= 0.0 or hfov > math.pi:
        raise SkillExecutionError("horizontal_fov_rad must be in (0, pi]")
    if vfov <= 0.0 or vfov > math.pi:
        raise SkillExecutionError("vertical_fov_rad must be in (0, pi]")

    yaw = (x - 0.5) * hfov
    pitch = -(y - 0.5) * vfov
    return clamp_target_yaw(yaw), clamp_target_pitch(pitch)


def make_manual_target(
    *,
    target_yaw_rad: Any,
    target_pitch_rad: Any,
    target_ref: str = "person",
    confidence: Any = 1.0,
) -> LookTarget:
    yaw = clamp_target_yaw(_coerce_float("target_yaw_rad", target_yaw_rad))
    pitch = clamp_target_pitch(_coerce_float("target_pitch_rad", target_pitch_rad))
    return LookTarget(
        target_ref=str(target_ref or "person"),
        yaw_rad=yaw,
        pitch_rad=pitch,
        confidence=clamp_confidence(_coerce_float("confidence", confidence)),
        source="manual_yaw_pitch",
        source_payload={"target_yaw_rad": yaw, "target_pitch_rad": pitch},
    )


def make_image_point_target(
    *,
    image_x_norm: Any,
    image_y_norm: Any,
    target_ref: str = "person",
    confidence: Any = 1.0,
    horizontal_fov_rad: Any = DEFAULT_HORIZONTAL_FOV_RAD,
    vertical_fov_rad: Any = DEFAULT_VERTICAL_FOV_RAD,
) -> LookTarget:
    hfov = _coerce_float("horizontal_fov_rad", horizontal_fov_rad)
    vfov = _coerce_float("vertical_fov_rad", vertical_fov_rad)
    yaw, pitch = image_point_to_yaw_pitch(
        image_x_norm=validate_image_norm("image_x_norm", image_x_norm),
        image_y_norm=validate_image_norm("image_y_norm", image_y_norm),
        horizontal_fov_rad=hfov,
        vertical_fov_rad=vfov,
    )
    return LookTarget(
        target_ref=str(target_ref or "person"),
        yaw_rad=yaw,
        pitch_rad=pitch,
        confidence=clamp_confidence(_coerce_float("confidence", confidence)),
        source="image_point_stub",
        source_payload={
            "image_x_norm": float(image_x_norm),
            "image_y_norm": float(image_y_norm),
            "horizontal_fov_rad": hfov,
            "vertical_fov_rad": vfov,
        },
    )


def _load_json_object(value: str) -> dict[str, Any]:
    text = str(value)
    maybe_path = Path(text)
    if maybe_path.exists():
        payload = json.loads(maybe_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SkillExecutionError("target JSON must be an object")
    return payload


def make_target_from_json(value: str) -> LookTarget:
    payload = _load_json_object(value)
    target_ref = str(payload.get("target_ref", payload.get("ref", "person")) or "person")
    confidence = payload.get("confidence", 1.0)

    if "target_yaw_rad" in payload or "target_pitch_rad" in payload:
        return make_manual_target(
            target_yaw_rad=payload.get("target_yaw_rad", 0.0),
            target_pitch_rad=payload.get("target_pitch_rad", 0.0),
            target_ref=target_ref,
            confidence=confidence,
        )

    if "yaw_rad" in payload or "pitch_rad" in payload:
        return make_manual_target(
            target_yaw_rad=payload.get("yaw_rad", 0.0),
            target_pitch_rad=payload.get("pitch_rad", 0.0),
            target_ref=target_ref,
            confidence=confidence,
        )

    if "image_x_norm" in payload and "image_y_norm" in payload:
        return make_image_point_target(
            image_x_norm=payload["image_x_norm"],
            image_y_norm=payload["image_y_norm"],
            target_ref=target_ref,
            confidence=confidence,
            horizontal_fov_rad=payload.get("horizontal_fov_rad", DEFAULT_HORIZONTAL_FOV_RAD),
            vertical_fov_rad=payload.get("vertical_fov_rad", DEFAULT_VERTICAL_FOV_RAD),
        )

    raise SkillExecutionError(
        "target JSON must contain target_yaw_rad/target_pitch_rad, yaw_rad/pitch_rad, "
        "or image_x_norm/image_y_norm"
    )


def resolve_target_from_mapping(options: Mapping[str, Any]) -> LookTarget:
    """Resolve a target from CLI-like options.

    Exactly one source should be supplied: target_json, manual yaw/pitch, or an
    image-space point.  This keeps upstream perception/planner boundaries
    explicit and avoids hidden fallbacks.
    """

    has_json = bool(options.get("target_json"))
    has_manual = options.get("target_yaw_rad") is not None or options.get("target_pitch_rad") is not None
    has_image = options.get("image_x_norm") is not None or options.get("image_y_norm") is not None
    sources = [has_json, has_manual, has_image]
    if sum(1 for item in sources if item) != 1:
        raise SkillExecutionError(
            "provide exactly one target source: --target-json, --target-yaw-rad/--target-pitch-rad, "
            "or --image-x-norm/--image-y-norm"
        )

    target_ref = str(options.get("target_ref", "person") or "person")
    confidence = options.get("confidence", 1.0)
    if has_json:
        return make_target_from_json(str(options["target_json"]))
    if has_manual:
        return make_manual_target(
            target_yaw_rad=options.get("target_yaw_rad", 0.0),
            target_pitch_rad=options.get("target_pitch_rad", 0.0),
            target_ref=target_ref,
            confidence=confidence,
        )
    return make_image_point_target(
        image_x_norm=options.get("image_x_norm"),
        image_y_norm=options.get("image_y_norm"),
        target_ref=target_ref,
        confidence=confidence,
        horizontal_fov_rad=options.get("horizontal_fov_rad", DEFAULT_HORIZONTAL_FOV_RAD),
        vertical_fov_rad=options.get("vertical_fov_rad", DEFAULT_VERTICAL_FOV_RAD),
    )
