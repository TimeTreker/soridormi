"""Physical target resolution contract between Chromie and Soridormi.

Chromie describes *what* it wants a body skill to act on: a person by referent
(``"requester"`` is the person who made the current request), an object by
description, a place by its cited name. Soridormi resolves the actual person,
object, or place from its own perception and body state. A target Soridormi
cannot uniquely resolve is a typed, declared non-completion, never a default
direction or a guessed object.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# The agreed referent for the person who made the current request.
REQUESTER_REFERENT_ID = "requester"

# Gaze skills whose target direction Soridormi resolves itself. Callers name the
# target with ``target_ref``; yaw/pitch offsets remain operator-tool inputs only.
GAZE_TARGET_SKILLS = frozenset({"look_at_person"})
OPERATOR_GAZE_PARAMETERS = ("target_yaw_rad", "target_pitch_rad")

# Typed source bindings that cite a place Soridormi must resolve before acting.
PLACE_BINDING_ENTITY_TYPES = frozenset({"location"})

# ``target_resolution.status`` values.
TARGET_RESOLVED = "resolved"
TARGET_NOT_OBSERVED = "not_observed"
TARGET_AMBIGUOUS = "ambiguous"
TARGET_OUTSIDE_HEAD_RANGE = "outside_head_range"
TARGET_PERCEPTION_UNAVAILABLE = "perception_unavailable"
PLACE_NOT_RESOLVED = "not_resolved"

# Temporary simulation stand-in until Soridormi can perceive people: a sim gaze
# request for the requester with no scene actor resolves to a declared mocked
# position, and ``target_resolution.source`` says so.
MOCKED_REQUESTER_OBJECT_REF = "mocked_requester"
MOCKED_REQUESTER_SOURCE = "mocked_requester_position"


def scene_terms(value: str) -> frozenset[str]:
    """Identifying words used to match a description against scene markers."""

    return frozenset(
        word for word in re.findall(r"\w+", value.casefold())
        if word not in {"a", "an", "the", "of", "some"}
    )


def reject_caller_gaze_coordinates(skill_id: str, parameters: Any) -> None:
    """A caller names whom to look at; Soridormi resolves where that is."""

    if skill_id not in GAZE_TARGET_SKILLS or not isinstance(parameters, Mapping):
        return
    supplied = [name for name in OPERATOR_GAZE_PARAMETERS if name in parameters]
    if supplied:
        raise ValueError(
            f"{skill_id} resolves target_ref from Soridormi perception; "
            f"{', '.join(supplied)} are operator-tool inputs, not caller arguments"
        )


def cited_source_places(parameters: Mapping[str, Any]) -> list[str]:
    """Return the places a resource request cites in typed ``source.bindings``.

    Distance, direction, and route bindings are not places; they stay request
    context. A binding is a place when its ``entity_type`` is ``location`` or,
    for an untyped binding, when its name is ``location`` or ``place``.
    """

    source = parameters.get("source")
    bindings = source.get("bindings") if isinstance(source, Mapping) else None
    if not isinstance(bindings, Mapping):
        return []
    places: list[str] = []
    for name, binding in bindings.items():
        if isinstance(binding, Mapping):
            entity_type = binding.get("entity_type")
            is_place = (
                entity_type in PLACE_BINDING_ENTITY_TYPES
                if entity_type is not None
                else str(name) in {"location", "place"}
            )
            value = binding.get("value")
        else:
            is_place = str(name) in {"location", "place"}
            value = binding
        if is_place:
            places.append(" ".join(str(value if value is not None else name).split()))
    return places


def gaze_target_label(target_ref: str) -> str:
    return "the requester" if target_ref == REQUESTER_REFERENT_ID else repr(target_ref)


def gaze_target_unavailable_result(resolution: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Declared non-completion for a gaze target Soridormi could not resolve."""

    status = str(resolution.get("status"))
    target = gaze_target_label(str(resolution.get("target_ref") or ""))
    if status == TARGET_NOT_OBSERVED:
        seen = f"did not observe {target} in the scene"
    elif status == TARGET_AMBIGUOUS:
        seen = f"observed more than one target matching {target} in the scene"
    elif status == TARGET_OUTSIDE_HEAD_RANGE:
        where = str(resolution.get("relative_direction") or "").strip()
        seen = (
            f"observed {target}{' ' + where if where else ''}, beyond the head's "
            "turning range"
        )
    else:
        seen = f"has no perception in {mode} mode to locate {target}"
    return {
        "completed": False,
        "no_motion": True,
        "summary": f"Soridormi {seen}, so it did not look.",
        "target_resolution": dict(resolution),
    }
