"""Sustained social attention: the long-running ``attend_to_person`` skill.

Chromie's Social Cognition chooses the person and when attention starts, changes
posture and stops; Soridormi closes the perception loop every control frame. This
module holds the frame-level behavior (follow the person, a listening or thinking
posture, natural glance-aways and blinks) and the run's lifetime, free of I/O so it
is tested deterministically. The runtime service owns the loop, the body locks and
the simulator.

The behavior carries no meaning: it never nods or shakes the head. A glance-away is
one-sided and at least ``GLANCE_INTERVAL_S[0]`` apart from the next, so it cannot
read as a head shake; a posture change moves once and holds.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

ATTEND_TO_PERSON_SKILL_ID = "attend_to_person"
ATTENTION_POSTURES = ("listening", "thinking")

# Head (yaw, pitch) offsets from the person, in radians. Listening lowers the gaze a
# little, as express_attention's neutral cue does; thinking looks a little aside and up.
POSTURE_OFFSETS_RAD: dict[str, tuple[float, float]] = {
    "listening": (0.0, -0.07),
    "thinking": (0.12, 0.08),
}
HEAD_YAW_LIMIT_RAD = 0.55
HEAD_PITCH_RANGE_RAD = (-0.25, 0.2)
FOLLOW_VELOCITY_RADPS = 0.35
GLANCE_INTERVAL_S = (7.0, 14.0)
GLANCE_DURATION_S = (0.6, 1.1)
GLANCE_OFFSET_RAD = (0.15, -0.05)
BLINK_INTERVAL_S = (2.5, 6.0)
BLINK_CLOSED_S = 0.12
OBSERVE_INTERVAL_S = 0.2
OUT_OF_RANGE_TURN_S = 2.0
TRANSITION_HISTORY = 20

# Caller-supplied gaze geometry is never accepted: Soridormi resolves the person.
_REJECTED_ARGUMENTS = {"target_yaw_rad", "target_pitch_rad", "yaw", "pitch", "bearing_rad"}


class AttentionRequestError(ValueError):
    """An attend_to_person request that violates the skill's declared parameters."""


@dataclass(frozen=True)
class AttentionRequest:
    target_ref: str
    posture: str = "listening"
    target_lost_s: float = 5.0
    max_duration_s: float = 600.0

    @classmethod
    def from_args(cls, args: Mapping[str, Any], skill: Mapping[str, Any]) -> AttentionRequest:
        """Validate caller arguments against the catalog entry's declared parameters."""

        parameters = skill.get("parameters") or {}
        unknown = set(args) - set(parameters)
        rejected = set(args) & _REJECTED_ARGUMENTS
        if rejected:
            raise AttentionRequestError(
                "attend_to_person resolves the person itself; caller gaze geometry "
                f"is not accepted: {sorted(rejected)}"
            )
        if unknown:
            raise AttentionRequestError(f"unknown attend_to_person arguments: {sorted(unknown)}")
        target_ref = str(args.get("target_ref") or "").strip()
        if not target_ref:
            raise AttentionRequestError("attend_to_person requires a non-empty target_ref")
        posture_rule = parameters.get("posture") or {}
        posture = str(args.get("posture") or posture_rule.get("default") or "listening")
        if posture not in ATTENTION_POSTURES:
            raise AttentionRequestError(f"unsupported attention posture: {posture!r}")
        return cls(
            target_ref=target_ref,
            posture=posture,
            target_lost_s=_bounded(args, parameters, "target_lost_s"),
            max_duration_s=_bounded(args, parameters, "max_duration_s"),
        )


def _bounded(args: Mapping[str, Any], parameters: Mapping[str, Any], name: str) -> float:
    rule = parameters.get(name) or {}
    value = args.get(name, rule.get("default"))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AttentionRequestError(f"attend_to_person {name} must be a number") from exc
    if not math.isfinite(number) or not float(rule["min"]) <= number <= float(rule["max"]):
        raise AttentionRequestError(
            f"attend_to_person {name} must be between {rule['min']} and {rule['max']}"
        )
    return number


def clamp_head(yaw: float, pitch: float) -> tuple[float, float]:
    return (
        max(-HEAD_YAW_LIMIT_RAD, min(HEAD_YAW_LIMIT_RAD, yaw)),
        max(HEAD_PITCH_RANGE_RAD[0], min(HEAD_PITCH_RANGE_RAD[1], pitch)),
    )


def step_head(
    current: tuple[float, float], goal: tuple[float, float], max_step: float
) -> tuple[float, float]:
    """One frame toward ``goal`` on a straight line, no axis faster than ``max_step``.

    Both axes arrive together, so a sideways glance that also lowers the gaze moves
    diagonally and never dips the head on its own the way a nod does.
    """

    delta_yaw, delta_pitch = goal[0] - current[0], goal[1] - current[1]
    largest = max(abs(delta_yaw), abs(delta_pitch))
    if largest <= max_step:
        return goal
    scale = max_step / largest
    return current[0] + delta_yaw * scale, current[1] + delta_pitch * scale


class AttentionBehavior:
    """Head goal and blink timing for one attention run; time is seconds since start."""

    def __init__(self, posture: str, *, rng: random.Random) -> None:
        self.posture = posture
        self._rng = rng
        self._next_glance_s = rng.uniform(*GLANCE_INTERVAL_S)
        self._glance_until_s = -math.inf
        self._glance_sign = 1.0
        self._next_blink_s = rng.uniform(*BLINK_INTERVAL_S)

    def head_goal(
        self,
        now_s: float,
        bearing_rad: float | None,
        current: tuple[float, float],
    ) -> tuple[float, float]:
        """Where the head should point now; it holds when the person is not seen."""

        if bearing_rad is None:
            return current
        if now_s >= self._next_glance_s:
            self._glance_until_s = now_s + self._rng.uniform(*GLANCE_DURATION_S)
            self._glance_sign = self._rng.choice((-1.0, 1.0))
            self._next_glance_s = self._glance_until_s + self._rng.uniform(*GLANCE_INTERVAL_S)
        yaw_offset, pitch_offset = POSTURE_OFFSETS_RAD[self.posture]
        if now_s < self._glance_until_s:
            yaw_offset += self._glance_sign * GLANCE_OFFSET_RAD[0]
            pitch_offset += GLANCE_OFFSET_RAD[1]
        return clamp_head(bearing_rad + yaw_offset, pitch_offset)

    def blink_due(self, now_s: float) -> bool:
        if now_s < self._next_blink_s:
            return False
        self._next_blink_s = now_s + self._rng.uniform(*BLINK_INTERVAL_S)
        return True


class AttentionLifetime:
    """Soridormi's own ends: the person unseen for target_lost_s, or max_duration_s."""

    def __init__(self, request: AttentionRequest) -> None:
        self._request = request
        self._lost_since_s: float | None = None

    def end_reason(self, now_s: float, *, seen: bool) -> str | None:
        if now_s >= self._request.max_duration_s:
            return "max_duration"
        if seen:
            self._lost_since_s = None
            return None
        if self._lost_since_s is None:
            self._lost_since_s = now_s
        if now_s - self._lost_since_s >= self._request.target_lost_s:
            return "target_lost"
        return None


@dataclass
class AttentionRun:
    """One attend_to_person invocation, reported through ``soridormi.attention.status``."""

    attention_id: str
    request: AttentionRequest
    posture: str
    status: str = "running"
    end_reason: str | None = None
    target_status: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    suspensions: int = 0
    body_turns: int = 0
    blinks: int = 0
    motor_commands_sent: bool = False
    stop_reason: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._record("running", "started")

    @property
    def active(self) -> bool:
        return self.status != "ended"

    def request_stop(self, reason: str) -> None:
        if self.active and self.stop_reason is None:
            self.stop_reason = reason

    def mark(self, status: str, reason: str) -> None:
        if not self.active or status == self.status:
            return
        if status == "suspended":
            self.suspensions += 1
        self.status = status
        self._record(status, reason)

    def end(self, reason: str) -> None:
        if not self.active:
            return
        self.status = "ended"
        self.end_reason = reason
        self.ended_at = time.time()
        self._record("ended", reason)

    def _record(self, status: str, reason: str) -> None:
        self.transitions.append({"status": status, "reason": reason, "at": time.time()})
        del self.transitions[:-TRANSITION_HISTORY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_id": self.attention_id,
            "skill_id": ATTEND_TO_PERSON_SKILL_ID,
            "active": self.active,
            "status": self.status,
            "end_reason": self.end_reason,
            "target_ref": self.request.target_ref,
            "posture": self.posture,
            "target_status": self.target_status,
            "target_lost_s": self.request.target_lost_s,
            "max_duration_s": self.request.max_duration_s,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "suspensions": self.suspensions,
            "body_turns": self.body_turns,
            "blinks": self.blinks,
            "motor_commands_sent": self.motor_commands_sent,
            "transitions": list(self.transitions),
        }
