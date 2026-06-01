from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class StrideStepThresholds:
    min_forward_speed_mps: float = 0.02
    max_stuck_sample_ratio: float = 0.40
    min_base_z_m: float = 0.12
    max_abs_roll_pitch_rad: float = 0.90
    min_touchdown_count: int = 4
    min_step_length_m: float = 0.01
    max_low_clearance_ratio: float = 0.35
    min_swing_clearance_m: float = 0.015
    contact_threshold: float = 0.5

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class StrideStepReport:
    ok: bool
    log_path: str
    sample_count: int
    samples_with_base: int
    samples_with_feet: int
    thresholds: dict[str, float | int]
    duration_s: float | None
    base_motion: dict[str, Any]
    foot_clearance: dict[str, Any]
    step_events: dict[str, Any]
    stuck: dict[str, Any]
    fall: dict[str, Any]
    scenario: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeSample:
    step_index: int | None = None
    time_s: float | None = None
    base_xyz: list[float] | None = None
    base_quat_wxyz: list[float] | None = None
    feet_xyz: list[list[float]] | None = None
    feet_contacts: list[float] | None = None
    scenario_id: str | None = None
    skill_id: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def _float_list(value: Any, *, min_len: int = 0) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < min_len:
        return None
    out: list[float] = []
    for item in value:
        converted = _as_float(item)
        if converted is None:
            return None
        out.append(float(converted))
    return out


def _feet_list(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    feet: list[list[float]] = []
    for item in value:
        parsed = _float_list(item, min_len=3)
        if parsed is None:
            return None
        feet.append(parsed[:3])
    return feet


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _nested_bool(payload: dict[str, Any], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        value: Any = payload
        found = True
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                found = False
                break
            value = value[part]
        if found:
            parsed = _as_bool(value)
            if parsed is not None:
                return parsed
    return None


def _extract_sample(payload: dict[str, Any]) -> RuntimeSample | None:
    state = payload.get("state")
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    if not isinstance(state, dict) and not isinstance(debug, dict):
        return None

    sample = RuntimeSample()

    raw_step = payload.get("step_index")
    if raw_step is None and isinstance(debug, dict):
        raw_step = debug.get("step_count")
    try:
        sample.step_index = int(raw_step) if raw_step is not None else None
    except (TypeError, ValueError):
        sample.step_index = None

    for candidate in (
        payload.get("robot_time"),
        state.get("time") if isinstance(state, dict) else None,
        debug.get("robot_time") if isinstance(debug, dict) else None,
        payload.get("time_s"),
    ):
        converted = _as_float(candidate)
        if converted is not None:
            sample.time_s = converted
            break

    if isinstance(state, dict):
        sample.base_xyz = _float_list(state.get("base_position_xyz"), min_len=3)
        sample.base_quat_wxyz = _float_list(state.get("base_quat_wxyz"), min_len=4)
        sample.feet_xyz = _feet_list(state.get("feet_position_xyz"))
        contacts = _float_list(state.get("feet_contacts"), min_len=2)
        sample.feet_contacts = contacts[:2] if contacts is not None else None

    # Scenario-aware collector rows and future rollout/eval rows can carry these at top-level
    # or inside task/environment context. Keep the parser permissive so old logs still work.
    sample.scenario_id = _first_string(
        payload.get("scenario_id"),
        payload.get("scenario"),
        debug.get("scenario_id") if isinstance(debug, dict) else None,
        (payload.get("task_context") or {}).get("scenario_id")
        if isinstance(payload.get("task_context"), dict)
        else None,
    )
    sample.skill_id = _first_string(
        payload.get("skill_id"),
        debug.get("skill_id") if isinstance(debug, dict) else None,
        (payload.get("task_context") or {}).get("skill_id")
        if isinstance(payload.get("task_context"), dict)
        else None,
    )

    for flag_name, keys in {
        "fallen": ("fallen", "fall", "is_fallen", "failure.fallen", "state.fallen"),
        "stuck": ("stuck", "is_stuck", "failure.stuck", "state.stuck"),
        "terminated": ("terminated", "done", "failure.terminated"),
        "failure": ("failure", "failed", "failure.failed"),
    }.items():
        parsed = _nested_bool(payload, keys)
        if parsed is not None:
            sample.flags[flag_name] = parsed

    if sample.base_xyz is None and sample.feet_xyz is None and sample.time_s is None:
        return None
    return sample


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[low]
    weight = rank - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _stats(values: list[float], suffix: str = "") -> dict[str, float | int | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            f"min{suffix}": None,
            f"p05{suffix}": None,
            f"p50{suffix}": None,
            f"mean{suffix}": None,
            f"p95{suffix}": None,
            f"max{suffix}": None,
        }
    return {
        "count": len(clean),
        f"min{suffix}": min(clean),
        f"p05{suffix}": _percentile(clean, 5.0),
        f"p50{suffix}": _percentile(clean, 50.0),
        f"mean{suffix}": sum(clean) / len(clean),
        f"p95{suffix}": _percentile(clean, 95.0),
        f"max{suffix}": max(clean),
    }


def _duration(samples: list[RuntimeSample], fallback_control_hz: float | None) -> float | None:
    times = [sample.time_s for sample in samples if sample.time_s is not None]
    if len(times) >= 2:
        duration = max(times) - min(times)
        if duration > 0.0:
            return float(duration)
    if fallback_control_hz and fallback_control_hz > 0 and len(samples) >= 2:
        return float((len(samples) - 1) / fallback_control_hz)
    return None


def _base_motion(samples: list[RuntimeSample], duration_s: float | None) -> tuple[dict[str, Any], list[float]]:
    base_samples = [sample for sample in samples if sample.base_xyz is not None]
    velocities: list[float] = []
    if len(base_samples) >= 2:
        previous = base_samples[0]
        for sample in base_samples[1:]:
            assert sample.base_xyz is not None
            assert previous.base_xyz is not None
            dt = None
            if sample.time_s is not None and previous.time_s is not None:
                dt = sample.time_s - previous.time_s
            if dt is not None and dt > 0.0:
                velocities.append((sample.base_xyz[0] - previous.base_xyz[0]) / dt)
            previous = sample

    if len(base_samples) < 2:
        return (
            {
                "available": False,
                "start_xyz": None,
                "end_xyz": None,
                "delta_xyz": None,
                "forward_x_m": None,
                "lateral_y_m": None,
                "vertical_z_m": None,
                "horizontal_distance_m": None,
                "mean_forward_speed_mps": None,
                "instant_forward_speed_mps": _stats(velocities, "_mps"),
            },
            velocities,
        )

    start = base_samples[0].base_xyz or [0.0, 0.0, 0.0]
    end = base_samples[-1].base_xyz or [0.0, 0.0, 0.0]
    delta = [float(end[i] - start[i]) for i in range(3)]
    horizontal = math.hypot(delta[0], delta[1])
    mean_speed = delta[0] / duration_s if duration_s and duration_s > 0.0 else None
    return (
        {
            "available": True,
            "start_xyz": start[:3],
            "end_xyz": end[:3],
            "delta_xyz": delta,
            "forward_x_m": delta[0],
            "lateral_y_m": delta[1],
            "vertical_z_m": delta[2],
            "horizontal_distance_m": horizontal,
            "mean_forward_speed_mps": mean_speed,
            "instant_forward_speed_mps": _stats(velocities, "_mps"),
        },
        velocities,
    )


def _foot_clearance(samples: list[RuntimeSample], thresholds: StrideStepThresholds) -> dict[str, Any]:
    all_clearances: list[float] = []
    swing_clearances: list[float] = []
    low_swing = 0
    samples_with_feet = 0
    for sample in samples:
        if sample.feet_xyz is None:
            continue
        samples_with_feet += 1
        contacts = sample.feet_contacts or [0.0, 0.0]
        for foot_index in (0, 1):
            z = sample.feet_xyz[foot_index][2]
            all_clearances.append(z)
            contact = contacts[foot_index] >= thresholds.contact_threshold
            if not contact:
                swing_clearances.append(z)
                if z < thresholds.min_swing_clearance_m:
                    low_swing += 1
    swing_count = len(swing_clearances)
    low_ratio = float(low_swing) / float(swing_count) if swing_count else None
    return {
        "samples_with_feet": samples_with_feet,
        "all": _stats(all_clearances, "_m"),
        "swing": _stats(swing_clearances, "_m"),
        "low_clearance_swing_steps": low_swing,
        "low_clearance_swing_ratio": low_ratio,
    }


def _touchdown_events(samples: list[RuntimeSample], thresholds: StrideStepThresholds) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_contacts: list[bool] | None = None
    foot_names = ["left", "right"]
    for sample in samples:
        if sample.feet_contacts is None:
            continue
        contacts = [value >= thresholds.contact_threshold for value in sample.feet_contacts[:2]]
        if previous_contacts is None:
            previous_contacts = contacts
            continue
        for foot_index, is_contact in enumerate(contacts):
            if is_contact and not previous_contacts[foot_index]:
                event: dict[str, Any] = {
                    "foot": foot_names[foot_index],
                    "foot_index": foot_index,
                    "step_index": sample.step_index,
                    "time_s": sample.time_s,
                    "base_x_m": sample.base_xyz[0] if sample.base_xyz is not None else None,
                    "base_y_m": sample.base_xyz[1] if sample.base_xyz is not None else None,
                    "foot_xyz": sample.feet_xyz[foot_index] if sample.feet_xyz is not None else None,
                }
                events.append(event)
        previous_contacts = contacts
    return events


def _step_event_summary(samples: list[RuntimeSample], duration_s: float | None, thresholds: StrideStepThresholds) -> dict[str, Any]:
    events = _touchdown_events(samples, thresholds)
    left = [event for event in events if event["foot"] == "left"]
    right = [event for event in events if event["foot"] == "right"]
    touchdown_intervals: list[float] = []
    alternating = 0
    step_lengths: list[float] = []
    base_progress_between_touchdowns: list[float] = []

    for previous, current in zip(events, events[1:]):
        if previous.get("time_s") is not None and current.get("time_s") is not None:
            interval = float(current["time_s"]) - float(previous["time_s"])
            if interval > 0.0:
                touchdown_intervals.append(interval)
        if previous.get("foot") != current.get("foot"):
            alternating += 1
        prev_foot = previous.get("foot_xyz")
        curr_foot = current.get("foot_xyz")
        if isinstance(prev_foot, list) and isinstance(curr_foot, list):
            step_lengths.append(abs(float(curr_foot[0]) - float(prev_foot[0])))
        if previous.get("base_x_m") is not None and current.get("base_x_m") is not None:
            base_progress_between_touchdowns.append(float(current["base_x_m"]) - float(previous["base_x_m"]))

    same_foot_strides: list[float] = []
    for side_events in (left, right):
        for previous, current in zip(side_events, side_events[1:]):
            if previous.get("base_x_m") is not None and current.get("base_x_m") is not None:
                same_foot_strides.append(float(current["base_x_m"]) - float(previous["base_x_m"]))

    return {
        "touchdown_count": len(events),
        "left_touchdown_count": len(left),
        "right_touchdown_count": len(right),
        "cadence_steps_per_s": (len(events) / duration_s if duration_s and duration_s > 0.0 else None),
        "touchdown_interval_s": _stats(touchdown_intervals, "_s"),
        "alternating_touchdown_ratio": (alternating / (len(events) - 1) if len(events) > 1 else None),
        "step_length_m": _stats(step_lengths, "_m"),
        "base_progress_per_touchdown_m": _stats(base_progress_between_touchdowns, "_m"),
        "same_foot_stride_progress_m": _stats(same_foot_strides, "_m"),
        "events_preview": events[:20],
    }


def _stuck_summary(
    samples: list[RuntimeSample],
    forward_velocities: list[float],
    thresholds: StrideStepThresholds,
) -> dict[str, Any]:
    explicit_stuck = sum(1 for sample in samples if sample.flags.get("stuck") is True)
    if forward_velocities:
        low = [abs(value) < thresholds.min_forward_speed_mps for value in forward_velocities]
        ratio = sum(1 for item in low if item) / len(low)
    else:
        ratio = None
    return {
        "explicit_stuck_samples": explicit_stuck,
        "speed_based_stuck_sample_ratio": ratio,
        "speed_threshold_mps": thresholds.min_forward_speed_mps,
        "detected": bool(explicit_stuck) or (ratio is not None and ratio > thresholds.max_stuck_sample_ratio),
    }


def _quat_roll_pitch(q: list[float]) -> tuple[float, float]:
    w, x, y, z = q[:4]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def _fall_summary(samples: list[RuntimeSample], thresholds: StrideStepThresholds) -> dict[str, Any]:
    explicit = sum(1 for sample in samples if sample.flags.get("fallen") is True or sample.flags.get("failure") is True)
    base_z_values = [sample.base_xyz[2] for sample in samples if sample.base_xyz is not None]
    low_base = sum(1 for z in base_z_values if z < thresholds.min_base_z_m)
    rolls: list[float] = []
    pitches: list[float] = []
    for sample in samples:
        if sample.base_quat_wxyz is None:
            continue
        roll, pitch = _quat_roll_pitch(sample.base_quat_wxyz)
        rolls.append(roll)
        pitches.append(pitch)
    max_abs_roll = max((abs(value) for value in rolls), default=None)
    max_abs_pitch = max((abs(value) for value in pitches), default=None)
    orientation_fall = (
        (max_abs_roll is not None and max_abs_roll > thresholds.max_abs_roll_pitch_rad)
        or (max_abs_pitch is not None and max_abs_pitch > thresholds.max_abs_roll_pitch_rad)
    )
    return {
        "detected": bool(explicit) or bool(low_base) or orientation_fall,
        "explicit_fall_or_failure_samples": explicit,
        "low_base_z_samples": low_base,
        "min_base_z_m": min(base_z_values) if base_z_values else None,
        "base_z_threshold_m": thresholds.min_base_z_m,
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
        "orientation_threshold_rad": thresholds.max_abs_roll_pitch_rad,
    }


def evaluate_stride_step_metrics(
    log_path: str | Path,
    *,
    thresholds: StrideStepThresholds | None = None,
    fallback_control_hz: float | None = 50.0,
) -> StrideStepReport:
    path = Path(log_path)
    cfg = thresholds or StrideStepThresholds()
    if not path.exists():
        raise FileNotFoundError(path)

    samples = [sample for payload in _iter_jsonl(path) if (sample := _extract_sample(payload)) is not None]
    duration_s = _duration(samples, fallback_control_hz)
    base_motion, forward_velocities = _base_motion(samples, duration_s)
    clearance = _foot_clearance(samples, cfg)
    steps = _step_event_summary(samples, duration_s, cfg)
    stuck = _stuck_summary(samples, forward_velocities, cfg)
    fall = _fall_summary(samples, cfg)

    warnings: list[str] = []
    errors: list[str] = []
    samples_with_base = sum(1 for sample in samples if sample.base_xyz is not None)
    samples_with_feet = sum(1 for sample in samples if sample.feet_xyz is not None)
    scenario_ids = sorted({sample.scenario_id for sample in samples if sample.scenario_id})
    skill_ids = sorted({sample.skill_id for sample in samples if sample.skill_id})

    if not samples:
        errors.append("log contains no runtime samples with state/debug data")
    if samples_with_base < 2:
        errors.append("log contains fewer than two state.base_position_xyz samples")
    if samples_with_feet == 0:
        warnings.append("log contains no state.feet_position_xyz samples; clearance and touchdown metrics are unavailable")
    if duration_s is None or duration_s <= 0.0:
        errors.append("unable to determine positive rollout duration from state time or fallback control_hz")

    mean_speed = base_motion.get("mean_forward_speed_mps")
    if mean_speed is not None and mean_speed < cfg.min_forward_speed_mps:
        warnings.append(
            "mean forward speed is below threshold: "
            f"{mean_speed:.4f} m/s < {cfg.min_forward_speed_mps:.4f} m/s"
        )
    if stuck.get("detected"):
        warnings.append("stuck behavior detected from explicit flags or low forward-speed ratio")
    if fall.get("detected"):
        errors.append("fall/failure detected from explicit flags, low base height, or roll/pitch threshold")

    touchdown_count = int(steps.get("touchdown_count") or 0)
    if samples_with_feet and touchdown_count < cfg.min_touchdown_count:
        warnings.append(
            "too few touchdown events for stride analysis: "
            f"{touchdown_count} < {cfg.min_touchdown_count}"
        )
    step_length_mean = steps.get("step_length_m", {}).get("mean_m")
    if step_length_mean is not None and float(step_length_mean) < cfg.min_step_length_m:
        warnings.append(
            "mean touchdown step length is below threshold: "
            f"{float(step_length_mean):.4f} m < {cfg.min_step_length_m:.4f} m"
        )
    low_clearance_ratio = clearance.get("low_clearance_swing_ratio")
    if low_clearance_ratio is not None and float(low_clearance_ratio) > cfg.max_low_clearance_ratio:
        warnings.append(
            "low swing-clearance ratio is high: "
            f"{float(low_clearance_ratio):.3f} > {cfg.max_low_clearance_ratio:.3f}"
        )

    ok = not errors
    return StrideStepReport(
        ok=ok,
        log_path=str(path),
        sample_count=len(samples),
        samples_with_base=samples_with_base,
        samples_with_feet=samples_with_feet,
        thresholds=cfg.as_dict(),
        duration_s=duration_s,
        base_motion=base_motion,
        foot_clearance=clearance,
        step_events=steps,
        stuck=stuck,
        fall=fall,
        scenario={"scenario_ids": scenario_ids, "skill_ids": skill_ids},
        warnings=warnings,
        errors=errors,
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def render_markdown(report: StrideStepReport) -> str:
    base = report.base_motion
    steps = report.step_events
    clearance = report.foot_clearance
    lines = [
        "# Soridormi stride/step metrics report",
        "",
        f"Result: {'PASS' if report.ok else 'FAILED'}",
        f"Log: {report.log_path}",
        f"Samples: {report.sample_count}",
        f"Duration: {_format_value(report.duration_s)} s",
        f"Scenario IDs: {', '.join(report.scenario['scenario_ids']) if report.scenario['scenario_ids'] else 'n/a'}",
        f"Skill IDs: {', '.join(report.scenario['skill_ids']) if report.scenario['skill_ids'] else 'n/a'}",
        "",
        "## Motion summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| forward_x_m | {_format_value(base.get('forward_x_m'))} |",
        f"| lateral_y_m | {_format_value(base.get('lateral_y_m'))} |",
        f"| horizontal_distance_m | {_format_value(base.get('horizontal_distance_m'))} |",
        f"| mean_forward_speed_mps | {_format_value(base.get('mean_forward_speed_mps'))} |",
        f"| speed_based_stuck_sample_ratio | {_format_value(report.stuck.get('speed_based_stuck_sample_ratio'))} |",
        f"| fall_detected | {_format_value(report.fall.get('detected'))} |",
        "",
        "## Step events",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| touchdown_count | {_format_value(steps.get('touchdown_count'))} |",
        f"| left_touchdown_count | {_format_value(steps.get('left_touchdown_count'))} |",
        f"| right_touchdown_count | {_format_value(steps.get('right_touchdown_count'))} |",
        f"| cadence_steps_per_s | {_format_value(steps.get('cadence_steps_per_s'))} |",
        f"| alternating_touchdown_ratio | {_format_value(steps.get('alternating_touchdown_ratio'))} |",
        f"| step_length_mean_m | {_format_value(steps.get('step_length_m', {}).get('mean_m'))} |",
        f"| same_foot_stride_progress_mean_m | {_format_value(steps.get('same_foot_stride_progress_m', {}).get('mean_m'))} |",
        "",
        "## Clearance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| swing_clearance_p50_m | {_format_value(clearance.get('swing', {}).get('p50_m'))} |",
        f"| swing_clearance_min_m | {_format_value(clearance.get('swing', {}).get('min_m'))} |",
        f"| low_clearance_swing_steps | {_format_value(clearance.get('low_clearance_swing_steps'))} |",
        f"| low_clearance_swing_ratio | {_format_value(clearance.get('low_clearance_swing_ratio'))} |",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Soridormi runtime JSONL stride/step metrics.")
    parser.add_argument("log", type=Path, help="Runtime JSONL log from a MuJoCo policy or skill rollout.")
    parser.add_argument("--fallback-control-hz", type=float, default=50.0)
    parser.add_argument("--min-forward-speed", type=float, default=0.02)
    parser.add_argument("--max-stuck-sample-ratio", type=float, default=0.40)
    parser.add_argument("--min-base-z", type=float, default=0.12)
    parser.add_argument("--max-abs-roll-pitch", type=float, default=0.90)
    parser.add_argument("--min-touchdown-count", type=int, default=4)
    parser.add_argument("--min-step-length", type=float, default=0.01)
    parser.add_argument("--min-swing-clearance", type=float, default=0.015)
    parser.add_argument("--max-low-clearance-ratio", type=float, default=0.35)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_stride_step_metrics(
        args.log,
        thresholds=StrideStepThresholds(
            min_forward_speed_mps=float(args.min_forward_speed),
            max_stuck_sample_ratio=float(args.max_stuck_sample_ratio),
            min_base_z_m=float(args.min_base_z),
            max_abs_roll_pitch_rad=float(args.max_abs_roll_pitch),
            min_touchdown_count=int(args.min_touchdown_count),
            min_step_length_m=float(args.min_step_length),
            min_swing_clearance_m=float(args.min_swing_clearance),
            max_low_clearance_ratio=float(args.max_low_clearance_ratio),
            contact_threshold=float(args.contact_threshold),
        ),
        fallback_control_hz=float(args.fallback_control_hz) if args.fallback_control_hz else None,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
