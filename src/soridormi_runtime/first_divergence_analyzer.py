from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from soridormi_runtime.compare_official_soridormi_trace import (
    OBS_SEGMENTS,
    TraceRecord,
    load_official_trace,
    load_soridormi_trace,
)

OBS_SEGMENT_RANGES: dict[str, tuple[int, int]] = {
    name: (start, stop) for name, start, stop in OBS_SEGMENTS
}

TOP_LEVEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("action", "action"),
    ("raw_action", "raw_action"),
    ("motor_targets", "motor_targets"),
    ("joint_positions", "joint_positions"),
    ("joint_velocities", "joint_velocities"),
    ("contacts", "contacts"),
    ("phase", "phase"),
    ("command", "command"),
    ("base_position_xyz", "base_position_xyz"),
)

HISTORY_EXPECTATIONS: tuple[tuple[str, str, int], ...] = (
    ("last_action", "action", -1),
    ("last_last_action", "action", -2),
    ("last_last_last_action", "action", -3),
    # The official loop builds the observation before writing the new motor targets.
    # If trace motor_targets are logged after inference, observation motor_targets
    # should best match the previous record's top-level motor_targets.
    ("motor_targets", "motor_targets", -1),
)


JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

OBSERVATION_CAUSAL_ORDER: dict[str, int] = {
    name: index for index, (name, _start, _stop) in enumerate(OBS_SEGMENTS)
}

# Within a single policy step, prefer upstream causes over large downstream
# effects. The runtime reads state, builds observation, runs ONNX, then maps the
# action to motor targets. A smaller state/observation mismatch at the same step
# is therefore more useful than a larger action mismatch.
TOP_LEVEL_CAUSAL_ORDER: dict[str, int] = {
    "joint_positions": 0,
    "joint_velocities": 1,
    "contacts": 2,
    "base_position_xyz": 3,
    "command": 4,
    "phase": 5,
    "raw_action": 200,
    "action": 210,
    "motor_targets": 220,
}


@dataclass(frozen=True)
class VectorDiff:
    name: str
    step: int
    official_step: int
    soridormi_step: int
    mae: float
    max_abs_diff: float
    threshold: float
    official: list[float]
    soridormi: list[float]
    kind: str = "field"
    observation_range: tuple[int, int] | None = None

    @property
    def exceeds_threshold(self) -> bool:
        return self.max_abs_diff > self.threshold

    def as_dict(self, *, preview: int = 6) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "step": self.step,
            "official_step": self.official_step,
            "soridormi_step": self.soridormi_step,
            "mae": self.mae,
            "max_abs_diff": self.max_abs_diff,
            "threshold": self.threshold,
            "exceeds_threshold": self.exceeds_threshold,
            "official_preview": self.official[:preview],
            "soridormi_preview": self.soridormi[:preview],
        }
        if self.observation_range is not None:
            out["observation_range"] = list(self.observation_range)
        return out


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _safe_mean(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _vector_diff(
    name: str,
    official: np.ndarray,
    soridormi: np.ndarray,
    *,
    step: int,
    official_step: int,
    soridormi_step: int,
    threshold: float,
    kind: str,
    observation_range: tuple[int, int] | None = None,
) -> VectorDiff | None:
    n = min(int(official.size), int(soridormi.size))
    if n <= 0:
        return None
    left = official[:n]
    right = soridormi[:n]
    diff = np.abs(left - right)
    return VectorDiff(
        name=name,
        step=int(step),
        official_step=int(official_step),
        soridormi_step=int(soridormi_step),
        mae=float(diff.mean()),
        max_abs_diff=float(diff.max()),
        threshold=float(threshold),
        official=[float(x) for x in left.tolist()],
        soridormi=[float(x) for x in right.tolist()],
        kind=kind,
        observation_range=observation_range,
    )


def _observation_segment(record: TraceRecord, segment_name: str) -> np.ndarray | None:
    observation = _array(record.observation)
    if observation is None:
        return None
    span = OBS_SEGMENT_RANGES.get(segment_name)
    if span is None:
        return None
    start, stop = span
    if observation.size < stop:
        return None
    return observation[start:stop]


def _record_vector(record: TraceRecord, attr: str) -> np.ndarray | None:
    return _array(getattr(record, attr, None))


def _paired_by_ordinal(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    steps: int,
) -> list[tuple[int, TraceRecord, TraceRecord]]:
    count = min(max(0, int(steps)), len(official), len(soridormi))
    return [(i, official[i], soridormi[i]) for i in range(count)]


def _diffs_for_pair(
    ordinal_step: int,
    official: TraceRecord,
    soridormi: TraceRecord,
    *,
    threshold: float,
) -> list[VectorDiff]:
    diffs: list[VectorDiff] = []

    for name, start, stop in OBS_SEGMENTS:
        left_obs = _array(official.observation)
        right_obs = _array(soridormi.observation)
        if left_obs is None or right_obs is None:
            continue
        if left_obs.size < stop or right_obs.size < stop:
            continue
        item = _vector_diff(
            name,
            left_obs[start:stop],
            right_obs[start:stop],
            step=ordinal_step,
            official_step=official.step_index,
            soridormi_step=soridormi.step_index,
            threshold=threshold,
            kind="observation_segment",
            observation_range=(start, stop),
        )
        if item is not None:
            diffs.append(item)

    for name, attr in TOP_LEVEL_FIELDS:
        left = _record_vector(official, attr)
        right = _record_vector(soridormi, attr)
        if left is None or right is None:
            continue
        item = _vector_diff(
            name,
            left,
            right,
            step=ordinal_step,
            official_step=official.step_index,
            soridormi_step=soridormi.step_index,
            threshold=threshold,
            kind="top_level_field",
        )
        if item is not None:
            diffs.append(item)

    return diffs


def _mean_mae_for_aligned_vectors(
    records_left: list[TraceRecord],
    records_right: list[TraceRecord],
    extractor_left: Callable[[TraceRecord], np.ndarray | None],
    extractor_right: Callable[[TraceRecord], np.ndarray | None],
    *,
    right_offset: int,
    steps: int,
) -> dict[str, Any]:
    values: list[float] = []
    maxes: list[float] = []
    samples = 0
    for i in range(min(steps, len(records_left))):
        j = i + int(right_offset)
        if j < 0 or j >= len(records_right):
            continue
        left = extractor_left(records_left[i])
        right = extractor_right(records_right[j])
        if left is None or right is None:
            continue
        n = min(int(left.size), int(right.size))
        if n <= 0:
            continue
        diff = np.abs(left[:n] - right[:n])
        values.append(float(diff.mean()))
        maxes.append(float(diff.max()))
        samples += 1
    return {
        "samples": samples,
        "mean_mae": _safe_mean(values),
        "max_abs_diff": max(maxes) if maxes else None,
    }


def _best_offset(metrics: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        (label, item.get("mean_mae"))
        for label, item in metrics.items()
        if item.get("mean_mae") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item[1]))[0]


def _shift_diagnostics(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    *,
    steps: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offsets = {
        "same_step": 0,
        "soridormi_plus_1": 1,
        "soridormi_minus_1": -1,
    }

    for name, _start, _stop in OBS_SEGMENTS:
        metrics = {
            label: _mean_mae_for_aligned_vectors(
                official,
                soridormi,
                lambda record, segment=name: _observation_segment(record, segment),
                lambda record, segment=name: _observation_segment(record, segment),
                right_offset=offset,
                steps=steps,
            )
            for label, offset in offsets.items()
        }
        out.append({"name": name, "kind": "observation_segment", "best_alignment": _best_offset(metrics), **metrics})

    for name, attr in TOP_LEVEL_FIELDS:
        metrics = {
            label: _mean_mae_for_aligned_vectors(
                official,
                soridormi,
                lambda record, field=attr: _record_vector(record, field),
                lambda record, field=attr: _record_vector(record, field),
                right_offset=offset,
                steps=steps,
            )
            for label, offset in offsets.items()
        }
        out.append({"name": name, "kind": "top_level_field", "best_alignment": _best_offset(metrics), **metrics})

    return out


def _history_offset_metrics(
    records: list[TraceRecord],
    *,
    obs_segment: str,
    source_attr: str,
    offsets: Iterable[int],
    steps: int,
) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    for offset in offsets:
        values: list[float] = []
        maxes: list[float] = []
        samples = 0
        for i in range(min(steps, len(records))):
            j = i + int(offset)
            if j < 0 or j >= len(records):
                continue
            obs_vec = _observation_segment(records[i], obs_segment)
            src_vec = _record_vector(records[j], source_attr)
            if obs_vec is None or src_vec is None:
                continue
            n = min(int(obs_vec.size), int(src_vec.size))
            if n <= 0:
                continue
            diff = np.abs(obs_vec[:n] - src_vec[:n])
            values.append(float(diff.mean()))
            maxes.append(float(diff.max()))
            samples += 1
        metrics[int(offset)] = {
            "samples": samples,
            "mean_mae": _safe_mean(values),
            "max_abs_diff": max(maxes) if maxes else None,
        }
    return metrics


def _history_diagnostics_for_trace(
    records: list[TraceRecord],
    *,
    label: str,
    steps: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offsets = (-3, -2, -1, 0, 1)
    for obs_segment, source_attr, expected_offset in HISTORY_EXPECTATIONS:
        metrics_by_offset = _history_offset_metrics(
            records,
            obs_segment=obs_segment,
            source_attr=source_attr,
            offsets=offsets,
            steps=steps,
        )
        usable = [
            (offset, metric["mean_mae"])
            for offset, metric in metrics_by_offset.items()
            if metric.get("mean_mae") is not None and int(metric.get("samples", 0)) > 0
        ]
        best_offset = None if not usable else int(min(usable, key=lambda item: float(item[1]))[0])
        out.append(
            {
                "trace": label,
                "observation_segment": obs_segment,
                "source_field": source_attr,
                "expected_post_inference_log_offset": expected_offset,
                "best_offset": best_offset,
                "offset_metrics": {str(k): v for k, v in sorted(metrics_by_offset.items())},
            }
        )
    return out


def _history_diagnostics(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    *,
    steps: int,
) -> list[dict[str, Any]]:
    return [
        *_history_diagnostics_for_trace(official, label="official", steps=steps),
        *_history_diagnostics_for_trace(soridormi, label="soridormi", steps=steps),
    ]


def _causal_rank(item: VectorDiff) -> tuple[int, int, float]:
    if item.kind == "top_level_field":
        return (0, TOP_LEVEL_CAUSAL_ORDER.get(item.name, 100), -item.max_abs_diff)
    if item.kind == "observation_segment":
        return (1, OBSERVATION_CAUSAL_ORDER.get(item.name, 100), -item.max_abs_diff)
    return (9, 100, -item.max_abs_diff)


def _top_element_diffs(
    name: str,
    official: np.ndarray,
    soridormi: np.ndarray,
    *,
    labels: tuple[str, ...] | None = None,
    preview: int = 6,
) -> list[dict[str, Any]]:
    n = min(int(official.size), int(soridormi.size))
    if n <= 0:
        return []
    diff = np.abs(official[:n] - soridormi[:n])
    order = np.argsort(-diff)[: max(1, int(preview))]
    out: list[dict[str, Any]] = []
    for index in order.tolist():
        label = None if labels is None or index >= len(labels) else labels[index]
        out.append(
            {
                "index": int(index),
                "label": label,
                "official": float(official[index]),
                "soridormi": float(soridormi[index]),
                "diff": float(diff[index]),
            }
        )
    return out


def _max_diff_dict(
    name: str,
    official: np.ndarray | None,
    soridormi: np.ndarray | None,
    *,
    labels: tuple[str, ...] | None = None,
    preview: int = 6,
) -> dict[str, Any]:
    if official is None or soridormi is None:
        return {"name": name, "available": False}
    n = min(int(official.size), int(soridormi.size))
    if n <= 0:
        return {"name": name, "available": False}
    diff = np.abs(official[:n] - soridormi[:n])
    return {
        "name": name,
        "available": True,
        "mae": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "top_diffs": _top_element_diffs(
            name, official[:n], soridormi[:n], labels=labels, preview=preview
        ),
    }


def _state_consistency_for_pair(
    official: TraceRecord,
    soridormi: TraceRecord,
    *,
    preview: int,
    threshold: float,
) -> dict[str, Any]:
    official_joint_offsets = _observation_segment(official, "joint_offsets")
    soridormi_joint_offsets = _observation_segment(soridormi, "joint_offsets")
    official_joint_positions = _record_vector(official, "joint_positions")
    soridormi_joint_positions = _record_vector(soridormi, "joint_positions")
    official_joint_velocities = _record_vector(official, "joint_velocities")
    soridormi_joint_velocities = _record_vector(soridormi, "joint_velocities")

    out: dict[str, Any] = {
        "step_index": int(official.step_index),
        "joint_offsets": _max_diff_dict(
            "joint_offsets",
            official_joint_offsets,
            soridormi_joint_offsets,
            labels=JOINT_NAMES,
            preview=preview,
        ),
        "joint_positions": _max_diff_dict(
            "joint_positions",
            official_joint_positions,
            soridormi_joint_positions,
            labels=JOINT_NAMES,
            preview=preview,
        ),
        "joint_velocities": _max_diff_dict(
            "joint_velocities",
            official_joint_velocities,
            soridormi_joint_velocities,
            labels=JOINT_NAMES,
            preview=preview,
        ),
    }

    if (
        official_joint_offsets is not None
        and soridormi_joint_offsets is not None
        and official_joint_positions is not None
        and soridormi_joint_positions is not None
    ):
        n = min(
            int(official_joint_offsets.size),
            int(soridormi_joint_offsets.size),
            int(official_joint_positions.size),
            int(soridormi_joint_positions.size),
        )
        official_implied_default = official_joint_positions[:n] - official_joint_offsets[:n]
        soridormi_implied_default = soridormi_joint_positions[:n] - soridormi_joint_offsets[:n]
        out["implied_default_actuator"] = _max_diff_dict(
            "implied_default_actuator",
            official_implied_default,
            soridormi_implied_default,
            labels=JOINT_NAMES,
            preview=preview,
        )

        joint_pos_max = float(out["joint_positions"].get("max_abs_diff") or 0.0)
        implied_default_max = float(out["implied_default_actuator"].get("max_abs_diff") or 0.0)
        if implied_default_max > threshold and joint_pos_max <= threshold:
            out["joint_offset_interpretation"] = (
                "joint_offsets differ while joint_positions match; inspect policy default actuator pose/bootstrap."
            )
        elif joint_pos_max > threshold:
            out["joint_offset_interpretation"] = (
                "joint_positions differ before inference; inspect simulator reset/step timing or previous command application."
            )
        else:
            out["joint_offset_interpretation"] = (
                "joint_offsets are not explained by available joint_positions; inspect log topic timing."
            )
    else:
        out["implied_default_actuator"] = {"name": "implied_default_actuator", "available": False}
        out["joint_offset_interpretation"] = (
            "joint_positions were not available in both traces, so joint_offsets cannot yet be decomposed into state versus default-pose mismatch."
        )

    return out


def _first_divergence(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    *,
    steps: int,
    threshold: float,
) -> tuple[VectorDiff | None, list[VectorDiff]]:
    for ordinal_step, official_record, soridormi_record in _paired_by_ordinal(official, soridormi, steps):
        diffs = _diffs_for_pair(
            ordinal_step,
            official_record,
            soridormi_record,
            threshold=threshold,
        )
        exceeding = [item for item in diffs if item.exceeds_threshold]
        if exceeding:
            # Within the first divergent control step, report the earliest causal
            # field, not necessarily the largest downstream effect. For example,
            # a small joint_offsets mismatch is more actionable than the action
            # mismatch it causes after ONNX inference.
            first = min(exceeding, key=_causal_rank)
            return first, sorted(exceeding, key=_causal_rank)
    return None, []


def _diagnose_shift(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    first = summary.get("first_divergence")
    if not first:
        findings.append("No compared vector exceeded the configured threshold.")
        return findings

    name = str(first["name"])
    step = int(first["step"])
    findings.append(
        f"First threshold crossing is at ordinal step {step} in {name} "
        f"(max_abs_diff={first['max_abs_diff']:.6g})."
    )

    shift_items = {
        item["name"]: item
        for item in summary.get("shift_diagnostics", [])
        if item.get("name") == name
    }
    shift = shift_items.get(name)
    if shift is not None:
        same = shift.get("same_step", {}).get("mean_mae")
        plus = shift.get("soridormi_plus_1", {}).get("mean_mae")
        minus = shift.get("soridormi_minus_1", {}).get("mean_mae")
        best = shift.get("best_alignment")
        if best in {"soridormi_plus_1", "soridormi_minus_1"}:
            findings.append(
                f"{name} aligns better with {best} than same_step "
                f"(same={_fmt(same)}, +1={_fmt(plus)}, -1={_fmt(minus)}), "
                "which is a strong pre-step/post-step logging-order signal."
            )

    history = summary.get("history_diagnostics", [])
    relevant_history = [
        item for item in history if item.get("observation_segment") == name
    ]
    for item in relevant_history:
        best_offset = item.get("best_offset")
        expected = item.get("expected_post_inference_log_offset")
        if best_offset is not None and expected is not None and int(best_offset) != int(expected):
            findings.append(
                f"{item['trace']} observation {name} best matches {item['source_field']} "
                f"at offset {best_offset}, not expected offset {expected}. "
                "Inspect when that history is updated or when that field is logged."
            )

    if name in {"last_action", "last_last_action", "last_last_last_action"}:
        findings.append(
            "Action-history divergence usually means the observation builder shifted history before/after inference differently from the official loop."
        )
    elif name == "motor_targets":
        findings.append(
            "Motor-target divergence usually means the observation motor_targets field is using the current command when the official loop still exposes the previous target, or vice versa."
        )
    elif name in {"gyro_xyz", "accelerometer_xyz", "feet_contacts", "contacts"}:
        findings.append(
            "Sensor/contact divergence before action-history divergence points to state-read timing: compare pre-step versus post-step simulator state, not policy math first."
        )
    elif name in {"command", "phase", "imitation_phase"}:
        findings.append(
            "Command/phase divergence is upstream of ONNX inference; fix generation/log order before checking actions."
        )
    elif name in {"action", "raw_action"}:
        findings.append(
            "Action divergence with matching observations points to ONNX/postprocessor differences; action divergence after observation divergence is likely downstream."
        )

    return findings


def analyze_first_divergence(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    *,
    steps: int = 100,
    threshold: float = 1e-4,
    preview: int = 6,
    top: int = 8,
) -> dict[str, Any]:
    compared_steps = min(max(0, int(steps)), len(official), len(soridormi))
    first, candidates = _first_divergence(
        official,
        soridormi,
        steps=compared_steps,
        threshold=float(threshold),
    )
    largest_at_first_step = None if not candidates else max(candidates, key=lambda item: item.max_abs_diff)
    first_pair_state = None
    if first is not None:
        ordinal = int(first.step)
        if 0 <= ordinal < compared_steps:
            first_pair_state = _state_consistency_for_pair(
                official[ordinal],
                soridormi[ordinal],
                preview=preview,
                threshold=float(threshold),
            )

    summary: dict[str, Any] = {
        "steps_compared": compared_steps,
        "official_records": len(official),
        "soridormi_records": len(soridormi),
        "threshold": float(threshold),
        "first_divergence": None if first is None else first.as_dict(preview=preview),
        "largest_divergence_at_first_step": (
            None if largest_at_first_step is None else largest_at_first_step.as_dict(preview=preview)
        ),
        "candidates_at_first_divergent_step": [
            item.as_dict(preview=preview) for item in candidates[: max(1, int(top))]
        ],
        "state_consistency_at_first_step": first_pair_state,
        "history_diagnostics": _history_diagnostics(
            official,
            soridormi,
            steps=compared_steps,
        ),
        "shift_diagnostics": _shift_diagnostics(
            official,
            soridormi,
            steps=compared_steps,
        ),
    }
    summary["diagnosis"] = _diagnose_shift(summary)
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def print_first_divergence(summary: dict[str, Any]) -> None:
    print("Official vs Soridormi first-divergence analyzer")
    print("================================================")
    print(f"Steps compared: {summary['steps_compared']}")
    print(f"Official records: {summary['official_records']}")
    print(f"Soridormi records: {summary['soridormi_records']}")
    print(f"Threshold: {summary['threshold']}")
    first = summary.get("first_divergence")
    print("First divergence:")
    if first is None:
        print("  none")
    else:
        print(f"  step: {first['step']}")
        print(f"  official_step: {first['official_step']}")
        print(f"  soridormi_step: {first['soridormi_step']}")
        print(f"  name: {first['name']}")
        print(f"  kind: {first['kind']}")
        if "observation_range" in first:
            print(f"  observation_range: {first['observation_range']}")
        print(f"  mae: {_fmt(first['mae'])}")
        print(f"  max_abs_diff: {_fmt(first['max_abs_diff'])}")
        print(f"  official_preview: {first['official_preview']}")
        print(f"  soridormi_preview: {first['soridormi_preview']}")

    largest = summary.get("largest_divergence_at_first_step")
    if largest is not None and first is not None and largest.get("name") != first.get("name"):
        print("Largest downstream divergence at that same step:")
        print(
            "  "
            f"{largest['name']} ({largest['kind']}): "
            f"mae={_fmt(largest['mae'])} max_abs_diff={_fmt(largest['max_abs_diff'])}"
        )

    print("Candidates at first divergent step, causal order:")
    for item in summary.get("candidates_at_first_divergent_step", []):
        print(
            "  "
            f"{item['name']} ({item['kind']}): "
            f"mae={_fmt(item['mae'])} max_abs_diff={_fmt(item['max_abs_diff'])}"
        )

    state = summary.get("state_consistency_at_first_step")
    if state is not None:
        print("State/observation consistency at first step:")
        print(f"  {state.get('joint_offset_interpretation')}")
        for key in ("joint_offsets", "joint_positions", "joint_velocities", "implied_default_actuator"):
            item = state.get(key, {})
            if not item.get("available"):
                print(f"  {key}: unavailable")
                continue
            print(
                "  "
                f"{key}: mae={_fmt(item.get('mae'))} "
                f"max_abs_diff={_fmt(item.get('max_abs_diff'))}"
            )
            for diff in item.get("top_diffs", [])[:3]:
                label = diff.get("label") or f"index_{diff.get('index')}"
                print(
                    "    "
                    f"{label}: official={_fmt(diff.get('official'))} "
                    f"soridormi={_fmt(diff.get('soridormi'))} "
                    f"diff={_fmt(diff.get('diff'))}"
                )

    print("History/order checks:")
    for item in summary.get("history_diagnostics", []):
        print(
            "  "
            f"{item['trace']} obs[{item['observation_segment']}] vs "
            f"{item['source_field']}: best_offset={item['best_offset']} "
            f"expected={item['expected_post_inference_log_offset']}"
        )

    print("One-step shift checks for first divergent field:")
    first_name = None if first is None else first.get("name")
    if first_name is None:
        print("  n/a")
    else:
        for item in summary.get("shift_diagnostics", []):
            if item.get("name") != first_name:
                continue
            print(
                "  "
                f"{first_name}: best={item.get('best_alignment')} "
                f"same={_fmt(item.get('same_step', {}).get('mean_mae'))} "
                f"soridormi+1={_fmt(item.get('soridormi_plus_1', {}).get('mean_mae'))} "
                f"soridormi-1={_fmt(item.get('soridormi_minus_1', {}).get('mean_mae'))}"
            )
            break

    print("Diagnosis:")
    for item in summary.get("diagnosis", []):
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the first official-vs-Soridormi closed-loop trace divergence."
    )
    parser.add_argument("--official", type=Path, required=True, help="Official trace JSONL or summary JSON")
    parser.add_argument("--soridormi", type=Path, required=True, help="Soridormi runtime .mcap or .jsonl log")
    parser.add_argument("--steps", type=int, default=100, help="Number of initial policy steps to compare")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Per-vector max_abs_diff threshold for first divergence",
    )
    parser.add_argument("--preview", type=int, default=6, help="Number of values to show per vector preview")
    parser.add_argument("--top", type=int, default=8, help="Number of candidates to show at first divergent step")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = analyze_first_divergence(
        load_official_trace(args.official),
        load_soridormi_trace(args.soridormi),
        steps=max(1, int(args.steps)),
        threshold=float(args.threshold),
        preview=max(1, int(args.preview)),
        top=max(1, int(args.top)),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_first_divergence(summary)


if __name__ == "__main__":
    main()
