from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soridormi_runtime.compare_official_soridormi_trace import (
    OBS_SEGMENTS,
    TraceRecord,
    load_official_trace,
    load_soridormi_trace,
)


DEFAULT_POLICY_PATH = Path("/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")


@dataclass(frozen=True)
class VectorMetric:
    count: int
    mean_mae: float | None
    max_abs_diff: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_mae": self.mean_mae,
            "max_abs_diff": self.max_abs_diff,
        }


def _array(value: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _metric_pairs(pairs: Iterable[tuple[np.ndarray | None, np.ndarray | None]]) -> VectorMetric:
    count = 0
    mae_values: list[float] = []
    max_values: list[float] = []
    for left, right in pairs:
        if left is None or right is None:
            continue
        if left.shape != right.shape:
            continue
        diff = np.abs(left - right)
        count += 1
        mae_values.append(float(diff.mean()))
        max_values.append(float(diff.max()))
    if count == 0:
        return VectorMetric(count=0, mean_mae=None, max_abs_diff=None)
    return VectorMetric(
        count=count,
        mean_mae=float(np.mean(mae_values)),
        max_abs_diff=float(np.max(max_values)),
    )


def _record_pairs(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    steps: int,
) -> list[tuple[TraceRecord, TraceRecord]]:
    # Compare by ordinal policy step. We intentionally do not match by robot_time,
    # because official policy parity.x debug is about reproducing the official policy-step loop.
    return list(zip(official[:steps], soridormi[:steps]))


def _series_metric(pairs: list[tuple[TraceRecord, TraceRecord]], attr: str) -> dict[str, Any]:
    metric = _metric_pairs((_array(getattr(a, attr)), _array(getattr(b, attr))) for a, b in pairs)
    return metric.as_dict()


def _observation_segment_metrics(pairs: list[tuple[TraceRecord, TraceRecord]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, start, end in OBS_SEGMENTS:
        metric = _metric_pairs(
            (
                None if _array(a.observation) is None else _array(a.observation)[start:end],
                None if _array(b.observation) is None else _array(b.observation)[start:end],
            )
            for a, b in pairs
        )
        output.append(
            {
                "name": name,
                "range": [start, end],
                **metric.as_dict(),
            }
        )
    return output


def _choose_providers(prefer_cuda: bool = True) -> list[str]:
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    providers: list[str] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available
    return providers


def _run_policy(
    policy_path: Path,
    observations: list[list[float]],
    *,
    input_name: str | None = None,
    output_name: str | None = None,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    """Run ONNX policy on a list of [101] observations and return [14] actions."""

    import onnxruntime as ort

    providers = _choose_providers(prefer_cuda=prefer_cuda)
    session = ort.InferenceSession(str(policy_path), providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs")
    if not outputs:
        raise RuntimeError("ONNX model has no outputs")

    input_info = next((item for item in inputs if item.name == input_name), inputs[0]) if input_name else inputs[0]
    output_info = next((item for item in outputs if item.name == output_name), outputs[0]) if output_name else outputs[0]

    actions: list[list[float]] = []
    for obs in observations:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        result = session.run([output_info.name], {input_info.name: obs_arr})[0]
        action = np.asarray(result, dtype=np.float32).reshape(-1)
        actions.append([float(x) for x in action])

    return {
        "policy_path": str(policy_path),
        "providers_requested": providers,
        "providers_active": list(getattr(session, "get_providers", lambda: providers)()),
        "input_name": input_info.name,
        "input_shape": list(input_info.shape) if hasattr(input_info, "shape") else None,
        "output_name": output_info.name,
        "output_shape": list(output_info.shape) if hasattr(output_info, "shape") else None,
        "actions": actions,
    }


def _valid_observations(records: list[TraceRecord], steps: int) -> list[list[float]]:
    observations: list[list[float]] = []
    for record in records[:steps]:
        obs = _array(record.observation)
        if obs is None:
            break
        observations.append([float(x) for x in obs])
    return observations


def _actions_metric_from_lists(left: list[list[float]], right: list[list[float]]) -> dict[str, Any]:
    n = min(len(left), len(right))
    metric = _metric_pairs((_array(left[i]), _array(right[i])) for i in range(n))
    return metric.as_dict()


def _record_actions(records: list[TraceRecord], steps: int) -> list[list[float]]:
    output: list[list[float]] = []
    for record in records[:steps]:
        action = _array(record.action)
        if action is None:
            break
        output.append([float(x) for x in action])
    return output


def _first_step_sample(record: TraceRecord | None) -> dict[str, Any]:
    if record is None:
        return {}
    obs = _array(record.observation)
    return {
        "step_index": record.step_index,
        "robot_time": record.robot_time,
        "gyro_xyz": None if obs is None else [float(x) for x in obs[0:3]],
        "accelerometer_xyz": None if obs is None else [float(x) for x in obs[3:6]],
        "command": None if obs is None else [float(x) for x in obs[6:13]],
        "feet_contacts": None if obs is None else [float(x) for x in obs[97:99]],
        "imitation_phase": None if obs is None else [float(x) for x in obs[99:101]],
        "action_first5": None if record.action is None else [float(x) for x in record.action[:5]],
    }


def build_diagnosis(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    obs = summary["metrics"]["observation"]
    action = summary["metrics"]["action"]
    if obs["count"] == 0:
        findings.append("No paired policy observations were found; enable /soridormi/policy_observation logging.")
    else:
        findings.append(
            f"Observation parity: mean_mae={obs['mean_mae']:.6f}, max_abs_diff={obs['max_abs_diff']:.6f}."
        )
    if action["count"]:
        findings.append(
            f"Logged action parity: mean_mae={action['mean_mae']:.6f}, max_abs_diff={action['max_abs_diff']:.6f}."
        )

    rerun = summary.get("onnx_rerun") or {}
    if rerun.get("enabled"):
        official_rerun = rerun.get("official_obs_vs_official_action") or {}
        soridormi_rerun = rerun.get("soridormi_obs_vs_soridormi_action") or {}
        cross = rerun.get("official_obs_action_vs_soridormi_obs_action") or {}
        if official_rerun.get("count"):
            findings.append(
                "ONNX wrapper check on official observations: "
                f"mean_mae={official_rerun['mean_mae']:.6f}, max_abs_diff={official_rerun['max_abs_diff']:.6f}."
            )
        if soridormi_rerun.get("count"):
            findings.append(
                "ONNX wrapper check on Soridormi observations: "
                f"mean_mae={soridormi_rerun['mean_mae']:.6f}, max_abs_diff={soridormi_rerun['max_abs_diff']:.6f}."
            )
        if cross.get("count"):
            findings.append(
                "Policy sensitivity to observation mismatch: "
                f"official_obs_action vs soridormi_obs_action mean_mae={cross['mean_mae']:.6f}."
            )
        if official_rerun.get("max_abs_diff") is not None and float(official_rerun["max_abs_diff"]) < 1e-4:
            findings.append("ONNX inference is parity-compatible on official observations; focus on Soridormi observation construction/timing.")
        elif official_rerun.get("count"):
            findings.append("ONNX inference on official observations differs from official actions; check provider/input/output/action dtype first.")
    elif rerun.get("error"):
        findings.append(f"ONNX re-run was skipped/failed: {rerun['error']}")
    else:
        findings.append("Run with --policy to verify ONNX wrapper parity on official observations.")

    worst = summary.get("worst_observation_segments", [])[:3]
    if worst:
        names = ", ".join(item["name"] for item in worst if item.get("mean_mae") is not None)
        if names:
            findings.append(f"Fix observation segments in this order: {names}.")
    return findings


def compare_observation_action_parity(
    official: list[TraceRecord],
    soridormi: list[TraceRecord],
    *,
    steps: int = 100,
    policy_path: str | Path | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    pairs = _record_pairs(official, soridormi, min(steps, len(official), len(soridormi)))
    segment_metrics = _observation_segment_metrics(pairs)
    worst_segments = sorted(
        segment_metrics,
        key=lambda item: -1.0 if item["mean_mae"] is None else float(item["mean_mae"]),
        reverse=True,
    )[:5]

    summary: dict[str, Any] = {
        "steps_compared": len(pairs),
        "official_records": len(official),
        "soridormi_records": len(soridormi),
        "metrics": {
            "observation": _series_metric(pairs, "observation"),
            "action": _series_metric(pairs, "action"),
            "motor_targets": _series_metric(pairs, "motor_targets"),
            "contacts": _series_metric(pairs, "contacts"),
            "phase": _series_metric(pairs, "phase"),
            "command": _series_metric(pairs, "command"),
        },
        "observation_segments": segment_metrics,
        "worst_observation_segments": worst_segments,
        "first_step": {
            "official": _first_step_sample(official[0] if official else None),
            "soridormi": _first_step_sample(soridormi[0] if soridormi else None),
        },
        "onnx_rerun": {"enabled": False},
    }

    if policy_path is not None:
        official_obs = _valid_observations(official, len(pairs))
        soridormi_obs = _valid_observations(soridormi, len(pairs))
        official_actions = _record_actions(official, len(pairs))
        soridormi_actions = _record_actions(soridormi, len(pairs))
        n = min(len(official_obs), len(soridormi_obs), len(official_actions), len(soridormi_actions))
        try:
            if n == 0:
                raise RuntimeError("no comparable observations/actions are available for ONNX re-run")
            policy = Path(policy_path)
            official_run = _run_policy(
                policy,
                official_obs[:n],
                input_name=input_name,
                output_name=output_name,
                prefer_cuda=prefer_cuda,
            )
            soridormi_run = _run_policy(
                policy,
                soridormi_obs[:n],
                input_name=input_name,
                output_name=output_name,
                prefer_cuda=prefer_cuda,
            )
            summary["onnx_rerun"] = {
                "enabled": True,
                "policy_path": str(policy),
                "input_name": official_run["input_name"],
                "output_name": official_run["output_name"],
                "providers_requested": official_run["providers_requested"],
                "providers_active": official_run["providers_active"],
                "observations_rerun": n,
                "official_obs_vs_official_action": _actions_metric_from_lists(
                    official_run["actions"], official_actions[:n]
                ),
                "soridormi_obs_vs_soridormi_action": _actions_metric_from_lists(
                    soridormi_run["actions"], soridormi_actions[:n]
                ),
                "official_obs_action_vs_soridormi_obs_action": _actions_metric_from_lists(
                    official_run["actions"], soridormi_run["actions"]
                ),
                "official_obs_action_first5": official_run["actions"][0][:5] if official_run["actions"] else None,
                "soridormi_obs_action_first5": soridormi_run["actions"][0][:5] if soridormi_run["actions"] else None,
            }
        except Exception as exc:  # pragma: no cover - onnxruntime/env dependent
            summary["onnx_rerun"] = {"enabled": False, "error": str(exc)}

    summary["diagnosis"] = build_diagnosis(summary)
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return str(value)


def print_summary(summary: dict[str, Any]) -> None:
    print("Official observation/action parity")
    print("==================================")
    print(f"Steps compared: {summary['steps_compared']}")
    print(f"Official records: {summary['official_records']}")
    print(f"Soridormi records: {summary['soridormi_records']}")
    print("Metrics:")
    for name, metric in summary["metrics"].items():
        print(
            f"  {name}: count={metric['count']} "
            f"mean_mae={_fmt(metric['mean_mae'])} max_abs_diff={_fmt(metric['max_abs_diff'])}"
        )
    print("Worst observation segments:")
    for segment in summary["worst_observation_segments"]:
        print(
            f"  {segment['name']} {segment['range']}: "
            f"mean_mae={_fmt(segment['mean_mae'])} max_abs_diff={_fmt(segment['max_abs_diff'])}"
        )
    rerun = summary.get("onnx_rerun") or {}
    print("ONNX re-run:")
    if rerun.get("enabled"):
        print(f"  policy: {rerun['policy_path']}")
        print(f"  providers: {rerun.get('providers_active')}")
        for key in (
            "official_obs_vs_official_action",
            "soridormi_obs_vs_soridormi_action",
            "official_obs_action_vs_soridormi_obs_action",
        ):
            metric = rerun[key]
            print(
                f"  {key}: count={metric['count']} "
                f"mean_mae={_fmt(metric['mean_mae'])} max_abs_diff={_fmt(metric['max_abs_diff'])}"
            )
    elif rerun.get("error"):
        print(f"  skipped/failed: {rerun['error']}")
    else:
        print("  disabled; pass --policy to verify ONNX wrapper parity")
    print("First step samples:")
    print("  official:", json.dumps(summary["first_step"]["official"], sort_keys=True))
    print("  soridormi:", json.dumps(summary["first_step"]["soridormi"], sort_keys=True))
    print("Diagnosis:")
    for item in summary["diagnosis"]:
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check official-vs-Soridormi observation/action parity.")
    parser.add_argument("--official", type=Path, required=True, help="Official trace JSONL or summary JSON")
    parser.add_argument("--soridormi", type=Path, required=True, help="Soridormi runtime .mcap or .jsonl log")
    parser.add_argument("--policy", type=Path, default=None, help="Optional ONNX policy path for re-running actions")
    parser.add_argument("--input-name", default=None, help="Optional ONNX input name, default first input")
    parser.add_argument("--output-name", default=None, help="Optional ONNX output name, default first output")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Prefer CPU provider instead of CUDA")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = compare_observation_action_parity(
        load_official_trace(args.official),
        load_soridormi_trace(args.soridormi),
        steps=max(1, int(args.steps)),
        policy_path=args.policy,
        input_name=args.input_name,
        output_name=args.output_name,
        prefer_cuda=not bool(args.cpu),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
