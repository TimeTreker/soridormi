from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from soridormi_api import MotorCommand
from soridormi_sim.mujoco_backend import MujocoBackend


DEFAULT_TRACE = Path("/data/official_baseline/latest_official_baseline.trace.jsonl")
DEFAULT_OUTPUT_DIR = Path("/data/official_baseline")


@dataclass(frozen=True)
class ReplayConfig:
    trace_path: Path = DEFAULT_TRACE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    summary_prefix: str = "official_target_replay"
    max_steps: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    xs = [float(x) for x in values]
    if not xs:
        return {"count": 0, "min": None, "max": None, "mean": None, "abs_max": None}
    return {
        "count": len(xs),
        "min": min(xs),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
        "abs_max": max(abs(x) for x in xs),
    }


def _vector_delta(records: list[dict[str, Any]]) -> dict[str, Any]:
    starts = [r.get("base_position_xyz") for r in records if isinstance(r.get("base_position_xyz"), list)]
    if len(starts) < 2:
        return {"available": False}
    start = [float(x) for x in starts[0][:3]]
    end = [float(x) for x in starts[-1][:3]]
    delta = [end[i] - start[i] for i in range(3)]
    return {
        "available": True,
        "start_xyz": start,
        "end_xyz": end,
        "delta_xyz": delta,
        "forward_x": delta[0],
    }


def _state_record(
    *,
    source: str,
    official: dict[str, Any],
    backend: MujocoBackend,
    command_positions: list[float],
    step_index: int,
) -> dict[str, Any]:
    state = backend.get_state()
    return {
        "source": source,
        "step_index": int(step_index),
        "policy_step": int(step_index + 1),
        "robot_time": float(state.time),
        "sim_time": float(state.time),
        "official_step_index": int(official.get("step_index", step_index)),
        "motor_targets": [float(x) for x in command_positions],
        "joint_positions": [float(x) for x in state.joints.positions],
        "joint_velocities": [float(x) for x in state.joints.velocities],
        "contacts": None if state.feet_contacts is None else [float(x) for x in state.feet_contacts],
        "base_position_xyz": None
        if state.base_position_xyz is None
        else [float(x) for x in state.base_position_xyz],
        "base_quat_wxyz": None
        if state.base_quat_wxyz is None
        else [float(x) for x in state.base_quat_wxyz],
        "actuator_ctrl": None if state.actuator_ctrl is None else [float(x) for x in state.actuator_ctrl],
    }


def replay(config: ReplayConfig) -> dict[str, Any]:
    if not config.trace_path.exists():
        raise FileNotFoundError(f"Official trace not found: {config.trace_path}")

    official_records = _read_jsonl(config.trace_path)
    if config.max_steps > 0:
        official_records = official_records[: config.max_steps]
    if not official_records:
        raise RuntimeError(f"No records found in official trace: {config.trace_path}")

    backend = MujocoBackend()
    replay_records: list[dict[str, Any]] = []
    motor_values: list[float] = []
    contact_values: list[float] = []

    try:
        # Official MjInfer computes its first policy action after 10 MuJoCo
        # physics steps from the home keyframe. Prime Soridormi's backend the
        # same way before applying the first official motor target.
        backend.step()

        for step_index, official in enumerate(official_records):
            targets = official.get("motor_targets")
            if not isinstance(targets, list) or len(targets) != len(backend.actuator_names):
                raise ValueError(
                    f"Official trace record {step_index} has invalid motor_targets: {targets!r}"
                )
            command_positions = [float(x) for x in targets]

            record = _state_record(
                source="soridormi_official_target_replay",
                official=official,
                backend=backend,
                command_positions=command_positions,
                step_index=step_index,
            )
            replay_records.append(record)
            motor_values.extend(command_positions)
            if isinstance(record.get("contacts"), list):
                contact_values.extend(float(x) for x in record["contacts"])

            backend.apply_command(
                MotorCommand(
                    names=list(backend.actuator_names),
                    positions=command_positions,
                    velocities=[0.0] * len(command_positions),
                    kp=[10.0] * len(command_positions),
                    kd=[0.5] * len(command_positions),
                    torques=[0.0] * len(command_positions),
                )
            )
            backend.step()
    finally:
        backend.close()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_path = config.output_dir / f"{config.summary_prefix}_{timestamp}.trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for record in replay_records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    latest_trace = config.output_dir / "latest_official_target_replay.trace.jsonl"
    latest_trace.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "kind": "soridormi_official_target_replay",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "official_trace": str(config.trace_path),
        "trace_jsonl": str(trace_path),
        "latest_trace_jsonl": str(latest_trace),
        "records": len(replay_records),
        "motor_target_stats": _stats(motor_values),
        "contact_stats": _stats(contact_values),
        "displacement": _vector_delta(replay_records),
    }
    summary_path = config.output_dir / f"{config.summary_prefix}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_summary = config.output_dir / "latest_official_target_replay.json"
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay official Open Duck motor targets through Soridormi's MuJoCo backend.",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-prefix", default="official_target_replay")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = replay(
        ReplayConfig(
            trace_path=args.trace,
            output_dir=args.output_dir,
            summary_prefix=args.summary_prefix,
            max_steps=args.max_steps,
        )
    )
    print("Soridormi official-target replay finished")
    print("=========================================")
    print(f"Official trace: {summary['official_trace']}")
    print(f"Trace: {summary['trace_jsonl']}")
    print(f"Summary: {summary['latest_trace_jsonl']}")
    print(f"Records: {summary['records']}")
    print(f"Displacement: {summary['displacement']}")
    print(f"Motor target stats: {summary['motor_target_stats']}")
    print(f"Contact stats: {summary['contact_stats']}")


if __name__ == "__main__":
    main()
