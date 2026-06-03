"""Acceptance checks for safe scripted social head/neck skills.

The evaluator is intentionally small and sim-first. By default it uses the
scripted-head dry-run path so CI can validate the planned trajectories without
connecting to MuJoCo. With ``--execute`` it streams the same skills to an
already-running MuJoCo simulator and additionally checks observed head ranges
and base-height fall telemetry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .scripted_head_skill import HEAD_JOINT_NAMES, execute_scripted_head_plan
from .skill_execution import SkillExecutionError, SkillExecutionRegistry
from .skill_manifest import DEFAULT_SKILL_MANIFEST


@dataclass(frozen=True)
class ScriptedSocialAcceptanceCase:
    skill_id: str
    args: dict[str, Any]
    required_axis: str | None = None
    required_min: float | None = None
    required_max: float | None = None
    max_command_abs_by_axis: dict[str, float] | None = None
    min_observed_range: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ACCEPTANCE_CASES: tuple[ScriptedSocialAcceptanceCase, ...] = (
    ScriptedSocialAcceptanceCase(
        skill_id="neutral_head",
        args={"duration_s": 3.0},
        max_command_abs_by_axis={
            "neck_pitch": 1e-9,
            "head_pitch": 1e-9,
            "head_yaw": 1e-9,
            "head_roll": 1e-9,
        },
    ),
    ScriptedSocialAcceptanceCase(
        skill_id="look_direction",
        args={"head_yaw_rad": 0.25, "head_pitch_rad": -0.08, "duration_s": 1.6},
        required_axis="head_yaw",
        required_min=None,
        required_max=0.22,
        min_observed_range=0.12,
    ),
    ScriptedSocialAcceptanceCase(
        skill_id="nod_yes",
        args={"count": 2, "amplitude": "small", "duration_s": 4.0},
        required_axis="head_pitch",
        required_min=-0.16,
        required_max=0.10,
        max_command_abs_by_axis={"neck_pitch": 1e-9, "head_yaw": 1e-9, "head_roll": 1e-9},
        min_observed_range=0.18,
    ),
    ScriptedSocialAcceptanceCase(
        skill_id="shake_no",
        args={"count": 2, "amplitude": "small", "duration_s": 4.0},
        required_axis="head_yaw",
        required_min=-0.25,
        required_max=0.25,
        max_command_abs_by_axis={"neck_pitch": 1e-9, "head_pitch": 1e-9, "head_roll": 1e-9},
        min_observed_range=0.30,
    ),
    ScriptedSocialAcceptanceCase(
        skill_id="bow",
        args={"depth": "small", "duration_s": 5.0},
        required_axis="head_pitch",
        required_min=-0.16,
        required_max=None,
        max_command_abs_by_axis={"head_yaw": 1e-9, "head_roll": 1e-9},
        min_observed_range=0.12,
    ),
    ScriptedSocialAcceptanceCase(
        skill_id="express_attention",
        args={"style": "curious", "duration_s": 4.0},
        required_axis="head_yaw",
        required_min=None,
        required_max=0.12,
        max_command_abs_by_axis={"neck_pitch": 1e-9, "head_roll": 1e-9},
        min_observed_range=0.08,
    ),
)


@dataclass(frozen=True)
class ScriptedSocialAcceptanceResult:
    skill_id: str
    ok: bool
    executed: bool
    errors: list[str]
    warnings: list[str]
    args: dict[str, Any]
    commanded_ranges: dict[str, dict[str, float]]
    observed_ranges: dict[str, dict[str, float]]
    requested_duration_s: float
    effective_duration_s: float
    steps: int
    fallen: bool | None = None
    min_base_height_m: float | None = None
    final_base_height_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptedSocialAcceptanceSummary:
    ok: bool
    executed: bool
    results: list[ScriptedSocialAcceptanceResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "executed": self.executed,
            "results": [result.to_dict() for result in self.results],
        }


def _range_payload(mins: Mapping[str, float], maxs: Mapping[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": float(mins.get(name, 0.0)),
            "max": float(maxs.get(name, 0.0)),
            "range": float(maxs.get(name, 0.0)) - float(mins.get(name, 0.0)),
        }
        for name in HEAD_JOINT_NAMES
    }


def _axis_range(ranges: Mapping[str, Mapping[str, float]], axis: str) -> tuple[float, float, float]:
    row = ranges.get(axis, {})
    return float(row.get("min", 0.0)), float(row.get("max", 0.0)), float(row.get("range", 0.0))


def evaluate_acceptance_case(
    case: ScriptedSocialAcceptanceCase,
    registry: SkillExecutionRegistry,
    *,
    execute: bool = False,
    backend: str = "mujoco",
    host: str = "127.0.0.1",
    port: int = 5555,
    control_hz: float = 50.0,
    kp: float = 10.0,
    kd: float = 0.35,
    transition_fraction: float = 0.40,
    max_head_velocity_radps: float | None = 0.35,
    auto_stretch_duration: bool = True,
    fall_height_m: float = 0.14,
    require_observed: bool = False,
) -> ScriptedSocialAcceptanceResult:
    plan = registry.create_plan(case.skill_id, case.args)
    result = execute_scripted_head_plan(
        plan,
        backend=backend,
        host=host,
        port=port,
        control_hz=control_hz,
        dry_run=not execute,
        kp=kp,
        kd=kd,
        transition_fraction=transition_fraction,
        max_head_velocity_radps=max_head_velocity_radps,
        auto_stretch_duration=auto_stretch_duration,
        fall_height_m=fall_height_m,
    )

    errors: list[str] = []
    warnings: list[str] = []
    commanded_ranges = _range_payload(result.target_min_positions_by_name, result.target_max_positions_by_name)
    observed_ranges = _range_payload(result.observed_min_positions_by_name, result.observed_max_positions_by_name)

    if result.executed != execute:
        errors.append(f"execution mode mismatch: expected executed={execute}, got {result.executed}")

    if case.required_axis is not None:
        lo, hi, _ = _axis_range(commanded_ranges, case.required_axis)
        if case.required_min is not None and lo > float(case.required_min):
            errors.append(
                f"{case.skill_id} commanded {case.required_axis} min {lo:.3f} > required {case.required_min:.3f}"
            )
        if case.required_max is not None and hi < float(case.required_max):
            errors.append(
                f"{case.skill_id} commanded {case.required_axis} max {hi:.3f} < required {case.required_max:.3f}"
            )

    for axis, max_abs in (case.max_command_abs_by_axis or {}).items():
        lo, hi, _ = _axis_range(commanded_ranges, axis)
        if max(abs(lo), abs(hi)) > float(max_abs):
            errors.append(
                f"{case.skill_id} commanded non-moving axis {axis} outside tolerance: min={lo:.3g}, max={hi:.3g}"
            )

    if execute:
        if result.fallen:
            errors.append(
                f"{case.skill_id} base height dropped below fall threshold {fall_height_m:.3f}m "
                f"(min={result.observed_min_base_height_m})"
            )
        if case.required_axis and case.min_observed_range is not None:
            _, _, observed_range = _axis_range(observed_ranges, case.required_axis)
            if observed_range < float(case.min_observed_range):
                message = (
                    f"{case.skill_id} observed {case.required_axis} range {observed_range:.3f} "
                    f"< required {case.min_observed_range:.3f}"
                )
                if require_observed:
                    errors.append(message)
                else:
                    warnings.append(message)
    elif require_observed:
        errors.append("--require-observed was set but --execute was not enabled")

    return ScriptedSocialAcceptanceResult(
        skill_id=case.skill_id,
        ok=not errors,
        executed=result.executed,
        errors=errors,
        warnings=warnings,
        args=dict(case.args),
        commanded_ranges=commanded_ranges,
        observed_ranges=observed_ranges,
        requested_duration_s=float(result.requested_duration_s),
        effective_duration_s=float(result.effective_duration_s),
        steps=int(result.steps),
        fallen=result.fallen,
        min_base_height_m=result.observed_min_base_height_m,
        final_base_height_m=result.final_base_height_m,
    )


def run_acceptance(
    *,
    manifest: str = str(DEFAULT_SKILL_MANIFEST),
    skill_ids: Sequence[str] | None = None,
    execute: bool = False,
    backend: str = "mujoco",
    host: str = "127.0.0.1",
    port: int = 5555,
    control_hz: float = 50.0,
    transition_fraction: float = 0.40,
    max_head_velocity_radps: float | None = 0.35,
    auto_stretch_duration: bool = True,
    fall_height_m: float = 0.14,
    require_observed: bool = False,
) -> ScriptedSocialAcceptanceSummary:
    registry = SkillExecutionRegistry.from_manifest_path(manifest)
    wanted = set(skill_ids or [])
    cases = [case for case in DEFAULT_ACCEPTANCE_CASES if not wanted or case.skill_id in wanted]
    if wanted and len(cases) != len(wanted):
        known = {case.skill_id for case in DEFAULT_ACCEPTANCE_CASES}
        missing = sorted(wanted - known)
        raise SkillExecutionError(f"unknown scripted social acceptance skill(s): {missing}")
    results = [
        evaluate_acceptance_case(
            case,
            registry,
            execute=execute,
            backend=backend,
            host=host,
            port=port,
            control_hz=control_hz,
            transition_fraction=transition_fraction,
            max_head_velocity_radps=max_head_velocity_radps,
            auto_stretch_duration=auto_stretch_duration,
            fall_height_m=fall_height_m,
            require_observed=require_observed,
        )
        for case in cases
    ]
    return ScriptedSocialAcceptanceSummary(ok=all(result.ok for result in results), executed=execute, results=results)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate scripted social skill acceptance gates.")
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument(
        "--skill",
        action="append",
        dest="skills",
        help="Skill id to evaluate; repeatable. Defaults to neutral_head, look_direction, nod_yes, shake_no, bow, and express_attention.",
    )
    parser.add_argument("--execute", action="store_true", help="Run against an already-running MuJoCo simulator.")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco"], help="Execution backend.")
    parser.add_argument("--host", default="127.0.0.1", help="MuJoCo API host.")
    parser.add_argument("--port", type=int, default=5555, help="MuJoCo API port.")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Scripted control frequency.")
    parser.add_argument("--transition-fraction", type=float, default=0.40, help="Trajectory ramp fraction per keyframe.")
    parser.add_argument(
        "--max-head-velocity-radps",
        type=float,
        default=0.35,
        help="Planned head target speed limit in rad/s; use 0 to disable.",
    )
    parser.add_argument(
        "--no-auto-stretch-duration",
        action="store_true",
        help="Do not stretch gestures to satisfy the speed limit.",
    )
    parser.add_argument("--fall-height-m", type=float, default=0.14, help="Base-height fall threshold for live eval.")
    parser.add_argument(
        "--require-observed",
        action="store_true",
        help="Fail live acceptance if observed joint range is below the threshold.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _print_human(summary: ScriptedSocialAcceptanceSummary) -> None:
    print("Soridormi scripted social acceptance")
    print("====================================")
    print(f"Mode: {'live MuJoCo execution' if summary.executed else 'dry-run trajectory'}")
    print(f"Overall: {'PASS' if summary.ok else 'FAIL'}")
    for result in summary.results:
        print("")
        print(f"{result.skill_id}: {'PASS' if result.ok else 'FAIL'}")
        print(f"  requested_duration_s: {result.requested_duration_s:.2f}")
        print(f"  effective_duration_s: {result.effective_duration_s:.2f}")
        print(f"  steps: {result.steps}")
        if result.executed:
            print(f"  fallen: {result.fallen}")
            if result.min_base_height_m is not None:
                print(f"  min_base_height_m: {result.min_base_height_m:.3f}")
            if result.final_base_height_m is not None:
                print(f"  final_base_height_m: {result.final_base_height_m:.3f}")
        print("  commanded head ranges:")
        for axis in HEAD_JOINT_NAMES:
            row = result.commanded_ranges.get(axis, {})
            print(f"    - {axis}: min={row.get('min', 0.0):.3f}, max={row.get('max', 0.0):.3f}")
        if result.executed:
            print("  observed head ranges:")
            for axis in HEAD_JOINT_NAMES:
                row = result.observed_ranges.get(axis, {})
                print(f"    - {axis}: min={row.get('min', 0.0):.3f}, max={row.get('max', 0.0):.3f}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        for error in result.errors:
            print(f"  error: {error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_acceptance(
            manifest=args.manifest,
            skill_ids=args.skills,
            execute=args.execute,
            backend=args.backend,
            host=args.host,
            port=args.port,
            control_hz=args.control_hz,
            transition_fraction=args.transition_fraction,
            max_head_velocity_radps=args.max_head_velocity_radps,
            auto_stretch_duration=not args.no_auto_stretch_duration,
            fall_height_m=args.fall_height_m,
            require_observed=args.require_observed,
        )
    except (SkillExecutionError, RuntimeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Scripted social acceptance failed: {exc}")
        return 2

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0 if summary.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
