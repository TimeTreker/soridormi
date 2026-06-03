"""Resolve a structured person target and execute ``look_at_person`` in MuJoCo."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .look_target_provider import (
    DEFAULT_HORIZONTAL_FOV_RAD,
    DEFAULT_VERTICAL_FOV_RAD,
    LookTarget,
    resolve_target_from_mapping,
)
from .scripted_head_skill import (
    DEFAULT_MAX_HEAD_VELOCITY_RADPS,
    DEFAULT_TRANSITION_FRACTION,
    _print_human,
    execute_scripted_head_plan,
)
from .skill_execution import SkillExecutionError, SkillExecutionRegistry
from .skill_manifest import DEFAULT_SKILL_MANIFEST


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a structured look-at-person target and stream the safe "
            "scripted look_at_person head trajectory. This is not person detection."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument("--target-json", help="JSON object or path with target_yaw_rad/target_pitch_rad or image_x_norm/image_y_norm.")
    parser.add_argument("--target-yaw-rad", type=float, help="Structured target yaw offset in radians.")
    parser.add_argument("--target-pitch-rad", type=float, help="Structured target pitch offset in radians.")
    parser.add_argument("--image-x-norm", type=float, help="Stub image target x in [0,1], where 0.5 is center.")
    parser.add_argument("--image-y-norm", type=float, help="Stub image target y in [0,1], where 0.5 is center and y increases downward.")
    parser.add_argument("--horizontal-fov-rad", type=float, default=DEFAULT_HORIZONTAL_FOV_RAD, help="Camera horizontal FOV for image-point stub.")
    parser.add_argument("--vertical-fov-rad", type=float, default=DEFAULT_VERTICAL_FOV_RAD, help="Camera vertical FOV for image-point stub.")
    parser.add_argument("--target-ref", default="person", help="Target reference label passed to look_at_person.")
    parser.add_argument("--confidence", type=float, default=1.0, help="Structured target confidence in [0,1].")
    parser.add_argument("--duration-s", type=float, default=4.0, help="look_at_person requested duration.")
    parser.add_argument("--hold-fraction", type=float, default=0.5, help="Fraction of duration spent holding target.")
    parser.add_argument(
        "--end-mode",
        choices=["hold_target", "return_neutral"],
        default="hold_target",
        help="Whether to keep looking at the target at the end or return to neutral. default: hold_target",
    )
    parser.add_argument("--backend", default="mujoco", choices=["mujoco"], help="Execution backend; hardware is not exposed.")
    parser.add_argument("--host", default="127.0.0.1", help="MuJoCo API host.")
    parser.add_argument("--port", type=int, default=5555, help="MuJoCo API port.")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Scripted control frequency.")
    parser.add_argument(
        "--transition-fraction",
        type=float,
        default=DEFAULT_TRANSITION_FRACTION,
        help=f"Fraction of each keyframe spent ramping before hold. default: {DEFAULT_TRANSITION_FRACTION}",
    )
    parser.add_argument(
        "--max-head-velocity-radps",
        type=float,
        default=DEFAULT_MAX_HEAD_VELOCITY_RADPS,
        help=f"Maximum planned head target speed in rad/s. Use 0 to disable. default: {DEFAULT_MAX_HEAD_VELOCITY_RADPS}",
    )
    parser.add_argument("--no-auto-stretch-duration", action="store_true", help="Do not extend too-short target motions.")
    parser.add_argument("--fall-height-m", type=float, default=0.14, help="Base-height fall threshold for live telemetry.")
    parser.add_argument("--kp", type=float, default=10.0, help="Position gain for scripted commands.")
    parser.add_argument("--kd", type=float, default=0.35, help="Velocity damping for scripted commands.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print without connecting to MuJoCo.")
    parser.add_argument("--resolve-only", action="store_true", help="Only print the resolved target and look_at_person skill args.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _target_options_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "target_json": args.target_json,
        "target_yaw_rad": args.target_yaw_rad,
        "target_pitch_rad": args.target_pitch_rad,
        "image_x_norm": args.image_x_norm,
        "image_y_norm": args.image_y_norm,
        "horizontal_fov_rad": args.horizontal_fov_rad,
        "vertical_fov_rad": args.vertical_fov_rad,
        "target_ref": args.target_ref,
        "confidence": args.confidence,
    }


def _print_target_human(target: LookTarget, skill_args: dict[str, object]) -> None:
    print("Soridormi look-at-person target provider")
    print("=========================================")
    print("This resolves a structured target only; it does not run camera perception.")
    print(f"Target source: {target.source}")
    print(f"Target ref: {target.target_ref}")
    print(f"Confidence: {target.confidence:.3f}")
    print(f"Resolved yaw: {target.yaw_rad:.3f} rad")
    print(f"Resolved pitch: {target.pitch_rad:.3f} rad")
    print("look_at_person skill args:")
    print(json.dumps(skill_args, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        target = resolve_target_from_mapping(_target_options_from_args(args))
        skill_args = target.to_skill_args(
            duration_s=args.duration_s,
            hold_fraction=args.hold_fraction,
            end_mode=args.end_mode,
        )
        registry = SkillExecutionRegistry.from_manifest_path(args.manifest)
        plan = registry.create_plan("look_at_person", skill_args)
        if args.resolve_only:
            payload = {"ok": True, "target": target.to_dict(), "skill_args": skill_args, "plan": plan.to_dict()}
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_target_human(target, skill_args)
            return 0

        result = execute_scripted_head_plan(
            plan,
            backend=args.backend,
            host=args.host,
            port=args.port,
            control_hz=args.control_hz,
            dry_run=args.dry_run,
            kp=args.kp,
            kd=args.kd,
            transition_fraction=args.transition_fraction,
            max_head_velocity_radps=args.max_head_velocity_radps,
            auto_stretch_duration=not args.no_auto_stretch_duration,
            fall_height_m=args.fall_height_m,
        )
    except (SkillExecutionError, RuntimeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Look-at-person target execution failed: {exc}")
        return 2

    payload = {"ok": True, "target": target.to_dict(), "skill_args": skill_args, "plan": plan.to_dict(), "result": result.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_target_human(target, skill_args)
        print("")
        _print_human(plan, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
