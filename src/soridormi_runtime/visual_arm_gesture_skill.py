"""Execute simulator-only visual arm gesture skills."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from soridormi_api import VisualArmPoseCommand

from .skill_execution import SkillExecutionError, SkillExecutionRegistry, _load_json_args

SUPPORTED_VISUAL_ARM_GESTURE_SKILLS = {"wave_hand", "celebrate", "hug_gesture"}


@dataclass(frozen=True)
class VisualArmGestureExecutionResult:
    skill_id: str
    backend: str
    executed: bool
    steps: int
    requested_duration_s: float
    visual_arm_poses: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_robot_api_client_class() -> Any:
    from soridormi_api import RobotApiClient

    return RobotApiClient


def validate_visual_arm_gesture_plan(skill_id: str, execution: str) -> None:
    if skill_id not in SUPPORTED_VISUAL_ARM_GESTURE_SKILLS:
        raise SkillExecutionError(f"unsupported visual arm gesture skill: {skill_id}")
    if execution != "visual_arm_gesture":
        raise SkillExecutionError(f"skill {skill_id} does not use visual_arm_gesture execution")


def execute_visual_arm_gesture_plan(
    skill_id: str,
    parameters: dict[str, Any] | None = None,
    *,
    backend: str = "mujoco",
    host: str = "127.0.0.1",
    port: int = 5555,
    dry_run: bool = False,
) -> tuple[dict[str, Any], VisualArmGestureExecutionResult]:
    if backend != "mujoco":
        raise SkillExecutionError("visual arm gestures currently require --backend mujoco")

    registry = SkillExecutionRegistry.from_manifest_path()
    plan = registry.create_plan(skill_id, parameters or {})
    validate_visual_arm_gesture_plan(plan.skill_id, plan.execution)
    visual_arm_poses = [pose.to_dict() for pose in plan.visual_arm_poses]

    if dry_run:
        return plan.to_dict(), VisualArmGestureExecutionResult(
            skill_id=plan.skill_id,
            backend=backend,
            executed=False,
            steps=len(plan.visual_arm_poses),
            requested_duration_s=plan.total_duration_s,
            visual_arm_poses=visual_arm_poses,
        )

    client_class = _load_robot_api_client_class()
    client = client_class(host=host, port=port, timeout_ms=1000)
    try:
        for pose in plan.visual_arm_poses:
            client.set_visual_arm_pose(
                VisualArmPoseCommand(pose=pose.pose, side=pose.side)
            )
            time.sleep(max(0.0, float(pose.duration_s)))
    finally:
        try:
            client.set_visual_arm_pose(VisualArmPoseCommand(pose="rest", side="both"))
        finally:
            client.close()

    return plan.to_dict(), VisualArmGestureExecutionResult(
        skill_id=plan.skill_id,
        backend=backend,
        executed=True,
        steps=len(plan.visual_arm_poses),
        requested_duration_s=plan.total_duration_s,
        visual_arm_poses=visual_arm_poses,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a simulator-only Soridormi visual arm gesture skill."
    )
    parser.add_argument("skill", help="Visual arm gesture skill id, e.g. wave_hand.")
    parser.add_argument("--args", default="{}", help="Skill parameter JSON object.")
    parser.add_argument(
        "--backend", default="mujoco", help="Simulator backend selector. Only mujoco is supported."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Simulator API host.")
    parser.add_argument("--port", type=int, default=5555, help="Simulator API port.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print without connecting to MuJoCo."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan, result = execute_visual_arm_gesture_plan(
            args.skill,
            _load_json_args(args.args),
            backend=args.backend,
            host=args.host,
            port=args.port,
            dry_run=args.dry_run,
        )
    except SkillExecutionError as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Visual arm gesture failed: {exc}")
        return 2

    payload = {"ok": True, "plan": plan, "result": result.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Soridormi visual arm gesture")
        print("=============================")
        print(plan["summary"])
        for pose in result.visual_arm_poses:
            print(
                f"- {pose['label']}: {pose['pose']} side={pose['side']} "
                f"duration={pose['duration_s']:.2f}s"
            )
        if args.dry_run:
            print("No simulator command was executed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
