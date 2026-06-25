"""Execute simulator-only visual expression skills such as eye blinking."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from soridormi_api import VisualExpressionCommand

from .skill_execution import SkillExecutionError, SkillExecutionRegistry, _load_json_args


SUPPORTED_VISUAL_EXPRESSION_SKILLS = {"blink_eyes"}


@dataclass(frozen=True)
class VisualExpressionExecutionResult:
    skill_id: str
    backend: str
    executed: bool
    steps: int
    requested_duration_s: float
    visual_expressions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_robot_api_client_class() -> Any:
    from soridormi_api import RobotApiClient

    return RobotApiClient


def validate_visual_expression_plan(skill_id: str, execution: str) -> None:
    if skill_id not in SUPPORTED_VISUAL_EXPRESSION_SKILLS:
        raise SkillExecutionError(f"unsupported visual expression skill: {skill_id}")
    if execution != "visual_expression":
        raise SkillExecutionError(f"skill {skill_id} does not use visual_expression execution")


def execute_visual_expression_plan(
    skill_id: str,
    parameters: dict[str, Any] | None = None,
    *,
    backend: str = "mujoco",
    host: str = "127.0.0.1",
    port: int = 5555,
    dry_run: bool = False,
) -> tuple[dict[str, Any], VisualExpressionExecutionResult]:
    if backend != "mujoco":
        raise SkillExecutionError("visual expression skills currently require --backend mujoco")

    registry = SkillExecutionRegistry.from_manifest_path()
    plan = registry.create_plan(skill_id, parameters or {})
    validate_visual_expression_plan(plan.skill_id, plan.execution)
    visual_expressions = [expression.to_dict() for expression in plan.visual_expressions]

    if dry_run:
        return plan.to_dict(), VisualExpressionExecutionResult(
            skill_id=plan.skill_id,
            backend=backend,
            executed=False,
            steps=len(plan.visual_expressions),
            requested_duration_s=plan.total_duration_s,
            visual_expressions=visual_expressions,
        )

    client_class = _load_robot_api_client_class()
    client = client_class(host=host, port=port, timeout_ms=1000)
    try:
        for expression in plan.visual_expressions:
            client.set_visual_expression(
                VisualExpressionCommand(expression=expression.expression, intensity=expression.intensity)
            )
            time.sleep(max(0.0, float(expression.duration_s)))
        client.set_visual_expression(VisualExpressionCommand(expression="eyes_open", intensity=1.0))
    finally:
        client.close()

    return plan.to_dict(), VisualExpressionExecutionResult(
        skill_id=plan.skill_id,
        backend=backend,
        executed=True,
        steps=len(plan.visual_expressions),
        requested_duration_s=plan.total_duration_s,
        visual_expressions=visual_expressions,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a simulator-only Soridormi visual expression skill.")
    parser.add_argument("skill", help="Visual expression skill id, e.g. blink_eyes.")
    parser.add_argument("--args", default="{}", help="Skill parameter JSON object.")
    parser.add_argument("--backend", default="mujoco", help="Simulator backend selector. Only mujoco is supported.")
    parser.add_argument("--host", default="127.0.0.1", help="Simulator API host.")
    parser.add_argument("--port", type=int, default=5555, help="Simulator API port.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without connecting to MuJoCo.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan, result = execute_visual_expression_plan(
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
            print(f"Visual expression skill failed: {exc}")
        return 2

    payload = {"ok": True, "plan": plan, "result": result.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Soridormi visual expression skill")
        print("==================================")
        print(plan["summary"])
        for expression in result.visual_expressions:
            print(
                f"- {expression['label']}: {expression['expression']} "
                f"duration={expression['duration_s']:.2f}s"
            )
        if args.dry_run:
            print("No simulator command was executed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
