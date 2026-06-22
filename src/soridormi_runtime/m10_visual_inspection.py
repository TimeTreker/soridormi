from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.m10_clearance_readiness import (
    DEFAULT_PROFILE,
    DEFAULT_REQUIRED_SCENARIOS,
)
from soridormi_runtime.scenario_curriculum import DEFAULT_SCENARIO_MANIFEST, get_scenario_definition

DEFAULT_OUTPUT_ROOT = Path("artifacts/policy_visual_inspection")
DEFAULT_CAMERA_DISTANCE = 1.4
DEFAULT_CAMERA_AZIMUTH = 135.0
DEFAULT_CAMERA_ELEVATION = -20.0


@dataclass(frozen=True)
class VisualInspectionScenarioPlan:
    scenario_id: str
    title: str
    family: str
    priority: int
    scenario_output_dir: str
    rollout_command: list[str]
    visual_checklist: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M10VisualInspectionPlan:
    ok: bool
    profile: str
    status: str
    scenario_manifest: str
    output_dir: str
    readiness_report: str | None
    readiness_status: str | None
    require_clearance_ready: bool
    camera: dict[str, Any]
    sim_server_command: list[str]
    readiness_command: list[str]
    scenarios: list[dict[str, Any]]
    blockers: list[str] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_csv(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            cleaned = part.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _load_json(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, Mapping) else {}


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _camera_dict(distance: float, azimuth: float, elevation: float) -> dict[str, Any]:
    return {
        "follow_camera": True,
        "distance": float(distance),
        "azimuth": float(azimuth),
        "elevation": float(elevation),
    }


def _readiness_command_output_args(readiness_path: Path, *, explicit_readiness_report: bool) -> list[str]:
    if explicit_readiness_report:
        return [
            "--output-dir",
            str(readiness_path.parent),
            "--json-output",
            str(readiness_path),
        ]
    return ["--output-dir", str(readiness_path.parent)]


def _scenario_checklist(scenario_id: str) -> list[str]:
    common = [
        "Confirm both swing feet visibly clear the ground instead of scraping through stance transitions.",
        "Watch for toe drag, heel scuffing, or single-foot skating that may not appear as a fall.",
        "Confirm base height and roll/pitch remain visually stable through the rollout.",
        "Compare the visible swing height against the JSON clearance metrics after the rollout.",
    ]
    if scenario_id == "start_stop_velocity_ramp_v1":
        return [
            "Inspect stand-to-walk and walk-to-stand transitions for foot scuffing.",
            "Confirm the robot settles cleanly at the stop instead of shuffling in place.",
            *common,
        ]
    if scenario_id == "curve_turn_walk_v1":
        return [
            "Inspect inside and outside feet separately during curved walking and turning.",
            "Confirm turning does not cause the swing foot to sweep sideways through the ground plane.",
            *common,
        ]
    return common


def _scenario_plan(
    *,
    scenario_id: str,
    profile: str,
    output_dir: Path,
    manifest_path: str | Path,
    control_hz: int,
    duration_s: float | None,
    steps: int | None,
) -> VisualInspectionScenarioPlan:
    scenario = get_scenario_definition(scenario_id, manifest_path)
    scenario_output_dir = output_dir / "scenario_rollouts" / scenario_id
    command = [
        "./scripts/evaluate_scenario_rollout.sh",
        "--scenario",
        scenario_id,
        "--scenario-manifest",
        str(manifest_path),
        "--backend",
        "mujoco",
        "--profile",
        profile,
        "--control-hz",
        str(control_hz),
        "--output-dir",
        str(scenario_output_dir),
        "--json",
    ]
    if duration_s is not None:
        command.extend(["--duration-s", f"{duration_s:g}"])
    if steps is not None:
        command.extend(["--steps", str(steps)])
    return VisualInspectionScenarioPlan(
        scenario_id=scenario.id,
        title=scenario.title,
        family=scenario.family,
        priority=scenario.priority,
        scenario_output_dir=str(scenario_output_dir),
        rollout_command=command,
        visual_checklist=_scenario_checklist(scenario_id),
    )


def build_m10_visual_inspection_plan(
    *,
    profile: str = DEFAULT_PROFILE,
    scenarios: Sequence[str] = DEFAULT_REQUIRED_SCENARIOS,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    output_dir: str | Path | None = None,
    readiness_report: str | Path | None = None,
    require_clearance_ready: bool = False,
    camera_distance: float = DEFAULT_CAMERA_DISTANCE,
    camera_azimuth: float = DEFAULT_CAMERA_AZIMUTH,
    camera_elevation: float = DEFAULT_CAMERA_ELEVATION,
    control_hz: int = 50,
    duration_s: float | None = None,
    steps: int | None = None,
) -> M10VisualInspectionPlan:
    scenario_ids = list(scenarios)
    resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / profile
    explicit_readiness_report = readiness_report is not None
    readiness_path = Path(readiness_report) if explicit_readiness_report else resolved_output_dir / "clearance_readiness.json"
    blockers: list[str] = []
    notes: list[str] = []
    readiness_status: str | None = None

    if readiness_path.exists():
        readiness = _load_json(readiness_path)
        readiness_status = str(readiness.get("gate_status") or "UNKNOWN")
        if readiness.get("ok") is not True:
            blockers.append(f"clearance readiness is not passing: {readiness_status}")
    elif require_clearance_ready:
        blockers.append(f"required clearance readiness report is missing: {readiness_path}")
    else:
        notes.append(
            "No readiness report was found; this plan can still be used as an inspection checklist, "
            "but promotion evidence should include a passing clearance readiness report."
        )

    if require_clearance_ready and blockers:
        status = "BLOCKED_BY_CLEARANCE_READINESS"
    else:
        status = "READY_FOR_VISUAL_INSPECTION_PLAN" if not blockers else "PLAN_WITH_CLEARANCE_BLOCKERS"

    sim_server_command = [
        "./scripts/run_sim_server.sh",
        "--backend",
        "mujoco",
        "--profile",
        profile,
        "--viewer",
        "--follow-camera",
        "--camera-distance",
        f"{camera_distance:g}",
        "--camera-azimuth",
        f"{camera_azimuth:g}",
        "--camera-elevation",
        f"{camera_elevation:g}",
    ]
    readiness_command = [
        "./scripts/analyze_clearance_readiness.sh",
        "--profile-name",
        profile,
        *_readiness_command_output_args(
            readiness_path,
            explicit_readiness_report=explicit_readiness_report,
        ),
        "--json",
        "--strict",
    ]

    scenario_plans = [
        _scenario_plan(
            scenario_id=scenario_id,
            profile=profile,
            output_dir=resolved_output_dir,
            manifest_path=scenario_manifest,
            control_hz=control_hz,
            duration_s=duration_s,
            steps=steps,
        )
        for scenario_id in scenario_ids
    ]

    checklist = [
        "Start the MuJoCo server with the generated follow-camera command before running rollout commands.",
        "Inspect every required scenario from the follow-camera viewer, not only the aggregate JSON pass/fail result.",
        "Record PASS/FAIL/UNCLEAR for foot clearance, base stability, toe drag, and command transition quality.",
        "Keep the readiness JSON, visual inspection Markdown, and scenario rollout reports in the same evidence package.",
        "Do not promote to hardware or Chromie integration from visual evidence alone; keep promotion tied to quantitative gates.",
    ]
    return M10VisualInspectionPlan(
        ok=not blockers or not require_clearance_ready,
        profile=profile,
        status=status,
        scenario_manifest=str(scenario_manifest),
        output_dir=str(resolved_output_dir),
        readiness_report=str(readiness_path),
        readiness_status=readiness_status,
        require_clearance_ready=require_clearance_ready,
        camera=_camera_dict(camera_distance, camera_azimuth, camera_elevation),
        sim_server_command=sim_server_command,
        readiness_command=readiness_command,
        scenarios=[item.to_dict() for item in scenario_plans],
        blockers=sorted(set(blockers)),
        checklist=checklist,
        notes=notes,
    )


def render_markdown(plan: M10VisualInspectionPlan) -> str:
    lines = [
        "# Soridormi policy visual inspection plan",
        "",
        f"Profile: `{plan.profile}`",
        f"Status: `{plan.status}`",
        f"Result: {'PASS' if plan.ok else 'BLOCKED'}",
        f"Output dir: `{plan.output_dir}`",
        f"Readiness report: `{plan.readiness_report or 'n/a'}`",
        f"Readiness status: `{plan.readiness_status or 'n/a'}`",
        "",
        "## Start MuJoCo follow-camera server",
        "",
        "```bash",
        _command_text(plan.sim_server_command),
        "```",
        "",
        "## Scenario rollout commands",
        "",
    ]
    for item in plan.scenarios:
        lines.extend(
            [
                f"### {item['scenario_id']} — {item['title']}",
                "",
                "```bash",
                _command_text(item["rollout_command"]),
                "```",
                "",
                "Visual checklist:",
            ]
        )
        lines.extend(f"- {entry}" for entry in item["visual_checklist"])
        lines.append("")
    lines.extend(
        [
            "## Re-run clearance readiness after inspection rollouts",
            "",
            "```bash",
            _command_text(plan.readiness_command),
            "```",
            "",
            "## Overall checklist",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan.checklist)
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in plan.blockers) if plan.blockers else lines.append("- none")
    if plan.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {item}" for item in plan.notes)
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a repeatable follow-camera visual inspection plan.")
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE, help="Policy profile to inspect.")
    parser.add_argument("--scenario", action="append", default=[], help="Required scenario id; repeat or comma-separate.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for visual inspection artifacts.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--readiness-report", type=Path, default=None, help="Clearance readiness JSON path.")
    parser.add_argument("--require-clearance-ready", action="store_true", help="Exit nonzero unless readiness report exists and passes.")
    parser.add_argument("--camera-distance", type=float, default=DEFAULT_CAMERA_DISTANCE)
    parser.add_argument("--camera-azimuth", type=float, default=DEFAULT_CAMERA_AZIMUTH)
    parser.add_argument("--camera-elevation", type=float, default=DEFAULT_CAMERA_ELEVATION)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the plan is blocked.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenario_ids = _normalise_csv(args.scenario) or list(DEFAULT_REQUIRED_SCENARIOS)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.profile_name
    plan = build_m10_visual_inspection_plan(
        profile=args.profile_name,
        scenarios=scenario_ids,
        scenario_manifest=args.scenario_manifest,
        output_dir=output_dir,
        readiness_report=args.readiness_report,
        require_clearance_ready=args.require_clearance_ready,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        control_hz=args.control_hz,
        duration_s=args.duration_s,
        steps=args.steps,
    )
    json_output = args.json_output or output_dir / "policy_visual_inspection_plan.json"
    markdown_output = args.output or output_dir / "policy_visual_inspection_plan.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(plan), encoding="utf-8")
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(plan))
    return 1 if args.strict and not plan.ok else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
