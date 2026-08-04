from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.clearance_readiness import DEFAULT_REQUIRED_SCENARIOS
from soridormi_runtime.scenario_curriculum import (
    DEFAULT_SCENARIO_MANIFEST,
    get_scenario_definition,
)

DEFAULT_CONTRACT_PATH = Path("configs/wbc/open_duck_mini_v2_clearance_contract.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/wbc_clearance_experiments/open_duck_mini_v2_v0")
DEFAULT_SCENARIO_EVAL_ROOT = Path("artifacts/scenario_eval")

REQUIRED_PARAMETERS = (
    "target_clearance_m",
    "startup_clearance_boost_m",
    "turn_clearance_boost_m",
    "swing_height_gain",
    "double_support_ratio",
    "step_length_scale",
    "max_lateral_swing_m",
    "posture_weight_scale",
)


@dataclass(frozen=True)
class WbcClearanceCandidatePlan:
    candidate_id: str
    profile_name: str
    description: str
    status: str
    parameter_values: dict[str, float]
    bound_margins: dict[str, dict[str, float]]
    scenario_ids: list[str]
    post_implementation_commands: dict[str, list[str]]
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WbcClearanceExperimentPlan:
    ok: bool
    status: str
    contract_path: str
    contract_version: str | None
    robot: str | None
    scope: str | None
    runtime_backend_ready: bool
    sim_only: bool
    hardware_allowed: bool
    raw_action_14d_allowed: bool
    chromie_raw_control_allowed: bool
    baseline_profile: str
    reference_profile: str
    scenario_manifest: str
    scenario_eval_root: str
    candidate_count: int
    candidates: list[dict[str, Any]]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"contract must contain a JSON object: {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    return value is True


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "candidate"


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _scenario_ids(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("scenario_ids")
    if isinstance(raw, list) and raw:
        return [str(item).strip() for item in raw if str(item).strip()]
    return list(DEFAULT_REQUIRED_SCENARIOS)


def _validate_scenarios(
    scenario_ids: Sequence[str],
    manifest_path: str | Path,
) -> list[str]:
    errors: list[str] = []
    for scenario_id in scenario_ids:
        try:
            get_scenario_definition(scenario_id, manifest_path)
        except Exception as exc:
            errors.append(f"unknown scenario {scenario_id!r}: {exc}")
    return errors


def _parameter_specs(payload: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    parameters = _mapping(payload.get("parameters"))
    errors: list[str] = []
    if not parameters:
        return {}, ["parameters must be a non-empty object"]
    for name in REQUIRED_PARAMETERS:
        if name not in parameters:
            errors.append(f"missing required WBC clearance parameter: {name}")
    for name, spec_value in parameters.items():
        spec = _mapping(spec_value)
        default = _as_float(spec.get("default"))
        minimum = _as_float(spec.get("minimum"))
        maximum = _as_float(spec.get("maximum"))
        if default is None or minimum is None or maximum is None:
            errors.append(f"{name}: default, minimum, and maximum must be numeric")
            continue
        if minimum > maximum:
            errors.append(f"{name}: minimum {minimum:g} is greater than maximum {maximum:g}")
        if not (minimum <= default <= maximum):
            errors.append(
                f"{name}: default {default:g} is outside [{minimum:g}, {maximum:g}]"
            )
    return dict(parameters), errors


def _candidate_values(
    *,
    candidate: Mapping[str, Any],
    parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[str]]:
    errors: list[str] = []
    values = _mapping(candidate.get("values"))
    full_values: dict[str, float] = {}
    margins: dict[str, dict[str, float]] = {}
    for name, raw_value in values.items():
        if name not in parameters:
            errors.append(f"{candidate.get('id', '<unnamed>')}: unknown parameter {name}")
    for name, spec in parameters.items():
        value = _as_float(values.get(name, spec.get("default")))
        minimum = _as_float(spec.get("minimum"))
        maximum = _as_float(spec.get("maximum"))
        if value is None or minimum is None or maximum is None:
            errors.append(f"{candidate.get('id', '<unnamed>')}: invalid parameter {name}")
            continue
        full_values[name] = value
        margins[name] = {
            "lower": value - minimum,
            "upper": maximum - value,
        }
        if not (minimum <= value <= maximum):
            errors.append(
                f"{candidate.get('id', '<unnamed>')}: {name}={value:g} is outside "
                f"[{minimum:g}, {maximum:g}]"
            )
    return full_values, margins, errors


def _candidate_commands(
    *,
    profile_name: str,
    scenario_eval_root: str | Path,
    reference_profile: str,
) -> dict[str, list[str]]:
    suite_dir = Path(scenario_eval_root) / profile_name
    reference_suite_dir = Path(scenario_eval_root) / reference_profile
    return {
        "validate_engineering_process": [
            "./scripts/validate_clearance_engineering_process.sh",
        ],
        "start_mujoco_no_viewer": [
            "./scripts/run_sim_server.sh",
            "--backend",
            "mujoco",
            "--profile",
            profile_name,
            "--no-viewer",
        ],
        "evaluate_scenario_suite": [
            "./scripts/evaluate_scenario_suite.sh",
            "--backend",
            "mujoco",
            "--profile",
            profile_name,
            "--output-dir",
            str(suite_dir),
            "--json",
        ],
        "analyze_clearance_readiness": [
            "./scripts/analyze_clearance_readiness.sh",
            "--profile-name",
            profile_name,
            "--suite-dir",
            str(suite_dir),
            "--reference-profile-name",
            reference_profile,
            "--reference-suite-dir",
            str(reference_suite_dir),
            "--output-dir",
            f"artifacts/clearance_readiness/{profile_name}",
            "--json",
            "--require-reference-improvement",
        ],
        "report_candidate_history": [
            "./scripts/report_clearance_candidate_history.sh",
            "--profile",
            reference_profile,
            "--profile",
            profile_name,
        ],
    }


def _candidate_plan(
    *,
    candidate: Mapping[str, Any],
    parameters: Mapping[str, Mapping[str, Any]],
    scenario_ids: Sequence[str],
    scenario_eval_root: str | Path,
    reference_profile: str,
    runtime_backend_ready: bool,
) -> tuple[WbcClearanceCandidatePlan | None, list[str]]:
    candidate_id = str(candidate.get("id") or "").strip()
    if not candidate_id:
        return None, ["candidate_sets item is missing id"]
    profile_name = str(candidate.get("profile_name") or f"wbc_clearance_{_slug(candidate_id)}")
    values, margins, errors = _candidate_values(candidate=candidate, parameters=parameters)
    blockers: list[str] = []
    status = "READY_FOR_SIM_EVALUATION"
    if not runtime_backend_ready:
        status = "WAITING_FOR_WBC_RUNTIME_BACKEND"
        blockers.append(
            "WBC runtime backend/profile generation is not implemented; keep this as "
            "a sim-only parameter plan until the backend exists."
        )
    if errors:
        status = "INVALID_PARAMETER_SET"
    return (
        WbcClearanceCandidatePlan(
            candidate_id=candidate_id,
            profile_name=profile_name,
            description=str(candidate.get("description") or ""),
            status=status,
            parameter_values=values,
            bound_margins=margins,
            scenario_ids=list(scenario_ids),
            post_implementation_commands=_candidate_commands(
                profile_name=profile_name,
                scenario_eval_root=scenario_eval_root,
                reference_profile=reference_profile,
            ),
            blockers=blockers,
        ),
        errors,
    )


def build_wbc_clearance_experiment_plan(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    scenario_eval_root: str | Path = DEFAULT_SCENARIO_EVAL_ROOT,
    baseline_profile: str | None = None,
    reference_profile: str | None = None,
) -> WbcClearanceExperimentPlan:
    resolved_contract_path = Path(contract_path)
    try:
        payload = _load_json(resolved_contract_path)
    except Exception as exc:
        return WbcClearanceExperimentPlan(
            ok=False,
            status="INVALID_CONTRACT",
            contract_path=str(resolved_contract_path),
            contract_version=None,
            robot=None,
            scope=None,
            runtime_backend_ready=False,
            sim_only=False,
            hardware_allowed=True,
            raw_action_14d_allowed=True,
            chromie_raw_control_allowed=True,
            baseline_profile=baseline_profile or "",
            reference_profile=reference_profile or "",
            scenario_manifest=str(scenario_manifest),
            scenario_eval_root=str(scenario_eval_root),
            candidate_count=0,
            candidates=[],
            blockers=[f"could not load WBC clearance contract: {exc}"],
        )

    safety = _mapping(payload.get("safety"))
    parameters, errors = _parameter_specs(payload)
    scenario_ids = _scenario_ids(payload)
    errors.extend(_validate_scenarios(scenario_ids, scenario_manifest))
    candidate_sets = payload.get("candidate_sets")
    if not isinstance(candidate_sets, list) or not candidate_sets:
        errors.append("candidate_sets must be a non-empty list")
        candidate_sets = []

    sim_only = _as_bool(safety.get("sim_only"))
    hardware_allowed = _as_bool(safety.get("hardware_allowed"))
    raw_action_allowed = _as_bool(safety.get("raw_action_14d_allowed"))
    chromie_raw_allowed = _as_bool(safety.get("chromie_raw_control_allowed"))
    if not sim_only:
        errors.append("safety.sim_only must be true for the first WBC clearance stage")
    if hardware_allowed:
        errors.append("safety.hardware_allowed must be false for the first WBC stage")
    if raw_action_allowed:
        errors.append("safety.raw_action_14d_allowed must be false")
    if chromie_raw_allowed:
        errors.append("safety.chromie_raw_control_allowed must be false")

    resolved_baseline = str(baseline_profile or payload.get("baseline_profile") or "")
    resolved_reference = str(reference_profile or payload.get("reference_profile") or "")
    runtime_backend_ready = _as_bool(payload.get("runtime_backend_ready"))
    if not resolved_baseline:
        errors.append("baseline_profile is required")
    if not resolved_reference:
        errors.append("reference_profile is required")

    candidates: list[dict[str, Any]] = []
    for candidate in candidate_sets:
        plan, candidate_errors = _candidate_plan(
            candidate=_mapping(candidate),
            parameters=parameters,
            scenario_ids=scenario_ids,
            scenario_eval_root=scenario_eval_root,
            reference_profile=resolved_reference,
            runtime_backend_ready=runtime_backend_ready,
        )
        errors.extend(candidate_errors)
        if plan is not None:
            candidates.append(plan.to_dict())

    if errors:
        status = "INVALID_CONTRACT"
    elif runtime_backend_ready:
        status = "READY_FOR_SIM_EVALUATION"
    else:
        status = "READY_FOR_WBC_BACKEND_IMPLEMENTATION"

    next_steps = [
        "Keep the contract sim-only until a WBC runtime backend can consume these parameters.",
        "Implement profile generation for WBC clearance candidates before running "
        "evaluation commands.",
        "Run each implemented candidate through the clearance qualification scenario suite and clearance "
        "readiness gate.",
        "Promote nothing without follow-camera review and teacher comparison evidence.",
    ]
    return WbcClearanceExperimentPlan(
        ok=not errors,
        status=status,
        contract_path=str(resolved_contract_path),
        contract_version=str(payload.get("contract_version") or ""),
        robot=str(payload.get("robot") or ""),
        scope=str(payload.get("scope") or ""),
        runtime_backend_ready=runtime_backend_ready,
        sim_only=sim_only,
        hardware_allowed=hardware_allowed,
        raw_action_14d_allowed=raw_action_allowed,
        chromie_raw_control_allowed=chromie_raw_allowed,
        baseline_profile=resolved_baseline,
        reference_profile=resolved_reference,
        scenario_manifest=str(scenario_manifest),
        scenario_eval_root=str(scenario_eval_root),
        candidate_count=len(candidates),
        candidates=candidates,
        blockers=sorted(set(errors)),
        warnings=[] if runtime_backend_ready else ["WBC runtime backend is not implemented yet."],
        next_steps=next_steps,
    )


def render_markdown(plan: WbcClearanceExperimentPlan) -> str:
    lines = [
        "# Soridormi WBC clearance experiment plan",
        "",
        f"Contract: `{plan.contract_path}`",
        f"Status: `{plan.status}`",
        f"Result: {'PASS' if plan.ok else 'BLOCKED'}",
        f"Robot: `{plan.robot or 'n/a'}`",
        f"Baseline profile: `{plan.baseline_profile}`",
        f"Reference profile: `{plan.reference_profile}`",
        f"Runtime backend ready: {'yes' if plan.runtime_backend_ready else 'no'}",
        f"Candidate count: {plan.candidate_count}",
        "",
        "## Safety boundary",
        "",
        f"- Sim only: {'yes' if plan.sim_only else 'no'}",
        f"- Hardware allowed: {'yes' if plan.hardware_allowed else 'no'}",
        f"- Raw action_14d allowed: {'yes' if plan.raw_action_14d_allowed else 'no'}",
        f"- Chromie raw control allowed: {'yes' if plan.chromie_raw_control_allowed else 'no'}",
        "",
        "## Candidate parameter sets",
        "",
        "| Candidate | Status | Profile | Target clearance m | Startup boost m | Turn boost m |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in plan.candidates:
        values = item.get("parameter_values", {})
        lines.append(
            "| {candidate} | {status} | {profile} | {target} | {startup} | {turn} |".format(
                candidate=item.get("candidate_id"),
                status=item.get("status"),
                profile=item.get("profile_name"),
                target=values.get("target_clearance_m"),
                startup=values.get("startup_clearance_boost_m"),
                turn=values.get("turn_clearance_boost_m"),
            )
        )
    lines.extend(["", "## Post-implementation commands", ""])
    for item in plan.candidates:
        lines.extend([f"### {item.get('candidate_id')}", ""])
        commands = item.get("post_implementation_commands", {})
        if isinstance(commands, Mapping):
            for name, command in commands.items():
                if isinstance(command, list):
                    lines.extend([f"{name}:", "", "```bash", _command_text(command), "```", ""])
        blockers = item.get("blockers", [])
        if blockers:
            lines.append("Blockers:")
            lines.extend(f"- {blocker}" for blocker in blockers if isinstance(blocker, str))
            lines.append("")
    lines.extend(["## Blockers", ""])
    lines.extend(f"- {item}" for item in plan.blockers) if plan.blockers else lines.append("- none")
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in plan.warnings)
    lines.extend(["", "## Next steps", ""])
    lines.extend(f"- {item}" for item in plan.next_steps)
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan bounded sim-only WBC clearance experiments.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--scenario-eval-root", type=Path, default=DEFAULT_SCENARIO_EVAL_ROOT)
    parser.add_argument("--baseline-profile", default=None)
    parser.add_argument("--reference-profile-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when contract is invalid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_wbc_clearance_experiment_plan(
        contract_path=args.contract,
        scenario_manifest=args.scenario_manifest,
        scenario_eval_root=args.scenario_eval_root,
        baseline_profile=args.baseline_profile,
        reference_profile=args.reference_profile_name,
    )
    output_dir = args.output_dir
    json_output = args.json_output or output_dir / "wbc_clearance_experiment_plan.json"
    markdown_output = args.output or output_dir / "wbc_clearance_experiment_plan.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(plan), encoding="utf-8")
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(plan))
    if args.strict and not plan.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
