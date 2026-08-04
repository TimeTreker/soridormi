from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from soridormi_runtime.clearance_readiness import DEFAULT_REQUIRED_SCENARIOS
from soridormi_runtime.scenario_curriculum import (
    DEFAULT_SCENARIO_MANIFEST,
    get_scenario_definition,
)
from soridormi_runtime.scenario_suite_eval import build_scenario_suite_plan
from soridormi_runtime.wbc_clearance_contract import (
    DEFAULT_CONTRACT_PATH,
    build_wbc_clearance_experiment_plan,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/pre_wbc_scenario_surface/open_duck_mini_v2_v0")
REQUIRED_ENRICHMENT_TAGS = frozenset({"wbc_clearance_v0", "clearance"})


@dataclass(frozen=True)
class PreWbcScenarioSurfaceReport:
    ok: bool
    status: str
    scenario_manifest: str
    wbc_contract: str
    clearance_core_scenarios: list[str]
    enrichment_scenarios: list[str]
    wbc_scenario_ids: list[str]
    default_suite_scenario_ids: list[str]
    selected_scenarios: list[dict[str, Any]]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_scenario_ids(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    for candidate in candidates:
        raw = candidate.get("scenario_ids")
        if isinstance(raw, list) and raw:
            return [str(item) for item in raw]
    return []


def _scenario_row(scenario_id: str, *, manifest_path: str | Path, run_plan: Mapping[str, Any]) -> dict[str, Any]:
    scenario = get_scenario_definition(scenario_id, manifest_path)
    task_context = _mapping(scenario.task_context)
    thresholds = _mapping(scenario.acceptance_thresholds)
    return {
        "scenario_id": scenario.id,
        "title": scenario.title,
        "role": "clearance_core" if scenario.id in DEFAULT_REQUIRED_SCENARIOS else "wbc_enrichment",
        "status": scenario.status,
        "family": scenario.family,
        "primary_skill": scenario.primary_skill,
        "dataset_tags": list(scenario.dataset_tags),
        "clearance_focus": task_context.get("clearance_focus"),
        "requires_progress": bool(task_context.get("requires_progress")),
        "require_foot_metrics": bool(thresholds.get("require_foot_metrics")),
        "min_swing_clearance_m": thresholds.get("min_swing_clearance_m"),
        "run_plan": dict(run_plan),
    }


def _validate_enrichment_row(row: Mapping[str, Any]) -> list[str]:
    scenario_id = str(row.get("scenario_id"))
    blockers: list[str] = []
    tags = set(row.get("dataset_tags") or [])
    missing_tags = sorted(REQUIRED_ENRICHMENT_TAGS - tags)
    if missing_tags:
        blockers.append(f"{scenario_id}: missing enrichment dataset tags {missing_tags}")
    if row.get("clearance_focus") in (None, ""):
        blockers.append(f"{scenario_id}: task_context.clearance_focus is required")
    if row.get("require_foot_metrics") is not True:
        blockers.append(f"{scenario_id}: acceptance_thresholds.require_foot_metrics must be true")
    try:
        min_clearance = float(row.get("min_swing_clearance_m"))
    except (TypeError, ValueError):
        min_clearance = 0.0
    if min_clearance <= 0.0:
        blockers.append(f"{scenario_id}: min_swing_clearance_m must be positive")
    return blockers


def build_pre_wbc_scenario_surface_report(
    *,
    scenario_manifest: str | Path = DEFAULT_SCENARIO_MANIFEST,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    profile: str = "open_duck_forward",
    minimum_enrichment_count: int = 3,
) -> PreWbcScenarioSurfaceReport:
    contract = build_wbc_clearance_experiment_plan(
        contract_path=contract_path,
        scenario_manifest=scenario_manifest,
    )
    suite = build_scenario_suite_plan(
        manifest_path=scenario_manifest,
        profile=profile,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not contract.ok:
        blockers.extend(contract.blockers)
    if not contract.runtime_backend_ready:
        warnings.append("WBC runtime backend is not implemented yet; this gate validates readiness before tuning.")
    if contract.hardware_allowed:
        blockers.append("WBC contract must not allow hardware during pre-WBC surface validation")
    if contract.raw_action_14d_allowed:
        blockers.append("WBC contract must not allow raw action_14d control")
    if contract.chromie_raw_control_allowed:
        blockers.append("WBC contract must not allow Chromie raw control")

    wbc_scenario_ids = _candidate_scenario_ids([_mapping(item) for item in contract.candidates])
    core_scenarios = list(DEFAULT_REQUIRED_SCENARIOS)
    enrichment_scenarios = [item for item in wbc_scenario_ids if item not in core_scenarios]
    default_suite_scenario_ids = list(suite.scenario_ids)

    if not wbc_scenario_ids:
        blockers.append("WBC contract does not declare scenario_ids")
    if wbc_scenario_ids[: len(core_scenarios)] != core_scenarios:
        blockers.append(
            "WBC contract scenario_ids must start with the stable clearance qualification core: "
            f"{core_scenarios}"
        )
    missing_core = [item for item in core_scenarios if item not in wbc_scenario_ids]
    if missing_core:
        blockers.append(f"WBC contract is missing clearance qualification core scenarios: {missing_core}")
    if len(enrichment_scenarios) < minimum_enrichment_count:
        blockers.append(
            "WBC contract must include at least "
            f"{minimum_enrichment_count} pre-WBC enrichment scenarios"
        )
    if default_suite_scenario_ids != wbc_scenario_ids:
        blockers.append(
            "Default ready locomotion suite must match WBC contract scenario_ids; "
            f"default={default_suite_scenario_ids}, contract={wbc_scenario_ids}"
        )

    selected_by_id = {
        str(item.get("scenario_id")): _mapping(item)
        for item in suite.selected
        if item.get("scenario_id")
    }
    selected_rows: list[dict[str, Any]] = []
    for scenario_id in wbc_scenario_ids:
        selected = selected_by_id.get(scenario_id)
        if not selected:
            blockers.append(f"{scenario_id}: not selected by the default ready locomotion suite")
            continue
        run_plan = _mapping(selected.get("run_plan"))
        if not run_plan:
            blockers.append(f"{scenario_id}: missing default suite run plan")
            continue
        try:
            row = _scenario_row(scenario_id, manifest_path=scenario_manifest, run_plan=run_plan)
        except Exception as exc:
            blockers.append(f"{scenario_id}: could not load scenario definition: {exc}")
            continue
        if row["status"] not in {"mujoco_registry_ready", "mujoco_eval_ready", "training_ready"}:
            blockers.append(f"{scenario_id}: status {row['status']!r} is not ready")
        if row["primary_skill"] is None:
            blockers.append(f"{scenario_id}: primary skill is required")
        if row["role"] == "wbc_enrichment":
            blockers.extend(_validate_enrichment_row(row))
        selected_rows.append(row)

    status = "PRE_WBC_SCENARIO_SURFACE_READY" if not blockers else "PRE_WBC_SCENARIO_SURFACE_BLOCKED"
    next_steps = [
        "Run the six-scenario MuJoCo suite for the retained reference profile.",
        "Review startup-tail, S-turn reversal, and turn-stop-settle clearance failures before tuning.",
        "Implement WBC candidate profile generation only after this surface remains stable.",
        "Keep hardware disabled until the MuJoCo evidence gate passes.",
    ]
    return PreWbcScenarioSurfaceReport(
        ok=not blockers,
        status=status,
        scenario_manifest=str(scenario_manifest),
        wbc_contract=str(contract_path),
        clearance_core_scenarios=core_scenarios,
        enrichment_scenarios=enrichment_scenarios,
        wbc_scenario_ids=wbc_scenario_ids,
        default_suite_scenario_ids=default_suite_scenario_ids,
        selected_scenarios=selected_rows,
        blockers=sorted(set(blockers)),
        warnings=warnings,
        next_steps=next_steps,
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_markdown(report: PreWbcScenarioSurfaceReport) -> str:
    lines = [
        "# Soridormi pre-WBC scenario surface report",
        "",
        f"Status: `{report.status}`",
        f"Result: {'PASS' if report.ok else 'BLOCKED'}",
        f"Scenario manifest: `{report.scenario_manifest}`",
        f"WBC contract: `{report.wbc_contract}`",
        "",
        "## Scenario surface",
        "",
        "| Scenario | Role | Status | Skill | Clearance focus | Foot metrics required | Min swing clearance m |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for item in report.selected_scenarios:
        lines.append(
            "| {scenario} | {role} | {status} | {skill} | {focus} | {foot} | {clearance} |".format(
                scenario=item.get("scenario_id"),
                role=item.get("role"),
                status=item.get("status"),
                skill=item.get("primary_skill"),
                focus=_format_value(item.get("clearance_focus")),
                foot=_format_value(item.get("require_foot_metrics")),
                clearance=_format_value(item.get("min_swing_clearance_m")),
            )
        )
    lines.extend(["", "## clearance qualification core", ""])
    lines.extend(f"- `{item}`" for item in report.clearance_core_scenarios)
    lines.extend(["", "## Pre-WBC enrichment", ""])
    lines.extend(f"- `{item}`" for item in report.enrichment_scenarios) if report.enrichment_scenarios else lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in report.blockers) if report.blockers else lines.append("- none")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report.warnings)
    lines.extend(["", "## Next steps", ""])
    lines.extend(f"- {item}" for item in report.next_steps)
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the pre-WBC Soridormi scenario surface.")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--profile", default="open_duck_forward")
    parser.add_argument("--minimum-enrichment-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the surface is blocked.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_pre_wbc_scenario_surface_report(
        scenario_manifest=args.scenario_manifest,
        contract_path=args.contract,
        profile=args.profile,
        minimum_enrichment_count=args.minimum_enrichment_count,
    )
    output_dir = args.output_dir
    json_output = args.json_output or output_dir / "pre_wbc_scenario_surface_report.json"
    markdown_output = args.output or output_dir / "pre_wbc_scenario_surface_report.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
