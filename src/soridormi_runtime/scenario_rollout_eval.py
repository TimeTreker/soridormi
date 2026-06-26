from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from soridormi_runtime.scenario_curriculum import (
    DEFAULT_SCENARIO_MANIFEST,
    ScenarioCurriculumError,
    ScenarioDefinition,
    get_scenario_definition,
)
from soridormi_runtime.skill_execution import MIN_FORWARD_WALK_SPEED_MPS, apply_min_forward_walk_speed
from soridormi_runtime.stride_step_metrics_eval import (
    StrideStepReport,
    StrideStepThresholds,
    evaluate_stride_step_metrics,
)


DEFAULT_PROFILE = "open_duck_forward"


@dataclass(frozen=True)
class ScenarioRolloutThresholds:
    """Acceptance thresholds for one MuJoCo scenario rollout report.

    M9B treats the scenario manifest as the default source of truth and keeps
    the CLI flags as explicit overrides for local experiments.
    """

    min_distance_m: float = 0.05
    min_mean_forward_speed_mps: float = 0.02
    max_stuck_sample_ratio: float = 0.40
    require_not_fallen: bool = True
    min_touchdown_count: int = 4
    min_swing_clearance_m: float = 0.015
    max_low_clearance_ratio: float = 0.35
    require_foot_metrics: bool = False
    min_base_z_m: float = 0.12
    max_abs_roll_pitch_rad: float = 0.90
    contact_threshold: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bool_from_manifest(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _threshold_number(raw: Mapping[str, Any], key: str, default: float | int) -> float:
    value = _as_float(raw.get(key), default=float(default))
    return float(default if value is None else value)


def thresholds_from_scenario_manifest(
    scenario: ScenarioDefinition,
    *,
    fallback: ScenarioRolloutThresholds | None = None,
) -> ScenarioRolloutThresholds:
    """Resolve scenario-specific rollout thresholds from manifest metadata.

    The normalized M9B field is ``acceptance_thresholds``.  Older M8
    ``success_metrics`` fields remain a compatibility fallback so historical
    manifests can still be evaluated.
    """

    base = fallback or ScenarioRolloutThresholds()
    raw = scenario.acceptance_thresholds
    success_metrics = scenario.payload.get("success_metrics", {})
    if not raw and isinstance(success_metrics, Mapping):
        raw = {
            "min_distance_m": success_metrics.get("minimum_forward_progress_m", base.min_distance_m),
            "max_stuck_sample_ratio": success_metrics.get("maximum_stuck_ratio", base.max_stuck_sample_ratio),
            "require_not_fallen": not bool(success_metrics.get("fall_allowed", not base.require_not_fallen)),
        }

    if not isinstance(raw, Mapping) or not raw:
        return base

    return ScenarioRolloutThresholds(
        min_distance_m=_threshold_number(raw, "min_distance_m", base.min_distance_m),
        min_mean_forward_speed_mps=_threshold_number(
            raw, "min_mean_forward_speed_mps", base.min_mean_forward_speed_mps
        ),
        max_stuck_sample_ratio=_threshold_number(raw, "max_stuck_sample_ratio", base.max_stuck_sample_ratio),
        require_not_fallen=_bool_from_manifest(raw.get("require_not_fallen"), default=base.require_not_fallen),
        min_touchdown_count=int(_threshold_number(raw, "min_touchdown_count", base.min_touchdown_count)),
        min_swing_clearance_m=_threshold_number(raw, "min_swing_clearance_m", base.min_swing_clearance_m),
        max_low_clearance_ratio=_threshold_number(raw, "max_low_clearance_ratio", base.max_low_clearance_ratio),
        require_foot_metrics=_bool_from_manifest(raw.get("require_foot_metrics"), default=base.require_foot_metrics),
        min_base_z_m=_threshold_number(raw, "min_base_z_m", base.min_base_z_m),
        max_abs_roll_pitch_rad=_threshold_number(raw, "max_abs_roll_pitch_rad", base.max_abs_roll_pitch_rad),
        contact_threshold=_threshold_number(raw, "contact_threshold", base.contact_threshold),
    )


def overlay_threshold_overrides(
    thresholds: ScenarioRolloutThresholds,
    *,
    min_distance_m: float | None = None,
    min_mean_forward_speed_mps: float | None = None,
    max_stuck_sample_ratio: float | None = None,
    require_not_fallen: bool | None = None,
    min_touchdown_count: int | None = None,
    min_swing_clearance_m: float | None = None,
    max_low_clearance_ratio: float | None = None,
    require_foot_metrics: bool | None = None,
    min_base_z_m: float | None = None,
    max_abs_roll_pitch_rad: float | None = None,
    contact_threshold: float | None = None,
) -> ScenarioRolloutThresholds:
    """Return thresholds with only explicitly provided CLI overrides applied."""

    return ScenarioRolloutThresholds(
        min_distance_m=thresholds.min_distance_m if min_distance_m is None else float(min_distance_m),
        min_mean_forward_speed_mps=(
            thresholds.min_mean_forward_speed_mps
            if min_mean_forward_speed_mps is None
            else float(min_mean_forward_speed_mps)
        ),
        max_stuck_sample_ratio=(
            thresholds.max_stuck_sample_ratio if max_stuck_sample_ratio is None else float(max_stuck_sample_ratio)
        ),
        require_not_fallen=thresholds.require_not_fallen if require_not_fallen is None else bool(require_not_fallen),
        min_touchdown_count=thresholds.min_touchdown_count if min_touchdown_count is None else int(min_touchdown_count),
        min_swing_clearance_m=(
            thresholds.min_swing_clearance_m if min_swing_clearance_m is None else float(min_swing_clearance_m)
        ),
        max_low_clearance_ratio=(
            thresholds.max_low_clearance_ratio if max_low_clearance_ratio is None else float(max_low_clearance_ratio)
        ),
        require_foot_metrics=(
            thresholds.require_foot_metrics if require_foot_metrics is None else bool(require_foot_metrics)
        ),
        min_base_z_m=thresholds.min_base_z_m if min_base_z_m is None else float(min_base_z_m),
        max_abs_roll_pitch_rad=(
            thresholds.max_abs_roll_pitch_rad if max_abs_roll_pitch_rad is None else float(max_abs_roll_pitch_rad)
        ),
        contact_threshold=thresholds.contact_threshold if contact_threshold is None else float(contact_threshold),
    )


@dataclass(frozen=True)
class ScenarioCheck:
    name: str
    ok: bool
    value: Any
    threshold: Any
    severity: str = "error"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioRunPlan:
    scenario_id: str
    skill_id: str
    profile: str
    args: dict[str, Any]
    duration_s: float
    steps: int
    control_hz: float
    log_prefix: str
    log_dir: str
    scenario_status: str
    task_context: dict[str, Any]
    environment_context: dict[str, Any]
    command_space: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioRolloutReport:
    ok: bool
    scenario_id: str
    scenario_title: str
    scenario_status: str
    scenario_family: str
    expected_skill_id: str | None
    log_path: str
    sample_count: int
    duration_s: float | None
    task_context: dict[str, Any]
    environment_context: dict[str, Any]
    command_space: dict[str, Any]
    acceptance_thresholds: dict[str, Any]
    threshold_source: str
    metrics: dict[str, Any]
    checks: list[dict[str, Any]]
    stride_step_report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _scenario_range_midpoint(scenario: ScenarioDefinition, field_name: str, fallback: float) -> float:
    try:
        minimum, maximum = scenario.command_range(field_name)
    except ScenarioCurriculumError:
        return float(fallback)
    return float((minimum + maximum) / 2.0)


def _scenario_positive_nominal(scenario: ScenarioDefinition, field_name: str, fallback: float) -> float:
    try:
        minimum, maximum = scenario.command_range(field_name)
    except ScenarioCurriculumError:
        return float(fallback)
    if maximum <= 0.0:
        return float(maximum)
    low = max(0.0, minimum)
    nominal = float((low + maximum) / 2.0)
    applied, _ = apply_min_forward_walk_speed(nominal, max_vx_mps=maximum)
    return applied


def _scenario_turn_nominal(scenario: ScenarioDefinition, field_name: str, fallback: float) -> float:
    try:
        minimum, maximum = scenario.command_range(field_name)
    except ScenarioCurriculumError:
        return float(fallback)
    magnitude = max(abs(minimum), abs(maximum))
    if magnitude <= 0.0:
        return 0.0
    # Pick a deterministic visible turn direction without always choosing the
    # manifest midpoint, which is often zero for symmetric ranges.
    sign = 1.0 if maximum >= abs(minimum) else -1.0
    return float(sign * min(magnitude, max(0.04, magnitude * 0.6)))


def _duration_from_scenario(scenario: ScenarioDefinition, requested_duration_s: float | None) -> float:
    if requested_duration_s is not None:
        if requested_duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        return float(requested_duration_s)
    raw = scenario.command_space.get("duration_s")
    if isinstance(raw, list) and len(raw) == 2:
        minimum = _as_float(raw[0])
        maximum = _as_float(raw[1])
        if minimum is not None and maximum is not None and maximum >= minimum and maximum > 0.0:
            return float((minimum + maximum) / 2.0)
    return 4.0


def build_scenario_run_plan(
    scenario_id: str,
    *,
    manifest_path: str | Path = DEFAULT_SCENARIO_MANIFEST,
    profile: str = DEFAULT_PROFILE,
    duration_s: float | None = None,
    steps: int | None = None,
    control_hz: float = 50.0,
    log_prefix: str | None = None,
    log_dir: str = "/data/logs",
) -> ScenarioRunPlan:
    scenario = get_scenario_definition(scenario_id, manifest_path)
    skill_id = scenario.primary_skill
    if skill_id is None:
        raise ScenarioCurriculumError(f"scenario {scenario.id!r} has no primary skill")

    planned_duration_s = _duration_from_scenario(scenario, duration_s)
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    planned_steps = int(steps) if steps is not None else max(1, int(math.ceil(planned_duration_s * control_hz)))
    if planned_steps <= 0:
        raise ValueError("steps must be positive")

    args: dict[str, Any] = {"duration_s": planned_duration_s}
    if skill_id == "walk_velocity":
        args.update(
            {
                "vx_mps": _scenario_positive_nominal(scenario, "vx_mps", 0.08),
                "vy_mps": _scenario_range_midpoint(scenario, "vy_mps", 0.0),
                "yaw_radps": 0.0,
            }
        )
    elif skill_id in {"stop", "stand", "stand_idle"}:
        args.update({"vx_mps": 0.0, "vy_mps": 0.0, "yaw_radps": 0.0})
    elif skill_id == "curve_walk":
        args.update(
            {
                "vx_mps": _scenario_positive_nominal(scenario, "vx_mps", 0.08),
                "yaw_radps": _scenario_turn_nominal(scenario, "yaw_radps", 0.10),
            }
        )
    elif skill_id == "turn_in_place":
        args.update({"yaw_radps": _scenario_turn_nominal(scenario, "yaw_radps", 0.12)})
    else:
        raise ScenarioCurriculumError(
            f"scenario {scenario.id!r} primary skill {skill_id!r} is not a locomotion policy skill "
            "for M9A scenario rollout evaluation"
        )

    return ScenarioRunPlan(
        scenario_id=scenario.id,
        skill_id=skill_id,
        profile=profile,
        args=args,
        duration_s=planned_duration_s,
        steps=planned_steps,
        control_hz=float(control_hz),
        log_prefix=log_prefix or f"scenario_{scenario.id}",
        log_dir=log_dir,
        scenario_status=scenario.status,
        task_context=scenario.task_context,
        environment_context=scenario.environment_context,
        command_space=scenario.command_space,
    )


def _metric(report: StrideStepReport, path: tuple[str, ...]) -> Any:
    value: Any = report.to_dict()
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _check_at_least(name: str, value: Any, threshold: float | int, *, severity: str = "error") -> ScenarioCheck:
    parsed = _as_float(value)
    ok = parsed is not None and parsed >= float(threshold)
    return ScenarioCheck(
        name=name,
        ok=ok,
        value=value,
        threshold=threshold,
        severity=severity,
        message=(f"{name}: {value} >= {threshold}" if ok else f"{name}: {value} < {threshold}"),
    )


def _check_at_most(name: str, value: Any, threshold: float | int, *, severity: str = "error") -> ScenarioCheck:
    parsed = _as_float(value)
    ok = parsed is not None and parsed <= float(threshold)
    return ScenarioCheck(
        name=name,
        ok=ok,
        value=value,
        threshold=threshold,
        severity=severity,
        message=(f"{name}: {value} <= {threshold}" if ok else f"{name}: {value} > {threshold}"),
    )


def _check_bool(name: str, value: Any, expected: bool, *, severity: str = "error") -> ScenarioCheck:
    ok = bool(value) is expected
    return ScenarioCheck(
        name=name,
        ok=ok,
        value=bool(value),
        threshold=expected,
        severity=severity,
        message=(f"{name}: {bool(value)} == {expected}" if ok else f"{name}: {bool(value)} != {expected}"),
    )


def evaluate_scenario_rollout(
    log_path: str | Path,
    *,
    scenario_id: str,
    manifest_path: str | Path = DEFAULT_SCENARIO_MANIFEST,
    thresholds: ScenarioRolloutThresholds | None = None,
    fallback_control_hz: float | None = 50.0,
) -> ScenarioRolloutReport:
    scenario = get_scenario_definition(scenario_id, manifest_path)
    if thresholds is None:
        cfg = thresholds_from_scenario_manifest(scenario)
        threshold_source = "scenario_manifest" if scenario.acceptance_thresholds else "default_fallback"
    else:
        cfg = thresholds
        threshold_source = "explicit"
    stride_thresholds = StrideStepThresholds(
        min_forward_speed_mps=cfg.min_mean_forward_speed_mps,
        max_stuck_sample_ratio=cfg.max_stuck_sample_ratio,
        min_base_z_m=cfg.min_base_z_m,
        max_abs_roll_pitch_rad=cfg.max_abs_roll_pitch_rad,
        min_touchdown_count=cfg.min_touchdown_count,
        min_swing_clearance_m=cfg.min_swing_clearance_m,
        max_low_clearance_ratio=cfg.max_low_clearance_ratio,
        contact_threshold=cfg.contact_threshold,
    )
    stride = evaluate_stride_step_metrics(log_path, thresholds=stride_thresholds, fallback_control_hz=fallback_control_hz)

    checks: list[ScenarioCheck] = []
    warnings: list[str] = list(stride.warnings)
    errors: list[str] = list(stride.errors)
    scenario_ids = set(stride.scenario.get("scenario_ids", []))
    skill_ids = set(stride.scenario.get("skill_ids", []))
    expected_skill = scenario.primary_skill

    if scenario_ids and scenario.id not in scenario_ids:
        checks.append(
            ScenarioCheck(
                name="scenario_id_matches_log",
                ok=False,
                value=sorted(scenario_ids),
                threshold=scenario.id,
                message=f"log scenario_ids {sorted(scenario_ids)} do not include expected {scenario.id}",
            )
        )
    elif scenario_ids:
        checks.append(
            ScenarioCheck(
                name="scenario_id_matches_log",
                ok=True,
                value=sorted(scenario_ids),
                threshold=scenario.id,
                severity="info",
                message=f"log includes expected scenario {scenario.id}",
            )
        )
    else:
        warnings.append("log does not carry scenario_id metadata; evaluated against requested scenario only")

    if expected_skill and skill_ids and expected_skill not in skill_ids:
        checks.append(
            ScenarioCheck(
                name="skill_id_matches_scenario",
                ok=False,
                value=sorted(skill_ids),
                threshold=expected_skill,
                message=f"log skill_ids {sorted(skill_ids)} do not include expected {expected_skill}",
            )
        )
    elif expected_skill and skill_ids:
        checks.append(
            ScenarioCheck(
                name="skill_id_matches_scenario",
                ok=True,
                value=sorted(skill_ids),
                threshold=expected_skill,
                severity="info",
                message=f"log includes expected skill {expected_skill}",
            )
        )

    requires_progress = bool(scenario.task_context.get("requires_progress", False))
    if requires_progress:
        checks.append(_check_at_least("forward_distance_m", stride.base_motion.get("forward_x_m"), cfg.min_distance_m))
        checks.append(
            _check_at_least(
                "mean_forward_speed_mps",
                stride.base_motion.get("mean_forward_speed_mps"),
                cfg.min_mean_forward_speed_mps,
            )
        )
    else:
        checks.append(
            ScenarioCheck(
                name="progress_required",
                ok=True,
                value=False,
                threshold=False,
                severity="info",
                message="scenario does not require forward progress",
            )
        )

    checks.append(
        _check_at_most(
            "stuck_ratio",
            stride.stuck.get("speed_based_stuck_sample_ratio"),
            cfg.max_stuck_sample_ratio,
        )
    )
    if cfg.require_not_fallen:
        checks.append(_check_bool("not_fallen", stride.fall.get("detected"), False))

    foot_metric_severity = "error" if cfg.require_foot_metrics else "warning"
    if stride.samples_with_feet > 0:
        checks.append(
            ScenarioCheck(
                name="foot_metrics_present",
                ok=True,
                value=stride.samples_with_feet,
                threshold=">0",
                severity="info",
                message=f"log includes {stride.samples_with_feet} feet_position samples",
            )
        )
        checks.append(
            _check_at_least(
                "touchdown_count",
                stride.step_events.get("touchdown_count"),
                cfg.min_touchdown_count,
                severity=foot_metric_severity,
            )
        )
        low_clearance_ratio = stride.foot_clearance.get("low_clearance_swing_ratio")
        if low_clearance_ratio is not None or cfg.require_foot_metrics:
            checks.append(
                _check_at_most(
                    "low_clearance_swing_ratio",
                    low_clearance_ratio,
                    cfg.max_low_clearance_ratio,
                    severity=foot_metric_severity,
                )
            )
        swing_p50 = stride.foot_clearance.get("swing", {}).get("p50_m")
        if swing_p50 is not None or cfg.require_foot_metrics:
            checks.append(
                _check_at_least(
                    "swing_clearance_p50_m",
                    swing_p50,
                    cfg.min_swing_clearance_m,
                    severity=foot_metric_severity,
                )
            )
    elif cfg.require_foot_metrics:
        checks.append(
            ScenarioCheck(
                name="foot_metrics_present",
                ok=False,
                value=0,
                threshold=">0",
                message="scenario rollout requires foot metrics but the log has no feet_position samples",
            )
        )
    else:
        warnings.append("foot metrics unavailable; clearance/touchdown checks are warnings-only for this log")

    for check in checks:
        if not check.ok and check.severity == "error":
            errors.append(check.message)
        elif not check.ok:
            warnings.append(check.message)

    metrics = {
        "forward_distance_m": stride.base_motion.get("forward_x_m"),
        "horizontal_distance_m": stride.base_motion.get("horizontal_distance_m"),
        "mean_forward_speed_mps": stride.base_motion.get("mean_forward_speed_mps"),
        "stuck_ratio": stride.stuck.get("speed_based_stuck_sample_ratio"),
        "fallen": stride.fall.get("detected"),
        "min_base_z_m": stride.fall.get("min_base_z_m"),
        "samples_with_feet": stride.samples_with_feet,
        "touchdown_count": stride.step_events.get("touchdown_count"),
        "cadence_steps_per_s": stride.step_events.get("cadence_steps_per_s"),
        "step_length_mean_m": stride.step_events.get("step_length_m", {}).get("mean_m"),
        "swing_clearance_p05_m": stride.foot_clearance.get("swing", {}).get("p05_m"),
        "swing_clearance_p50_m": stride.foot_clearance.get("swing", {}).get("p50_m"),
        "low_clearance_swing_ratio": stride.foot_clearance.get("low_clearance_swing_ratio"),
    }

    return ScenarioRolloutReport(
        ok=not errors,
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        scenario_status=scenario.status,
        scenario_family=scenario.family,
        expected_skill_id=expected_skill,
        log_path=str(log_path),
        sample_count=stride.sample_count,
        duration_s=stride.duration_s,
        task_context=scenario.task_context,
        environment_context=scenario.environment_context,
        command_space=scenario.command_space,
        acceptance_thresholds=cfg.as_dict(),
        threshold_source=threshold_source,
        metrics=metrics,
        checks=[check.to_dict() for check in checks],
        stride_step_report=stride.to_dict(),
        warnings=warnings,
        errors=errors,
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def render_markdown(report: ScenarioRolloutReport) -> str:
    lines = [
        "# Soridormi scenario rollout report",
        "",
        f"Result: {'PASS' if report.ok else 'FAILED'}",
        f"Scenario: `{report.scenario_id}` — {report.scenario_title}",
        f"Status: {report.scenario_status}",
        f"Family: {report.scenario_family}",
        f"Expected skill: {report.expected_skill_id or 'n/a'}",
        f"Threshold source: {report.threshold_source}",
        f"Log: {report.log_path}",
        f"Samples: {report.sample_count}",
        f"Duration: {_format_value(report.duration_s)} s",
        "",
        "## Key metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "forward_distance_m",
        "horizontal_distance_m",
        "mean_forward_speed_mps",
        "stuck_ratio",
        "fallen",
        "min_base_z_m",
        "samples_with_feet",
        "touchdown_count",
        "cadence_steps_per_s",
        "step_length_mean_m",
        "swing_clearance_p05_m",
        "swing_clearance_p50_m",
        "low_clearance_swing_ratio",
    ):
        lines.append(f"| {key} | {_format_value(report.metrics.get(key))} |")

    lines.extend(["", "## Acceptance checks", "", "| Check | Result | Value | Threshold | Severity |", "| --- | --- | ---: | ---: | --- |"])
    for check in report.checks:
        result = "PASS" if check.get("ok") else "FAIL"
        lines.append(
            "| {name} | {result} | {value} | {threshold} | {severity} |".format(
                name=check.get("name"),
                result=result,
                value=_format_value(check.get("value")),
                threshold=_format_value(check.get("threshold")),
                severity=check.get("severity", "error"),
            )
        )

    lines.extend(["", "## Acceptance thresholds", "", "```json"])
    lines.append(json.dumps(report.acceptance_thresholds, indent=2, sort_keys=True))
    lines.extend(["```", "", "## Scenario context", "", "```json"])
    lines.append(
        json.dumps(
            {
                "task_context": report.task_context,
                "environment_context": report.environment_context,
                "command_space": report.command_space,
            },
            indent=2,
            sort_keys=True,
        )
    )
    lines.extend(["```", "", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.warnings) if report.warnings else lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Soridormi MuJoCo rollout JSONL against one scenario.")
    parser.add_argument("--scenario", required=True, help="Scenario id from configs/scenarios/open_duck_mini_v2_scenarios.json")
    parser.add_argument("--scenario-manifest", type=Path, default=DEFAULT_SCENARIO_MANIFEST)
    parser.add_argument("--log", type=Path, default=None, help="Runtime JSONL log to evaluate")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profile used for generated run plans")
    parser.add_argument("--duration-s", type=float, default=None, help="Duration override for generated run plans")
    parser.add_argument("--steps", type=int, default=None, help="Step-count override for generated run plans")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--log-prefix", default=None)
    parser.add_argument("--log-dir", default="/data/logs")
    parser.add_argument("--print-run-plan", action="store_true", help="Print the deterministic skill-run plan and exit")
    parser.add_argument("--fallback-control-hz", type=float, default=50.0)
    parser.add_argument("--min-distance-m", type=float, default=None, help="Override manifest min_distance_m")
    parser.add_argument(
        "--min-mean-forward-speed-mps", type=float, default=None, help="Override manifest min_mean_forward_speed_mps"
    )
    parser.add_argument("--max-stuck-sample-ratio", type=float, default=None, help="Override manifest max_stuck_sample_ratio")
    parser.add_argument("--allow-fallen", action="store_true", help="Override manifest require_not_fallen=false")
    parser.add_argument("--min-touchdown-count", type=int, default=None, help="Override manifest min_touchdown_count")
    parser.add_argument("--min-swing-clearance-m", type=float, default=None, help="Override manifest min_swing_clearance_m")
    parser.add_argument("--max-low-clearance-ratio", type=float, default=None, help="Override manifest max_low_clearance_ratio")
    parser.add_argument("--require-foot-metrics", action="store_true", help="Override manifest require_foot_metrics=true")
    parser.add_argument("--min-base-z-m", type=float, default=None, help="Override manifest min_base_z_m")
    parser.add_argument("--max-abs-roll-pitch-rad", type=float, default=None, help="Override manifest max_abs_roll_pitch_rad")
    parser.add_argument("--contact-threshold", type=float, default=None, help="Override manifest contact_threshold")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.print_run_plan:
        plan = build_scenario_run_plan(
            args.scenario,
            manifest_path=args.scenario_manifest,
            profile=args.profile,
            duration_s=args.duration_s,
            steps=args.steps,
            control_hz=args.control_hz,
            log_prefix=args.log_prefix,
            log_dir=args.log_dir,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.log is None:
        raise SystemExit("--log is required unless --print-run-plan is used")

    override_requested = any(
        value is not None
        for value in (
            args.min_distance_m,
            args.min_mean_forward_speed_mps,
            args.max_stuck_sample_ratio,
            args.min_touchdown_count,
            args.min_swing_clearance_m,
            args.max_low_clearance_ratio,
            args.min_base_z_m,
            args.max_abs_roll_pitch_rad,
            args.contact_threshold,
        )
    ) or args.allow_fallen or args.require_foot_metrics
    resolved_thresholds = None
    if override_requested:
        resolved_thresholds = overlay_threshold_overrides(
            thresholds_from_scenario_manifest(get_scenario_definition(args.scenario, args.scenario_manifest)),
            min_distance_m=args.min_distance_m,
            min_mean_forward_speed_mps=args.min_mean_forward_speed_mps,
            max_stuck_sample_ratio=args.max_stuck_sample_ratio,
            require_not_fallen=False if args.allow_fallen else None,
            min_touchdown_count=args.min_touchdown_count,
            min_swing_clearance_m=args.min_swing_clearance_m,
            max_low_clearance_ratio=args.max_low_clearance_ratio,
            require_foot_metrics=True if args.require_foot_metrics else None,
            min_base_z_m=args.min_base_z_m,
            max_abs_roll_pitch_rad=args.max_abs_roll_pitch_rad,
            contact_threshold=args.contact_threshold,
        )

    report = evaluate_scenario_rollout(
        args.log,
        scenario_id=args.scenario,
        manifest_path=args.scenario_manifest,
        thresholds=resolved_thresholds,
        fallback_control_hz=args.fallback_control_hz,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
