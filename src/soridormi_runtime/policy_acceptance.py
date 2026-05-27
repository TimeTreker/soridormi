from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soridormi_runtime.onnx_providers import parse_provider_csv
from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_manifest import build_policy_manifest
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.validate_policy_profiles import validate_policy_profiles


@dataclass(frozen=True)
class PolicyAcceptanceArtifacts:
    directory: str
    contract_json: str
    manifest_json: str
    suite_validation_json: str
    report_markdown: str


@dataclass(frozen=True)
class PolicyAcceptanceResult:
    ok: bool
    acceptance_version: int
    generated_at_utc: str
    profile_name: str
    profile_path: str
    artifacts: PolicyAcceptanceArtifacts
    contract_ok: bool
    manifest_ok: bool
    suite_ok: bool
    model_checked: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    value = value.strip("._-")
    return value or "policy"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_suite_targets(profile: PolicyProfile, *, profile_only: bool) -> list[str] | None:
    if profile_only:
        return [str(profile.path)]
    return None


def _report_markdown(
    result: PolicyAcceptanceResult,
    *,
    contract_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    suite_payload: dict[str, Any],
) -> str:
    model = manifest_payload.get("model_artifact", {})
    active_providers: list[str] = []
    model_check = manifest_payload.get("model_check")
    if isinstance(model_check, dict):
        active_providers = list(model_check.get("providers") or [])

    lines = [
        "# Soridormi policy acceptance report",
        "",
        f"- Result: {'OK' if result.ok else 'FAILED'}",
        f"- Generated: {result.generated_at_utc}",
        f"- Profile: `{result.profile_name}`",
        f"- Profile file: `{result.profile_path}`",
        f"- Model checked: {'yes' if result.model_checked else 'no'}",
        "",
        "## Gates",
        "",
        f"- Static contract: {'OK' if result.contract_ok else 'FAILED'}",
        f"- Manifest: {'OK' if result.manifest_ok else 'FAILED'}",
        f"- Profile suite: {'OK' if result.suite_ok else 'FAILED'}",
        "",
        "## Model artifact",
        "",
        f"- Declared path: `{model.get('declared_path', '')}`",
        f"- Resolved path: `{model.get('resolved_path', '')}`",
        f"- Exists: {'yes' if model.get('exists') else 'no'}",
    ]
    if model.get("size_bytes") is not None:
        lines.append(f"- Size bytes: {model.get('size_bytes')}")
    if model.get("sha256"):
        lines.append(f"- SHA256: `{model.get('sha256')}`")
    if active_providers:
        lines.append(f"- Active ONNX providers: `{', '.join(active_providers)}`")

    observation = contract_payload.get("observation", {})
    action = contract_payload.get("action", {})
    lines.extend(
        [
            "",
            "## Runtime contract",
            "",
            f"- Observation size: {observation.get('size')}",
            f"- Action size: {action.get('size')}",
            f"- Action scale: {action.get('action_scale')}",
            f"- Max motor velocity: {action.get('max_motor_velocity')}",
            "",
            "## Artifacts",
            "",
            f"- Contract JSON: `{result.artifacts.contract_json}`",
            f"- Manifest JSON: `{result.artifacts.manifest_json}`",
            f"- Suite validation JSON: `{result.artifacts.suite_validation_json}`",
            f"- Report: `{result.artifacts.report_markdown}`",
        ]
    )

    suite_results = suite_payload.get("results") or []
    if suite_results:
        lines.extend(["", "## Profile-suite summary", ""])
        for item in suite_results:
            status = "OK" if item.get("ok") else "FAILED"
            lines.append(f"- `{item.get('name')}`: {status}")

    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in result.warnings:
            lines.append(f"- {warning}")
    if result.errors:
        lines.extend(["", "## Errors", ""])
        for error in result.errors:
            lines.append(f"- {error}")

    lines.append("")
    return "\n".join(lines)


def accept_policy_profile(
    profile: str | Path | PolicyProfile,
    *,
    output_dir: str | Path = "data/policy_acceptance",
    robot_config_path: str | Path | None = None,
    check_model: bool = False,
    require_model: bool = False,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
    hash_model: bool = True,
    profile_only: bool = False,
    force: bool = False,
) -> PolicyAcceptanceResult:
    """Run the replacement-profile acceptance gate and write artifacts.

    The gate intentionally does not start MuJoCo or run a policy experiment. It is
    the model-replacement handoff step between profile creation and simulation:
    static contract, manifest/model metadata, and profile-suite validation all
    land in one reproducible artifact directory.
    """
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
    stamp = _utc_stamp()
    root = Path(output_dir).expanduser()
    run_dir = root / f"{_slug(policy_profile.name)}_{stamp}"
    if run_dir.exists() and not force:
        raise FileExistsError(f"Acceptance output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    contract = build_policy_contract(policy_profile, robot_config_path=robot_config_path)
    manifest = build_policy_manifest(
        policy_profile,
        robot_config_path=robot_config_path,
        check_model=check_model,
        require_model=require_model,
        providers=providers,
        require_providers=require_providers,
        prefer_cuda=prefer_cuda,
        hash_model=hash_model,
    )
    suite = validate_policy_profiles(
        _profile_suite_targets(policy_profile, profile_only=profile_only),
        robot_config_path=robot_config_path,
        check_models=check_model,
        providers=providers,
        require_providers=require_providers,
        prefer_cuda=prefer_cuda,
    )

    contract_path = run_dir / "contract.json"
    manifest_path = run_dir / "manifest.json"
    suite_path = run_dir / "profile_suite.json"
    report_path = run_dir / "acceptance_report.md"

    contract_payload = asdict(contract)
    manifest_payload = asdict(manifest)
    suite_payload = asdict(suite)
    _write_json(contract_path, contract_payload)
    _write_json(manifest_path, manifest_payload)
    _write_json(suite_path, suite_payload)

    errors: list[str] = []
    warnings: list[str] = []
    for source in (contract, manifest, suite):
        for error in getattr(source, "errors", []):
            if error not in errors:
                errors.append(error)
        for warning in getattr(source, "warnings", []):
            if warning not in warnings:
                warnings.append(warning)

    artifacts = PolicyAcceptanceArtifacts(
        directory=str(run_dir),
        contract_json=str(contract_path),
        manifest_json=str(manifest_path),
        suite_validation_json=str(suite_path),
        report_markdown=str(report_path),
    )
    result = PolicyAcceptanceResult(
        ok=contract.ok and manifest.ok and suite.ok and not errors,
        acceptance_version=1,
        generated_at_utc=_utc_iso(),
        profile_name=policy_profile.name,
        profile_path=str(policy_profile.path),
        artifacts=artifacts,
        contract_ok=contract.ok,
        manifest_ok=manifest.ok,
        suite_ok=suite.ok,
        model_checked=check_model,
        errors=errors,
        warnings=warnings,
    )

    report_path.write_text(
        _report_markdown(
            result,
            contract_payload=contract_payload,
            manifest_payload=manifest_payload,
            suite_payload=suite_payload,
        ),
        encoding="utf-8",
    )
    _write_json(run_dir / "acceptance.json", asdict(result))
    return result


def print_acceptance_summary(result: PolicyAcceptanceResult) -> None:
    print("Soridormi policy acceptance gate")
    print("=================================")
    print(f"Profile: {result.profile_name}")
    print(f"Profile file: {result.profile_path}")
    print(f"Output directory: {result.artifacts.directory}")
    print(f"Static contract: {'OK' if result.contract_ok else 'FAILED'}")
    print(f"Manifest: {'OK' if result.manifest_ok else 'FAILED'}")
    print(f"Profile suite: {'OK' if result.suite_ok else 'FAILED'}")
    print(f"Model checked: {'yes' if result.model_checked else 'no'}")
    print("Artifacts:")
    print(f"  contract: {result.artifacts.contract_json}")
    print(f"  manifest: {result.artifacts.manifest_json}")
    print(f"  suite: {result.artifacts.suite_validation_json}")
    print(f"  report: {result.artifacts.report_markdown}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Soridormi replacement-policy acceptance gate and write artifacts."
    )
    parser.add_argument("profile", help="Policy profile name or YAML path")
    parser.add_argument(
        "--output-dir",
        default="data/policy_acceptance",
        help="Directory where a timestamped acceptance artifact folder is created.",
    )
    parser.add_argument("--robot-config", default=None, help="Robot YAML path used for runtime contracts")
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also load the ONNX model and validate metadata/provider selection.",
    )
    parser.add_argument("--require-model", action="store_true", help="Fail if the ONNX model is missing")
    parser.add_argument("--providers", default=None, help="Comma-separated ONNX Runtime provider order")
    parser.add_argument(
        "--require-provider",
        action="append",
        default=None,
        help="ONNX provider that must be active with --check-model. May be repeated.",
    )
    parser.add_argument("--cpu", action="store_true", help="Prefer CPU when checking a model")
    parser.add_argument("--no-hash", action="store_true", help="Do not hash the ONNX model artifact")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Validate only this profile in the suite step instead of all configured profiles.",
    )
    parser.add_argument("--force", action="store_true", help="Allow reusing an existing output directory")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    result = accept_policy_profile(
        args.profile,
        output_dir=args.output_dir,
        robot_config_path=args.robot_config,
        check_model=args.check_model,
        require_model=args.require_model,
        providers=parse_provider_csv(args.providers) or None,
        require_providers=args.require_provider,
        prefer_cuda=not args.cpu,
        hash_model=not args.no_hash,
        profile_only=args.profile_only,
        force=args.force,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_acceptance_summary(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
