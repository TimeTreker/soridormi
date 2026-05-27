from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soridormi_runtime.check_policy_model import PolicyCheckResult, check_profile_model
from soridormi_runtime.onnx_providers import parse_provider_csv
from soridormi_runtime.policy_contract import PolicyContractResult, build_policy_contract
from soridormi_runtime.policy_profiles import PolicyProfile


@dataclass(frozen=True)
class PolicyModelArtifactInfo:
    declared_path: str
    resolved_path: str
    exists: bool
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class PolicyManifestResult:
    ok: bool
    manifest_version: int
    generated_at_utc: str
    profile_name: str
    profile_path: str
    robot_config_path: str
    model_artifact: PolicyModelArtifactInfo
    contract: dict[str, Any]
    model_check: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_artifact(path: str | Path, *, hash_model: bool = True) -> PolicyModelArtifactInfo:
    declared = str(path)
    resolved = Path(path).expanduser()
    exists = resolved.exists()
    if not exists:
        return PolicyModelArtifactInfo(declared_path=declared, resolved_path=str(resolved), exists=False)

    size_bytes = resolved.stat().st_size
    sha256 = _sha256_file(resolved) if hash_model else None
    return PolicyModelArtifactInfo(
        declared_path=declared,
        resolved_path=str(resolved),
        exists=True,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _contract_to_dict(contract: PolicyContractResult) -> dict[str, Any]:
    return asdict(contract)


def _model_check_to_dict(result: PolicyCheckResult | None) -> dict[str, Any] | None:
    return None if result is None else asdict(result)


def build_policy_manifest(
    profile: str | Path | PolicyProfile,
    *,
    robot_config_path: str | Path | None = None,
    check_model: bool = False,
    require_model: bool = False,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
    hash_model: bool = True,
) -> PolicyManifestResult:
    """Build a reproducible manifest for one policy replacement profile.

    Static mode does not require the ONNX file to exist. Use ``require_model`` or
    ``check_model`` when producing a release/preflight manifest where the model
    artifact must be present.
    """
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
    contract = build_policy_contract(policy_profile, robot_config_path=robot_config_path)
    artifact = inspect_model_artifact(policy_profile.model.path, hash_model=hash_model)

    model_check: PolicyCheckResult | None = None
    errors: list[str] = list(contract.errors)
    warnings: list[str] = list(contract.warnings)

    if not artifact.exists:
        message = f"Model artifact not found at declared path: {artifact.declared_path}"
        if require_model or check_model:
            errors.append(message)
        else:
            warnings.append(message)

    if check_model:
        model_check = check_profile_model(
            policy_profile,
            robot_config_path=robot_config_path,
            providers=providers,
            require_providers=require_providers,
            prefer_cuda=prefer_cuda,
        )
        errors.extend(error for error in model_check.errors if error not in errors)
        warnings.extend(warning for warning in model_check.warnings if warning not in warnings)

    return PolicyManifestResult(
        ok=not errors,
        manifest_version=1,
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        profile_name=policy_profile.name,
        profile_path=str(policy_profile.path),
        robot_config_path=contract.robot_config_path,
        model_artifact=artifact,
        contract=_contract_to_dict(contract),
        model_check=_model_check_to_dict(model_check),
        errors=errors,
        warnings=warnings,
    )


def print_manifest_summary(result: PolicyManifestResult) -> None:
    print("Soridormi policy replacement manifest")
    print("======================================")
    print(f"Profile: {result.profile_name}")
    print(f"Profile file: {result.profile_path}")
    print(f"Robot config: {result.robot_config_path}")
    print(f"Model path: {result.model_artifact.declared_path}")
    print(f"Model exists: {'yes' if result.model_artifact.exists else 'no'}")
    if result.model_artifact.exists:
        print(f"Model size: {result.model_artifact.size_bytes} bytes")
        if result.model_artifact.sha256:
            print(f"Model sha256: {result.model_artifact.sha256}")
    print("Runtime contract:", "OK" if result.contract.get("ok") else "FAILED")
    if result.model_check is not None:
        print("ONNX model check:", "OK" if result.model_check.get("ok") else "FAILED")
        providers = result.model_check.get("providers") or []
        if providers:
            print(f"Providers: {providers}")
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
        description="Export a reproducible Soridormi policy replacement manifest."
    )
    parser.add_argument("profile", help="Policy profile name or YAML path")
    parser.add_argument("--robot-config", default=None, help="Robot YAML path used for runtime contracts")
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also load the ONNX file and include input/output/provider validation.",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if the declared ONNX model file is not present.",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help="Comma-separated ONNX Runtime provider order used with --check-model.",
    )
    parser.add_argument(
        "--require-provider",
        action="append",
        default=None,
        help="ONNX provider that must be active with --check-model. May be repeated.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Prefer CPU when --check-model is used and no explicit providers are set.",
    )
    parser.add_argument("--no-hash", action="store_true", help="Do not compute a SHA256 hash")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    result = build_policy_manifest(
        args.profile,
        robot_config_path=args.robot_config,
        check_model=args.check_model,
        require_model=args.require_model,
        providers=parse_provider_csv(args.providers) or None,
        require_providers=args.require_provider,
        prefer_cuda=not args.cpu,
        hash_model=not args.no_hash,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_manifest_summary(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
