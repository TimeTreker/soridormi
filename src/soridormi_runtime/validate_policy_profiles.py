from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from soridormi_runtime.check_policy_model import check_profile_model
from soridormi_runtime.onnx_providers import parse_provider_csv
from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_profiles import PolicyProfile, list_policy_profiles


@dataclass(frozen=True)
class PolicyProfileValidationResult:
    name: str
    path: str
    ok: bool
    contract_ok: bool
    model_checked: bool
    model_ok: bool | None
    model_path: str
    providers: list[str]
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class PolicyProfileSuiteResult:
    ok: bool
    profile_count: int
    model_checked_count: int
    results: list[PolicyProfileValidationResult]
    errors: list[str]
    warnings: list[str]


def _load_profiles(profiles: Iterable[str | Path] | None = None) -> list[PolicyProfile]:
    requested = list(profiles or [])
    if requested:
        return [PolicyProfile.load(item) for item in requested]
    paths = list_policy_profiles()
    return [PolicyProfile.load(path) for path in paths]


def validate_policy_profile(
    profile: str | Path | PolicyProfile,
    *,
    robot_config_path: str | Path | None = None,
    check_model: bool = False,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> PolicyProfileValidationResult:
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)

    if check_model:
        model_result = check_profile_model(
            policy_profile,
            robot_config_path=robot_config_path,
            providers=providers,
            require_providers=require_providers,
            prefer_cuda=prefer_cuda,
        )
        contract_errors = list(model_result.contract_errors or [])
        model_errors = [error for error in model_result.errors if error not in contract_errors]
        return PolicyProfileValidationResult(
            name=policy_profile.name,
            path=str(policy_profile.path),
            ok=model_result.ok,
            contract_ok=bool(model_result.contract_ok),
            model_checked=True,
            model_ok=not model_errors,
            model_path=model_result.policy_path,
            providers=list(model_result.providers),
            errors=list(model_result.errors),
            warnings=list(model_result.warnings),
        )

    contract = build_policy_contract(policy_profile, robot_config_path=robot_config_path)
    return PolicyProfileValidationResult(
        name=policy_profile.name,
        path=str(policy_profile.path),
        ok=contract.ok,
        contract_ok=contract.ok,
        model_checked=False,
        model_ok=None,
        model_path=str(contract.model["path"]),
        providers=[],
        errors=list(contract.errors),
        warnings=list(contract.warnings),
    )


def validate_policy_profiles(
    profiles: Iterable[str | Path] | None = None,
    *,
    robot_config_path: str | Path | None = None,
    check_models: bool = False,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> PolicyProfileSuiteResult:
    suite_errors: list[str] = []
    suite_warnings: list[str] = []
    try:
        loaded_profiles = _load_profiles(profiles)
    except Exception as exc:
        return PolicyProfileSuiteResult(
            ok=False,
            profile_count=0,
            model_checked_count=0,
            results=[],
            errors=[f"Failed to load policy profiles: {exc!r}"],
            warnings=[],
        )

    if not loaded_profiles:
        suite_errors.append("No policy profiles found")

    results: list[PolicyProfileValidationResult] = []
    for profile in loaded_profiles:
        try:
            result = validate_policy_profile(
                profile,
                robot_config_path=robot_config_path,
                check_model=check_models,
                providers=providers,
                require_providers=require_providers,
                prefer_cuda=prefer_cuda,
            )
        except Exception as exc:
            result = PolicyProfileValidationResult(
                name=getattr(profile, "name", "<unknown>"),
                path=str(getattr(profile, "path", "")),
                ok=False,
                contract_ok=False,
                model_checked=check_models,
                model_ok=False if check_models else None,
                model_path=str(getattr(getattr(profile, "model", None), "path", "")),
                providers=[],
                errors=[f"Unhandled profile validation error: {exc!r}"],
                warnings=[],
            )
        results.append(result)

    profile_names = [result.name for result in results]
    duplicate_names = sorted({name for name in profile_names if profile_names.count(name) > 1})
    for name in duplicate_names:
        suite_errors.append(f"Duplicate policy profile name: {name}")

    all_warnings = [warning for result in results for warning in result.warnings]
    suite_warnings.extend(all_warnings)
    ok = not suite_errors and all(result.ok for result in results)
    return PolicyProfileSuiteResult(
        ok=ok,
        profile_count=len(results),
        model_checked_count=sum(1 for result in results if result.model_checked),
        results=results,
        errors=suite_errors,
        warnings=suite_warnings,
    )


def print_suite_result(result: PolicyProfileSuiteResult) -> None:
    print("Soridormi policy profile validation")
    print("====================================")
    print(f"Profiles: {result.profile_count}")
    print(f"Model checks: {result.model_checked_count}")
    for item in result.results:
        status = "OK" if item.ok else "FAILED"
        print(f"- {item.name}: {status}")
        print(f"  file: {item.path}")
        print(f"  model: {item.model_path}")
        print(f"  contract: {'OK' if item.contract_ok else 'FAILED'}")
        if item.model_checked:
            print(f"  ONNX model: {'OK' if item.model_ok else 'FAILED'}")
            if item.providers:
                print(f"  providers: {item.providers}")
        if item.warnings:
            print("  warnings:")
            for warning in item.warnings:
                print(f"    - {warning}")
        if item.errors:
            print("  errors:")
            for error in item.errors:
                print(f"    - {error}")
    if result.errors:
        print("Suite errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Soridormi policy profiles as a model-replacement suite."
    )
    parser.add_argument(
        "profiles",
        nargs="*",
        help="Optional profile names or YAML paths. Defaults to all configs/policies/*.yaml profiles.",
    )
    parser.add_argument("--robot-config", default=None, help="Robot YAML path used for runtime contracts")
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Also load each ONNX model and validate metadata/provider selection.",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help="Comma-separated ONNX Runtime provider order used with --check-models.",
    )
    parser.add_argument(
        "--require-provider",
        action="append",
        default=None,
        help="ONNX provider that must be active with --check-models. May be repeated.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Prefer CPU when --check-models is used and no explicit providers are set.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    result = validate_policy_profiles(
        args.profiles,
        robot_config_path=args.robot_config,
        check_models=args.check_models,
        providers=parse_provider_csv(args.providers) or None,
        require_providers=args.require_provider,
        prefer_cuda=not args.cpu,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True, default=_json_default))
    else:
        print_suite_result(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
