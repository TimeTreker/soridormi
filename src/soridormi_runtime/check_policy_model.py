from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

try:
    import onnxruntime as ort  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depends on optional runtime dependency
    ort = None  # type: ignore[assignment]

from soridormi_runtime.linear_behavior_clone_policy import load_linear_behavior_clone_model
from soridormi_runtime.onnx_providers import parse_provider_csv, resolve_onnx_providers, verify_active_providers
from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_profiles import PolicyProfile


@dataclass(frozen=True)
class PolicyCheckResult:
    ok: bool
    policy_path: str
    providers: list[str]
    input_name: str | None
    input_shape: list[Any] | None
    input_type: str | None
    output_name: str | None
    output_shape: list[Any] | None
    output_type: str | None
    errors: list[str]
    warnings: list[str]
    profile_name: str | None = None
    profile_path: str | None = None
    robot_config_path: str | None = None
    contract_ok: bool | None = None
    contract_errors: list[str] | None = None
    contract_warnings: list[str] | None = None
    available_providers: list[str] | None = None
    requested_providers: list[str] | None = None
    required_providers: list[str] | None = None


def _is_dynamic_onnx_dim(value: Any) -> bool:
    if value in {None, "", "?", -1}:
        return True
    return isinstance(value, str) and not value.strip().lstrip("+-").isdigit()


def _shape_matches(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    for index, (a, e) in enumerate(zip(actual, expected)):
        if e in {None, "", "?", -1}:
            continue
        if _is_dynamic_onnx_dim(a):
            # ONNX exporters commonly emit symbolic batch names such as
            # "batch". Soridormi's runtime contract still sends batch size 1,
            # so allow a dynamic first dimension when the expected batch is 1
            # while keeping feature/action dimensions strict.
            if index == 0 and str(e) == "1":
                continue
            return False
        if str(a) != str(e):
            return False
    return True


def _meta_to_dict(meta: Any) -> dict[str, Any]:
    return {"name": str(getattr(meta, "name", "")), "shape": list(getattr(meta, "shape", []) or []), "type": str(getattr(meta, "type", ""))}


def _load_onnxruntime() -> Any:
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install the runtime/sim extras or run this check inside the runtime container."
        )
    return ort


def check_policy_model(
    policy_path: str | Path,
    *,
    expected_input_name: str = "obs",
    expected_output_name: str = "continuous_actions",
    expected_input_shape: list[Any] | None = None,
    expected_output_shape: list[Any] | None = None,
    expected_input_type: str = "tensor(float)",
    expected_output_type: str = "tensor(float)",
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> PolicyCheckResult:
    path = Path(policy_path)
    if not path.exists():
        return PolicyCheckResult(False, str(path), [], None, None, None, None, None, None, [f"Policy file not found: {path}"], [])

    try:
        ort = _load_onnxruntime()
    except RuntimeError as exc:
        return PolicyCheckResult(False, str(path), [], None, None, None, None, None, None, [str(exc)], [])

    available = list(ort.get_available_providers())
    selection = resolve_onnx_providers(
        available,
        requested=providers,
        required=require_providers,
        prefer_cuda=prefer_cuda,
    )
    if not selection.ok:
        return PolicyCheckResult(
            False,
            str(path),
            selection.providers,
            None,
            None,
            None,
            None,
            None,
            None,
            selection.errors,
            selection.warnings,
            available_providers=selection.available,
            requested_providers=selection.requested,
            required_providers=selection.required,
        )

    try:
        session = ort.InferenceSession(str(path), providers=selection.providers)
    except Exception as exc:
        return PolicyCheckResult(
            False,
            str(path),
            selection.providers,
            None,
            None,
            None,
            None,
            None,
            None,
            [f"Failed to load ONNX policy: {exc!r}"],
            selection.warnings,
            available_providers=selection.available,
            requested_providers=selection.requested,
            required_providers=selection.required,
        )

    active_providers = list(session.get_providers()) if hasattr(session, "get_providers") else list(selection.providers)
    provider_errors = verify_active_providers(
        active_providers,
        requested=selection.requested,
        required=selection.required,
    )

    inputs = [_meta_to_dict(item) for item in session.get_inputs()]
    outputs = [_meta_to_dict(item) for item in session.get_outputs()]
    errors: list[str] = list(provider_errors)
    warnings: list[str] = list(selection.warnings)

    input_meta = next((item for item in inputs if item["name"] == expected_input_name), None)
    output_meta = next((item for item in outputs if item["name"] == expected_output_name), None)
    if input_meta is None:
        errors.append(f"Expected input {expected_input_name!r} not found. Available inputs: {[i['name'] for i in inputs]}")
        input_meta = inputs[0] if inputs else {"name": None, "shape": None, "type": None}
    if output_meta is None:
        errors.append(f"Expected output {expected_output_name!r} not found. Available outputs: {[o['name'] for o in outputs]}")
        output_meta = outputs[0] if outputs else {"name": None, "shape": None, "type": None}
    if len(inputs) != 1:
        warnings.append(f"Expected one model input; found {len(inputs)}")
    if not outputs:
        errors.append("Policy model has no outputs")
    if expected_input_shape is not None and input_meta.get("shape") is not None and not _shape_matches(list(input_meta["shape"]), list(expected_input_shape)):
        errors.append(f"Input shape mismatch: expected {expected_input_shape}, got {input_meta['shape']}")
    if expected_output_shape is not None and output_meta.get("shape") is not None and not _shape_matches(list(output_meta["shape"]), list(expected_output_shape)):
        errors.append(f"Output shape mismatch: expected {expected_output_shape}, got {output_meta['shape']}")
    if expected_input_type and input_meta.get("type") and input_meta["type"] != expected_input_type:
        errors.append(f"Input type mismatch: expected {expected_input_type}, got {input_meta['type']}")
    if expected_output_type and output_meta.get("type") and output_meta["type"] != expected_output_type:
        errors.append(f"Output type mismatch: expected {expected_output_type}, got {output_meta['type']}")

    return PolicyCheckResult(
        not errors,
        str(path),
        active_providers,
        input_meta.get("name"),
        input_meta.get("shape"),
        input_meta.get("type"),
        output_meta.get("name"),
        output_meta.get("shape"),
        output_meta.get("type"),
        errors,
        warnings,
        available_providers=selection.available,
        requested_providers=selection.requested,
        required_providers=selection.required,
    )


def check_linear_behavior_clone_model(
    policy_path: str | Path,
    *,
    expected_input_name: str = "obs",
    expected_output_name: str = "continuous_actions",
    expected_input_shape: list[Any] | None = None,
    expected_output_shape: list[Any] | None = None,
    require_providers: list[str] | str | None = None,
) -> PolicyCheckResult:
    path = Path(policy_path)
    required = parse_provider_csv(require_providers)
    model = load_linear_behavior_clone_model(path)
    errors = list(model.errors)
    warnings = list(model.warnings)
    if required:
        warnings.append("ONNX execution provider requirements are ignored for linear_behavior_clone NPZ models")
    input_shape = [1, model.observation_size]
    output_shape = [1, model.action_size]
    if expected_input_shape is not None and not _shape_matches(input_shape, list(expected_input_shape)):
        errors.append(f"Input shape mismatch: expected {expected_input_shape}, got {input_shape}")
    if expected_output_shape is not None and not _shape_matches(output_shape, list(expected_output_shape)):
        errors.append(f"Output shape mismatch: expected {expected_output_shape}, got {output_shape}")
    return PolicyCheckResult(
        not errors,
        str(path),
        [],
        expected_input_name,
        input_shape,
        "tensor(float)",
        expected_output_name,
        output_shape,
        "tensor(float)",
        errors,
        warnings,
        requested_providers=[],
        required_providers=required,
    )


def check_profile_model(
    profile: PolicyProfile,
    model_path_override: str | Path | None = None,
    *,
    robot_config_path: str | Path | None = None,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> PolicyCheckResult:
    """Validate a policy profile against the runtime contract and ONNX metadata.

    This is the replaceable policy interface replacement-model gate used by scripts/check_policy_model.sh
    and run_policy_experiment.sh. It combines the static profile/robot contract
    from policy_contract.py with the actual ONNX file IO check, so a profile
    cannot pass merely by declaring shapes that disagree with Soridormi's runtime
    observation/action interface.
    """
    contract = build_policy_contract(profile, robot_config_path=robot_config_path)
    model = profile.model
    model_kind = str(getattr(model, "kind", "onnx")).strip().lower().replace("-", "_")
    if model_kind in {"linear", "linear_npz", "linear_behavior_clone", "behavior_clone_linear"}:
        model_result = check_linear_behavior_clone_model(
            model_path_override or model.path,
            expected_input_name=model.input_name,
            expected_output_name=model.output_name,
            expected_input_shape=model.input_shape,
            expected_output_shape=model.output_shape,
            require_providers=require_providers,
        )
    else:
        model_result = check_policy_model(
            model_path_override or model.path,
            expected_input_name=model.input_name,
            expected_output_name=model.output_name,
            expected_input_shape=model.input_shape,
            expected_output_shape=model.output_shape,
            expected_input_type=model.input_type,
            expected_output_type=model.output_type,
            providers=providers,
            require_providers=require_providers,
            prefer_cuda=prefer_cuda,
        )
    errors = [*contract.errors, *model_result.errors]
    warnings = [*contract.warnings, *model_result.warnings]
    return replace(
        model_result,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        profile_name=profile.name,
        profile_path=str(profile.path),
        robot_config_path=contract.robot_config_path,
        contract_ok=contract.ok,
        contract_errors=list(contract.errors),
        contract_warnings=list(contract.warnings),
    )


def print_result(result: PolicyCheckResult) -> None:
    print("Soridormi policy model check")
    print("============================")
    if result.profile_name:
        print(f"Profile: {result.profile_name}")
        print(f"Profile file: {result.profile_path}")
    if result.robot_config_path:
        print(f"Robot config: {result.robot_config_path}")
    if result.contract_ok is not None:
        print("Runtime contract:", "OK" if result.contract_ok else "FAILED")
    print(f"Policy path: {result.policy_path}")
    if result.available_providers is not None:
        print(f"Available providers: {result.available_providers}")
    if result.requested_providers:
        print(f"Requested providers: {result.requested_providers}")
    if result.required_providers:
        print(f"Required providers: {result.required_providers}")
    print(f"Providers: {result.providers}")
    print(f"Input:  name={result.input_name!r} shape={result.input_shape} type={result.input_type}")
    print(f"Output: name={result.output_name!r} shape={result.output_shape} type={result.output_type}")
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
    parser = argparse.ArgumentParser(description="Validate an ONNX model against a Soridormi policy contract.")
    parser.add_argument("model_path", nargs="?", help="Optional ONNX path. Overrides profile model.path.")
    parser.add_argument("--profile", default=None, help="Policy profile name or YAML path")
    parser.add_argument("--robot-config", default=None, help="Robot YAML path used for runtime observation/action contract")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--providers",
        default=None,
        help="Comma-separated ONNX Runtime provider order, e.g. CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--require-provider",
        action="append",
        default=None,
        help="Provider that must be active. May be repeated. Useful for GPU preflight gates.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Prefer CPU when no explicit --providers/SORIDORMI_ONNX_PROVIDERS is set.",
    )
    args = parser.parse_args()

    profile = PolicyProfile.load(args.profile)
    result = check_profile_model(
        profile,
        model_path_override=args.model_path,
        robot_config_path=args.robot_config,
        providers=parse_provider_csv(args.providers) or None,
        require_providers=args.require_provider,
        prefer_cuda=not args.cpu,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_result(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
