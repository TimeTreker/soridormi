from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort

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


def _shape_matches(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e in {None, "", "?", -1}:
            continue
        if a in {None, "", "?"}:
            continue
        if str(a) != str(e):
            return False
    return True


def _meta_to_dict(meta: Any) -> dict[str, Any]:
    return {"name": str(getattr(meta, "name", "")), "shape": list(getattr(meta, "shape", []) or []), "type": str(getattr(meta, "type", ""))}


def check_policy_model(
    policy_path: str | Path,
    *,
    expected_input_name: str = "obs",
    expected_output_name: str = "continuous_actions",
    expected_input_shape: list[Any] | None = None,
    expected_output_shape: list[Any] | None = None,
    expected_input_type: str = "tensor(float)",
    expected_output_type: str = "tensor(float)",
    providers: list[str] | None = None,
) -> PolicyCheckResult:
    path = Path(policy_path)
    if not path.exists():
        return PolicyCheckResult(False, str(path), [], None, None, None, None, None, None, [f"Policy file not found: {path}"], [])

    available = list(ort.get_available_providers())
    selected = providers or (["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available[:1])
    try:
        session = ort.InferenceSession(str(path), providers=selected)
    except Exception as exc:
        return PolicyCheckResult(False, str(path), selected, None, None, None, None, None, None, [f"Failed to load ONNX policy: {exc!r}"], [])

    inputs = [_meta_to_dict(item) for item in session.get_inputs()]
    outputs = [_meta_to_dict(item) for item in session.get_outputs()]
    errors: list[str] = []
    warnings: list[str] = []

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

    return PolicyCheckResult(not errors, str(path), selected, input_meta.get("name"), input_meta.get("shape"), input_meta.get("type"), output_meta.get("name"), output_meta.get("shape"), output_meta.get("type"), errors, warnings)


def check_profile_model(profile: PolicyProfile, model_path_override: str | Path | None = None) -> PolicyCheckResult:
    model = profile.model
    return check_policy_model(
        model_path_override or model.path,
        expected_input_name=model.input_name,
        expected_output_name=model.output_name,
        expected_input_shape=model.input_shape,
        expected_output_shape=model.output_shape,
        expected_input_type=model.input_type,
        expected_output_type=model.output_type,
    )


def print_result(result: PolicyCheckResult) -> None:
    print("Soridormi policy model check")
    print("============================")
    print(f"Policy path: {result.policy_path}")
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    profile = PolicyProfile.load(args.profile)
    result = check_profile_model(profile, model_path_override=args.model_path)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_result(result)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
