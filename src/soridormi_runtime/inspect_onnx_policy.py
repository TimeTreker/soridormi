from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POLICY_PATH = "/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[Any]
    dtype: str


@dataclass(frozen=True)
class PolicyInspection:
    policy_path: str
    onnxruntime_version: str
    available_providers: list[str]
    selected_providers: list[str]
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    dummy_output_shapes: dict[str, list[int]]
    dummy_output_dtypes: dict[str, str]


def resolve_policy_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_POLICY_PATH") or DEFAULT_POLICY_PATH
    return Path(explicit)


def prefer_cuda_provider() -> bool:
    value = os.environ.get("SORIDORMI_USE_CUDA_PROVIDER", "1")
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def select_onnxruntime_providers(available: list[str]) -> list[str]:
    providers: list[str] = []

    if prefer_cuda_provider() and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        # Let ONNX Runtime choose its default provider list.
        return available

    return providers


def normalize_dummy_dim(dim: Any) -> int:
    """Convert ONNX dynamic/static dimensions into safe dummy dimensions."""
    if isinstance(dim, int) and dim > 0:
        return dim
    return 1


def normalize_dummy_shape(shape: list[Any]) -> list[int]:
    if not shape:
        return [1]
    return [normalize_dummy_dim(dim) for dim in shape]


def dtype_from_onnxruntime_type(type_name: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(uint64)": np.dtype(np.uint64),
        "tensor(uint32)": np.dtype(np.uint32),
        "tensor(uint16)": np.dtype(np.uint16),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    return mapping.get(type_name, np.dtype(np.float32))


def make_dummy_input(shape: list[Any], type_name: str) -> np.ndarray:
    dummy_shape = normalize_dummy_shape(shape)
    dtype = dtype_from_onnxruntime_type(type_name)
    return np.zeros(dummy_shape, dtype=dtype)


def _tensor_info(value: Any) -> TensorInfo:
    return TensorInfo(
        name=str(value.name),
        shape=list(value.shape),
        dtype=str(value.type),
    )


def inspect_policy(policy_path: str | os.PathLike[str] | None = None) -> PolicyInspection:
    import onnxruntime as ort

    path = resolve_policy_path(policy_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ONNX policy not found: {path}. "
            "Set SORIDORMI_POLICY_PATH or pass the model path explicitly."
        )

    available = list(ort.get_available_providers())
    providers = select_onnxruntime_providers(available)

    session = ort.InferenceSession(str(path), providers=providers)

    inputs = [_tensor_info(v) for v in session.get_inputs()]
    outputs = [_tensor_info(v) for v in session.get_outputs()]

    dummy_feed = {
        item.name: make_dummy_input(item.shape, item.dtype)
        for item in inputs
    }
    output_names = [item.name for item in outputs]
    dummy_outputs = session.run(output_names, dummy_feed)

    dummy_output_shapes = {
        name: list(array.shape)
        for name, array in zip(output_names, dummy_outputs)
    }
    dummy_output_dtypes = {
        name: str(array.dtype)
        for name, array in zip(output_names, dummy_outputs)
    }

    return PolicyInspection(
        policy_path=str(path),
        onnxruntime_version=str(ort.__version__),
        available_providers=available,
        selected_providers=list(session.get_providers()),
        inputs=inputs,
        outputs=outputs,
        dummy_output_shapes=dummy_output_shapes,
        dummy_output_dtypes=dummy_output_dtypes,
    )


def print_human_readable(inspection: PolicyInspection) -> None:
    print("Soridormi ONNX policy inspection")
    print("================================")
    print(f"Policy path:          {inspection.policy_path}")
    print(f"ONNX Runtime version: {inspection.onnxruntime_version}")
    print(f"Available providers:  {inspection.available_providers}")
    print(f"Selected providers:   {inspection.selected_providers}")

    print()
    print("Inputs")
    print("------")
    for i, item in enumerate(inspection.inputs):
        print(f"{i:02d} name={item.name!r} shape={item.shape} dtype={item.dtype}")

    print()
    print("Outputs")
    print("-------")
    for i, item in enumerate(inspection.outputs):
        shape = inspection.dummy_output_shapes.get(item.name)
        dtype = inspection.dummy_output_dtypes.get(item.name)
        print(
            f"{i:02d} name={item.name!r} shape={item.shape} dtype={item.dtype} "
            f"dummy_output_shape={shape} dummy_output_dtype={dtype}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Soridormi/Open Duck ONNX policy.")
    parser.add_argument(
        "policy_path",
        nargs="?",
        default=None,
        help=f"Path to ONNX policy. Defaults to SORIDORMI_POLICY_PATH or {DEFAULT_POLICY_PATH}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    inspection = inspect_policy(args.policy_path)

    if args.json:
        print(json.dumps(asdict(inspection), indent=2))
    else:
        print_human_readable(inspection)


if __name__ == "__main__":
    main()
