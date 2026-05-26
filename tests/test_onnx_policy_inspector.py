from __future__ import annotations

import numpy as np

from soridormi_runtime.inspect_onnx_policy import (
    dtype_from_onnxruntime_type,
    make_dummy_input,
    normalize_dummy_shape,
    select_onnxruntime_providers,
)


def test_normalize_dummy_shape_replaces_dynamic_dims() -> None:
    assert normalize_dummy_shape(["batch", 48]) == [1, 48]
    assert normalize_dummy_shape([None, "obs_dim"]) == [1, 1]
    assert normalize_dummy_shape([]) == [1]


def test_make_dummy_input_uses_dtype_and_shape() -> None:
    array = make_dummy_input(["batch", 12], "tensor(float)")
    assert array.shape == (1, 12)
    assert array.dtype == np.float32

    array_i64 = make_dummy_input([2, 3], "tensor(int64)")
    assert array_i64.shape == (2, 3)
    assert array_i64.dtype == np.int64


def test_dtype_mapping_falls_back_to_float32() -> None:
    assert dtype_from_onnxruntime_type("tensor(float)") == np.dtype(np.float32)
    assert dtype_from_onnxruntime_type("unknown") == np.dtype(np.float32)


def test_provider_selection_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setenv("SORIDORMI_USE_CUDA_PROVIDER", "1")
    assert select_onnxruntime_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setenv("SORIDORMI_USE_CUDA_PROVIDER", "0")
    assert select_onnxruntime_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ) == ["CPUExecutionProvider"]
