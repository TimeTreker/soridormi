from __future__ import annotations

from pathlib import Path
from typing import Any

from soridormi_runtime.onnx_providers import parse_provider_csv, resolve_onnx_providers


class FakeIo:
    def __init__(self, name: str, shape: list[int], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    active_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, path: str, providers=None) -> None:
        self.path = path
        self.requested_providers = providers

    def get_providers(self):
        return list(self.active_providers)

    def get_inputs(self):
        return [FakeIo("obs", [1, 101])]

    def get_outputs(self):
        return [FakeIo("continuous_actions", [1, 14])]


class FakeCudaOrt:
    @staticmethod
    def get_available_providers():
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    InferenceSession = FakeSession


class FakeCpuOrt:
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]

    InferenceSession = FakeSession


def test_parse_provider_csv_preserves_order_and_deduplicates() -> None:
    assert parse_provider_csv("CUDAExecutionProvider, CPUExecutionProvider CUDAExecutionProvider") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_resolve_onnx_providers_prefers_cuda_but_not_tensorrt_by_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("SORIDORMI_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("SORIDORMI_ONNX_REQUIRE_PROVIDER", raising=False)

    selection = resolve_onnx_providers(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    assert selection.ok
    assert selection.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_onnx_providers_env_request_is_strict(monkeypatch: Any) -> None:
    monkeypatch.setenv("SORIDORMI_ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")

    selection = resolve_onnx_providers(["CPUExecutionProvider"])

    assert not selection.ok
    assert "CUDAExecutionProvider" in selection.errors[0]


def test_check_policy_model_uses_cuda_provider_when_available(tmp_path: Path, monkeypatch: Any) -> None:
    from soridormi_runtime import check_policy_model as checker

    model = tmp_path / "policy.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(checker, "ort", FakeCudaOrt)
    monkeypatch.delenv("SORIDORMI_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("SORIDORMI_ONNX_REQUIRE_PROVIDER", raising=False)

    result = checker.check_policy_model(model)

    assert result.ok
    assert result.available_providers == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert result.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_check_policy_model_fails_when_required_provider_missing(tmp_path: Path, monkeypatch: Any) -> None:
    from soridormi_runtime import check_policy_model as checker

    model = tmp_path / "policy.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(checker, "ort", FakeCpuOrt)

    result = checker.check_policy_model(model, require_providers=["CUDAExecutionProvider"])

    assert not result.ok
    assert any("Required ONNX provider" in error for error in result.errors)
