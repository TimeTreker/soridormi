from __future__ import annotations

from pathlib import Path

from soridormi_runtime import check_policy_model as checker


class FakeIo:
    def __init__(self, name: str, shape: list[int], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    def __init__(self, path: str, providers=None) -> None:
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [FakeIo("obs", [1, 101])]

    def get_outputs(self):
        return [FakeIo("continuous_actions", [1, 14])]


class FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]

    InferenceSession = FakeSession


def test_check_policy_model_ok(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(checker, "ort", FakeOrt)

    result = checker.check_policy_model(model)

    assert result.ok is True
    assert result.input_name == "obs"
    assert result.input_shape == [1, 101]
    assert result.output_name == "continuous_actions"
    assert result.output_shape == [1, 14]


def test_check_policy_model_reports_shape_mismatch(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "policy.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(checker, "ort", FakeOrt)

    result = checker.check_policy_model(model, expected_input_shape=[1, 100])

    assert result.ok is False
    assert any("Input shape mismatch" in error for error in result.errors)


def test_check_policy_model_missing_file() -> None:
    result = checker.check_policy_model("/tmp/does-not-exist.onnx")

    assert result.ok is False
    assert "not found" in result.errors[0]
