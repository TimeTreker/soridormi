from __future__ import annotations

import os
from pathlib import Path


class OnnxPolicy:
    """Thin ONNX Runtime wrapper.

    On PC simulation/dev, this prefers CUDAExecutionProvider when the installed
    ONNX Runtime package exposes it. On Jetson, keep policy loading isolated here
    so TensorRT or a Jetson-specific execution provider can be added later without
    changing the controller API.
    """

    def __init__(self, policy_path: str | Path) -> None:
        self.policy_path = Path(policy_path)
        self.session = None
        self.providers: list[str] = []

    def load(self) -> None:
        import onnxruntime as ort

        if not self.policy_path.exists():
            raise FileNotFoundError(self.policy_path)

        available = ort.get_available_providers()
        prefer_cuda = os.environ.get("SORIDORMI_USE_CUDA_PROVIDER", "1") == "1"
        preferred = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if prefer_cuda
            else ["CPUExecutionProvider"]
        )
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            providers = available

        self.session = ort.InferenceSession(str(self.policy_path), providers=providers)
        self.providers = list(self.session.get_providers())

    def is_loaded(self) -> bool:
        return self.session is not None
