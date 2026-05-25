from __future__ import annotations

from pathlib import Path


class OnnxPolicy:
    """Thin ONNX Runtime wrapper placeholder.

    Keep policy loading isolated here. That makes it easier to swap ONNX Runtime for
    TensorRT on Jetson later without touching the controller API.
    """

    def __init__(self, policy_path: str | Path) -> None:
        self.policy_path = Path(policy_path)
        self.session = None

    def load(self) -> None:
        import onnxruntime as ort

        if not self.policy_path.exists():
            raise FileNotFoundError(self.policy_path)
        self.session = ort.InferenceSession(str(self.policy_path))

    def is_loaded(self) -> bool:
        return self.session is not None
