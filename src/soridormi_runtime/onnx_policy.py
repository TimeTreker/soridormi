from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import onnxruntime as ort

from soridormi_api import RobotState
from soridormi_runtime.observation_builder import ObservationBuilder


DEFAULT_POLICY_PATH = Path("/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")


def choose_onnx_providers(prefer_cuda: bool = True) -> list[str]:
    """Choose ONNX Runtime providers for Soridormi policy inference."""
    available = ort.get_available_providers()
    providers: list[str] = []

    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError(f"No supported ONNX Runtime providers found: {available}")

    return providers


def resolve_policy_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_POLICY_PATH")
    return Path(explicit) if explicit else DEFAULT_POLICY_PATH


class OnnxPolicy:
    """Persistent ONNX policy wrapper.

    This class loads the ONNX Runtime session once, owns an ObservationBuilder,
    keeps policy action history, and returns one 14-dimensional action vector
    per RobotState.

    It intentionally does not convert actions to MotorCommand. That mapping is
    handled later by the M3.3 action-to-command layer.
    """

    ACTION_SIZE = 14
    OBS_BATCH_SHAPE = (1, 101)

    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        robot_config_path: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        observation_builder: ObservationBuilder | None = None,
        session_factory: Callable[..., Any] | None = None,
        prefer_cuda: bool | None = None,
    ) -> None:
        if prefer_cuda is None:
            prefer_cuda = os.environ.get("SORIDORMI_USE_CUDA_PROVIDER", "1").lower() not in {
                "0",
                "false",
                "no",
                "off",
            }

        self.policy_path = resolve_policy_path(policy_path)
        self.providers = list(providers) if providers is not None else choose_onnx_providers(prefer_cuda)
        self.observation_builder = observation_builder or ObservationBuilder.from_robot_config(
            path=robot_config_path
        )

        if session_factory is None:
            if not self.policy_path.exists():
                raise FileNotFoundError(f"ONNX policy file not found: {self.policy_path}")
            self.session = ort.InferenceSession(str(self.policy_path), providers=self.providers)
        else:
            self.session = session_factory(str(self.policy_path), providers=self.providers)

        self.input_name = self._single_input_name()
        self.output_name = self._first_output_name()

    def _single_input_name(self) -> str:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected exactly one ONNX input, got {len(inputs)}")
        return str(inputs[0].name)

    def _first_output_name(self) -> str:
        outputs = self.session.get_outputs()
        if not outputs:
            raise RuntimeError("ONNX policy has no outputs")
        return str(outputs[0].name)

    @property
    def joint_names(self) -> list[str]:
        return list(self.observation_builder.config.joint_names)

    def compute_action(self, state: RobotState) -> np.ndarray:
        obs = self.observation_builder.build_batch(state)

        if obs.shape != self.OBS_BATCH_SHAPE:
            raise RuntimeError(
                f"Policy observation must have shape {self.OBS_BATCH_SHAPE}, got {obs.shape}"
            )

        outputs = self.session.run([self.output_name], {self.input_name: obs})
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one ONNX output, got {len(outputs)}")

        action = np.asarray(outputs[0], dtype=np.float32)
        if action.shape == (1, self.ACTION_SIZE):
            action = action.reshape(self.ACTION_SIZE)

        if action.shape != (self.ACTION_SIZE,):
            raise RuntimeError(f"Policy action must have shape ({self.ACTION_SIZE},), got {action.shape}")

        self.observation_builder.update_action_history(action)
        return action

    def describe(self) -> dict[str, Any]:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        return {
            "policy_path": str(self.policy_path),
            "providers": list(self.providers),
            "input_name": self.input_name,
            "input_shape": list(inputs[0].shape),
            "output_name": self.output_name,
            "output_shape": list(outputs[0].shape),
            "joint_names": self.joint_names,
        }
