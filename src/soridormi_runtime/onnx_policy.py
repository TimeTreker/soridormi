from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from soridormi_api import RobotState
from soridormi_runtime.observation_builder import ObservationBuilder
from soridormi_runtime.onnx_providers import resolve_onnx_providers, verify_active_providers
from soridormi_runtime.policy_input_features import (
    INPUT_MODE_OBSERVATION,
    build_policy_input_batch,
    input_size_for,
    normalize_policy_input_mode,
)


DEFAULT_POLICY_PATH = Path("/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required for live ONNX policy inference; "
            "install the runtime or sim extra to load a real ONNX model"
        ) from exc
    return ort


def choose_onnx_providers(prefer_cuda: bool = True) -> list[str]:
    """Choose ONNX Runtime providers for Soridormi policy inference."""
    selection = resolve_onnx_providers(_load_onnxruntime().get_available_providers(), prefer_cuda=prefer_cuda)
    if not selection.ok:
        raise RuntimeError("; ".join(selection.errors))
    return selection.providers


def resolve_policy_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_POLICY_PATH")
    return Path(explicit) if explicit else DEFAULT_POLICY_PATH


def _array_stats(name: str, values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "l2_norm": float(np.linalg.norm(arr)) if arr.size else 0.0,
    }


class OnnxPolicy:
    """Persistent ONNX policy wrapper.

    This class loads the ONNX Runtime session once, owns an ObservationBuilder,
    keeps policy action history, and returns one 14-dimensional action vector
    per RobotState.
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
        if providers is not None:
            self.providers = list(providers)
            provider_selection_errors: list[str] = []
            requested_providers = list(providers)
        else:
            provider_selection = resolve_onnx_providers(_load_onnxruntime().get_available_providers(), prefer_cuda=prefer_cuda)
            self.providers = provider_selection.providers
            provider_selection_errors = list(provider_selection.errors)
            requested_providers = provider_selection.requested
        if provider_selection_errors:
            raise RuntimeError("; ".join(provider_selection_errors))
        self.observation_builder = observation_builder or ObservationBuilder.from_robot_config(
            path=robot_config_path
        )
        self.expected_input_name = os.environ.get("SORIDORMI_POLICY_INPUT_NAME") or None
        self.expected_output_name = os.environ.get("SORIDORMI_POLICY_OUTPUT_NAME") or None
        self.input_mode = normalize_policy_input_mode(os.environ.get("SORIDORMI_POLICY_INPUT_MODE"))
        self.last_command_vector: list[float] = [0.0] * 7

        if session_factory is None:
            if not self.policy_path.exists():
                raise FileNotFoundError(f"ONNX policy file not found: {self.policy_path}")
            self.session = _load_onnxruntime().InferenceSession(str(self.policy_path), providers=self.providers)
        else:
            self.session = session_factory(str(self.policy_path), providers=self.providers)

        active_providers = self._active_session_providers()
        provider_errors = verify_active_providers(
            active_providers,
            requested=requested_providers,
        )
        if provider_errors:
            raise RuntimeError("; ".join(provider_errors))
        self.providers = active_providers or self.providers

        self.input_name = self._select_input_name(self.expected_input_name)
        self.output_name = self._select_output_name(self.expected_output_name)
        self._validate_io_contract_from_env()
        self.policy_input_size = self._policy_input_size_from_session_or_env()
        self.last_observation: np.ndarray | None = None
        self.last_observation_stats: dict[str, object] | None = None
        self.last_action: np.ndarray | None = None
        self.last_action_stats: dict[str, object] | None = None

    def _active_session_providers(self) -> list[str]:
        getter = getattr(self.session, "get_providers", None)
        if callable(getter):
            return [str(provider) for provider in getter()]
        return list(self.providers)

    def _select_input_name(self, expected_name: str | None = None) -> str:
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX policy has no inputs")
        if expected_name:
            for item in inputs:
                if str(item.name) == expected_name:
                    return expected_name
            available = [str(item.name) for item in inputs]
            raise RuntimeError(f"Expected ONNX input {expected_name!r}, available inputs={available}")
        if len(inputs) != 1:
            raise RuntimeError(f"Expected exactly one ONNX input, got {len(inputs)}")
        return str(inputs[0].name)

    def _select_output_name(self, expected_name: str | None = None) -> str:
        outputs = self.session.get_outputs()
        if not outputs:
            raise RuntimeError("ONNX policy has no outputs")
        if expected_name:
            for item in outputs:
                if str(item.name) == expected_name:
                    return expected_name
            available = [str(item.name) for item in outputs]
            raise RuntimeError(f"Expected ONNX output {expected_name!r}, available outputs={available}")
        return str(outputs[0].name)

    def _validate_io_contract_from_env(self) -> None:
        expected_input_shape = _parse_shape_env("SORIDORMI_POLICY_EXPECTED_INPUT_SHAPE")
        expected_output_shape = _parse_shape_env("SORIDORMI_POLICY_EXPECTED_OUTPUT_SHAPE")
        expected_input_type = os.environ.get("SORIDORMI_POLICY_EXPECTED_INPUT_TYPE") or None
        expected_output_type = os.environ.get("SORIDORMI_POLICY_EXPECTED_OUTPUT_TYPE") or None
        inputs = {str(item.name): item for item in self.session.get_inputs()}
        outputs = {str(item.name): item for item in self.session.get_outputs()}
        input_info = inputs[self.input_name]
        output_info = outputs[self.output_name]
        if expected_input_shape and not _shape_matches(list(input_info.shape), expected_input_shape):
            raise RuntimeError(
                f"Policy input shape mismatch for {self.input_name!r}: expected {expected_input_shape}, got {list(input_info.shape)}"
            )
        if expected_output_shape and not _shape_matches(list(output_info.shape), expected_output_shape):
            raise RuntimeError(
                f"Policy output shape mismatch for {self.output_name!r}: expected {expected_output_shape}, got {list(output_info.shape)}"
            )
        if expected_input_type and str(input_info.type) != expected_input_type:
            raise RuntimeError(f"Policy input type mismatch for {self.input_name!r}: expected {expected_input_type}, got {input_info.type}")
        if expected_output_type and str(output_info.type) != expected_output_type:
            raise RuntimeError(f"Policy output type mismatch for {self.output_name!r}: expected {expected_output_type}, got {output_info.type}")

    @property
    def joint_names(self) -> list[str]:
        return list(self.observation_builder.config.joint_names)

    def set_command_vector(self, command: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.last_command_vector = [float(x) for x in np.asarray(command, dtype=np.float32).reshape(-1).tolist()]
        self.observation_builder.set_command(command)

    def set_imitation_phase(self, imitation_phase: list[float] | tuple[float, ...] | np.ndarray) -> None:
        self.observation_builder.set_imitation_phase(imitation_phase)

    def set_motor_targets_by_name(self, targets_by_name: dict[str, float]) -> None:
        self.observation_builder.set_motor_targets_by_name(targets_by_name)

    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        setter = getattr(self.observation_builder, "set_default_positions_by_name", None)
        if callable(setter):
            setter(positions_by_name)
        else:
            self.observation_builder.config.default_positions_by_name.update(
                {str(name): float(value) for name, value in positions_by_name.items()}
            )

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        """Bootstrap default_actuator/motor_targets from backend metadata.

        In MuJoCo official-compatibility mode the simulator exposes data.ctrl
        from model.keyframe("home").ctrl as state.actuator_ctrl. That is the
        closest match to Open Duck's `default_actuator`. Fall back to joint qpos
        for non-MuJoCo or older backends.
        """
        source_values = state.actuator_ctrl
        if source_values is not None and len(source_values) == len(state.joints.names):
            values = source_values
        else:
            values = state.joints.positions

        defaults = {
            str(name): float(value)
            for name, value in zip(state.joints.names, values)
            if name in self.joint_names
        }
        self.set_default_positions_by_name(defaults)
        self.set_motor_targets_by_name(defaults)
        return defaults

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        self.observation_builder.set_motor_targets(joint_names, positions)

    def reset_state(self) -> None:
        resetter = getattr(self.observation_builder, "reset_action_history", None)
        if callable(resetter):
            resetter()
        self.last_observation = None
        self.last_observation_stats = None
        self.last_action = None
        self.last_action_stats = None

    def compute_action(self, state: RobotState) -> np.ndarray:
        robot_observation = self.observation_builder.build_batch(state)

        if robot_observation.shape != self.OBS_BATCH_SHAPE:
            raise RuntimeError(
                f"Policy robot observation must have shape {self.OBS_BATCH_SHAPE}, got {robot_observation.shape}"
            )
        obs = build_policy_input_batch(
            robot_observation,
            input_mode=self.input_mode,
            command_vector=self.last_command_vector,
        )
        expected_shape = (1, self.policy_input_size)
        if obs.shape != expected_shape:
            raise RuntimeError(f"Policy input must have shape {expected_shape}, got {obs.shape}")

        self.last_observation = obs.copy()
        self.last_observation_stats = _array_stats("observation", obs)

        outputs = self.session.run([self.output_name], {self.input_name: obs})
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one ONNX output, got {len(outputs)}")

        action = np.asarray(outputs[0], dtype=np.float32)
        if action.shape == (1, self.ACTION_SIZE):
            action = action.reshape(self.ACTION_SIZE)

        if action.shape != (self.ACTION_SIZE,):
            raise RuntimeError(f"Policy action must have shape ({self.ACTION_SIZE},), got {action.shape}")

        self.last_action = action.copy()
        self.last_action_stats = _array_stats("action", action)
        self.observation_builder.update_action_history(action)
        return action

    def get_observation(self) -> list[float] | None:
        if self.last_observation is None:
            return None
        return [float(x) for x in np.asarray(self.last_observation, dtype=np.float32).reshape(-1)]

    def get_observation_stats(self) -> dict[str, object] | None:
        return None if self.last_observation_stats is None else dict(self.last_observation_stats)

    def get_action_stats(self) -> dict[str, object] | None:
        return None if self.last_action_stats is None else dict(self.last_action_stats)

    def describe(self) -> dict[str, Any]:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        return {
            "policy_path": str(self.policy_path),
            "providers": list(self.providers),
            "input_name": self.input_name,
            "input_shape": _shape_for_name(inputs, self.input_name),
            "input_mode": self.input_mode,
            "policy_input_size": self.policy_input_size,
            "output_name": self.output_name,
            "output_shape": _shape_for_name(outputs, self.output_name),
            "available_inputs": [str(item.name) for item in inputs],
            "available_outputs": [str(item.name) for item in outputs],
            "joint_names": self.joint_names,
        }

    def _policy_input_size_from_session_or_env(self) -> int:
        expected_shape = _parse_shape_env("SORIDORMI_POLICY_EXPECTED_INPUT_SHAPE")
        if expected_shape:
            last = expected_shape[-1]
            if isinstance(last, int):
                return int(last)
        shape = _shape_for_name(self.session.get_inputs(), self.input_name)
        if shape:
            last = shape[-1]
            try:
                return int(last)
            except (TypeError, ValueError):
                pass
        return input_size_for(self.input_mode, robot_observation_size=self.OBS_BATCH_SHAPE[1])


def _parse_shape_env(name: str) -> list[object] | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    out: list[object] = []
    for raw in value.split(","):
        item = raw.strip()
        if item in {"", "?", "None", "none", "null", "-1"}:
            out.append(None)
        else:
            try:
                out.append(int(item))
            except ValueError:
                out.append(item)
    return out


def _shape_matches(actual: list[object], expected: list[object]) -> bool:
    if len(actual) != len(expected):
        return False

    for index, (a, e) in enumerate(zip(actual, expected)):
        # Expected None means caller explicitly allows any dimension.
        if e is None:
            continue

        # ONNX Runtime may report dynamic dimensions as None, "?", "", or a
        # symbolic name such as "batch". Soridormi always feeds a batch of 1 at
        # runtime, so a symbolic first dimension is compatible with expected 1.
        if a in {None, "", "?"}:
            continue

        if index == 0 and isinstance(a, str):
            stripped = a.strip()
            if stripped and not stripped.lstrip("-").isdigit():
                continue

        if str(a) != str(e):
            return False

    return True


def _shape_for_name(items: list[object], name: str) -> list[object] | None:
    for item in items:
        if str(getattr(item, "name", "")) == name:
            return list(getattr(item, "shape", []) or [])
    return None
