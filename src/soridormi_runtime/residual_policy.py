from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from soridormi_api import RobotState
from soridormi_runtime.onnx_policy import OnnxPolicy
from soridormi_runtime.onnx_providers import resolve_onnx_providers, verify_active_providers
from soridormi_runtime.policy_profiles import DEFAULT_PROFILE_NAME, PolicyProfile


ACTION_SIZE = 14


def _load_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required for live residual ONNX policy inference; "
            "install the runtime or sim extra to load a real ONNX model"
        ) from exc
    return ort


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterable[None]:
    old_values: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _resolve_residual_path(path: str | os.PathLike[str] | None = None) -> Path:
    explicit = path or os.environ.get("SORIDORMI_POLICY_PATH")
    if not explicit:
        raise ValueError("Residual policy path is required")
    return Path(explicit)


def _action_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (1, ACTION_SIZE):
        arr = arr.reshape(ACTION_SIZE)
    if arr.shape != (ACTION_SIZE,):
        raise RuntimeError(f"{name} must have shape ({ACTION_SIZE},) or (1, {ACTION_SIZE}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} must contain only finite values")
    return arr


class ResidualOnnxPolicy:
    """Teacher ONNX policy plus a learned residual ONNX correction.

    This is the M6.19/M6.20 deployment form for residual fine-tuning. The
    residual model keeps the same Soridormi model IO contract, but its output is
    interpreted as a bounded correction, not a full action:

        final_action = teacher_action + residual_scale * clip(residual_output)

    The controller still sees one 14D action and uses the same action mapper,
    motor target history, and runtime logging path as the original policy.
    """

    ACTION_SIZE = ACTION_SIZE
    def __init__(
        self,
        policy_path: str | os.PathLike[str] | None = None,
        robot_config_path: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        session_factory: Any | None = None,
        teacher_policy: Any | None = None,
        teacher_profile: str | PolicyProfile | None = None,
        residual_scale: float | None = None,
        residual_clip_abs: float | None = None,
        final_action_clip_abs: float | None = None,
        prefer_cuda: bool | None = None,
    ) -> None:
        if prefer_cuda is None:
            prefer_cuda = os.environ.get("SORIDORMI_USE_CUDA_PROVIDER", "1").lower() not in {
                "0",
                "false",
                "no",
                "off",
            }

        self.policy_path = _resolve_residual_path(policy_path)
        self.residual_scale = float(residual_scale if residual_scale is not None else _env_float("SORIDORMI_RESIDUAL_SCALE", 0.05))
        self.residual_clip_abs = float(
            residual_clip_abs if residual_clip_abs is not None else _env_float("SORIDORMI_RESIDUAL_CLIP_ABS", 1.0)
        )
        final_clip = final_action_clip_abs if final_action_clip_abs is not None else _env_float(
            "SORIDORMI_RESIDUAL_FINAL_ACTION_CLIP_ABS",
            0.0,
        )
        self.final_action_clip_abs = float(final_clip) if final_clip and float(final_clip) > 0.0 else None
        self.input_name = os.environ.get("SORIDORMI_POLICY_INPUT_NAME", "obs")
        self.output_name = os.environ.get("SORIDORMI_POLICY_OUTPUT_NAME", "continuous_actions")

        if teacher_policy is not None:
            self.teacher = teacher_policy
            self.teacher_profile: PolicyProfile | None = None
            self.expected_input_size: int | None = None
        else:
            teacher_profile_name = teacher_profile or os.environ.get("SORIDORMI_RESIDUAL_TEACHER_PROFILE") or DEFAULT_PROFILE_NAME
            self.teacher_profile = teacher_profile_name if isinstance(teacher_profile_name, PolicyProfile) else PolicyProfile.load(teacher_profile_name)
            self.expected_input_size = _profile_input_size(self.teacher_profile)
            with _temporary_env(self.teacher_profile.env()):
                from soridormi_runtime.policy_factory import make_runtime_policy

                self.teacher = make_runtime_policy(
                    policy_path=self.teacher_profile.model.path,
                    robot_config_path=robot_config_path or os.environ.get("SORIDORMI_ROBOT_CONFIG"),
                )

        if providers is not None:
            self.providers = list(providers)
            requested_providers = list(providers)
            provider_selection_errors: list[str] = []
        else:
            selection = resolve_onnx_providers(_load_onnxruntime().get_available_providers(), prefer_cuda=prefer_cuda)
            self.providers = selection.providers
            requested_providers = selection.requested
            provider_selection_errors = list(selection.errors)
        if provider_selection_errors:
            raise RuntimeError("; ".join(provider_selection_errors))
        if not self.policy_path.exists() and session_factory is None:
            raise FileNotFoundError(f"Residual ONNX policy file not found: {self.policy_path}")
        if session_factory is None:
            self.session = _load_onnxruntime().InferenceSession(str(self.policy_path), providers=self.providers)
        else:
            self.session = session_factory(str(self.policy_path), providers=self.providers)
        active_providers = self._active_session_providers()
        provider_errors = verify_active_providers(active_providers, requested=requested_providers)
        if provider_errors:
            raise RuntimeError("; ".join(provider_errors))
        self.providers = active_providers or self.providers

        self.last_observation: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.last_teacher_action: np.ndarray | None = None
        self.last_residual_raw: np.ndarray | None = None
        self.last_residual_applied: np.ndarray | None = None
        self.last_debug: dict[str, Any] | None = None

    def _active_session_providers(self) -> list[str]:
        getter = getattr(self.session, "get_providers", None)
        if callable(getter):
            return [str(provider) for provider in getter()]
        return list(self.providers)

    def compute_action(self, state: RobotState) -> np.ndarray:
        teacher_action = _action_array(self.teacher.compute_action(state), "teacher_action")
        observation = self._teacher_observation()
        outputs = self.session.run([self.output_name], {self.input_name: observation})
        residual_raw = _action_array(outputs[0], "residual_action")
        residual_clipped = np.clip(residual_raw, -self.residual_clip_abs, self.residual_clip_abs)
        residual_applied = residual_clipped * self.residual_scale
        final_action = teacher_action + residual_applied
        if self.final_action_clip_abs is not None:
            final_action = np.clip(final_action, -self.final_action_clip_abs, self.final_action_clip_abs)

        self.last_observation = observation.copy()
        self.last_teacher_action = teacher_action.copy()
        self.last_residual_raw = residual_raw.copy()
        self.last_residual_applied = residual_applied.astype(np.float32).copy()
        self.last_action = final_action.astype(np.float32).copy()
        self.last_debug = {
            "policy_kind": "residual_onnx",
            "teacher_profile": None if self.teacher_profile is None else self.teacher_profile.name,
            "residual_scale": self.residual_scale,
            "residual_clip_abs": self.residual_clip_abs,
            "final_action_clip_abs": self.final_action_clip_abs,
            "teacher_action_abs_max": float(np.max(np.abs(teacher_action))),
            "residual_raw_abs_max": float(np.max(np.abs(residual_raw))),
            "residual_applied_abs_max": float(np.max(np.abs(residual_applied))),
            "final_action_abs_max": float(np.max(np.abs(final_action))),
        }
        return self.last_action

    def _teacher_observation(self) -> np.ndarray:
        observation = getattr(self.teacher, "last_observation", None)
        if observation is None:
            raise RuntimeError("Teacher policy did not expose last_observation after compute_action")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, obs.shape[0])
        if obs.ndim != 2 or obs.shape[0] != 1:
            raise RuntimeError(f"Teacher observation must have shape (1, N), got {obs.shape}")
        if self.expected_input_size is not None and obs.shape[1] != self.expected_input_size:
            raise RuntimeError(
                f"Teacher observation must have shape (1, {self.expected_input_size}), got {obs.shape}"
            )
        return obs

    def set_command_vector(self, command: list[float]) -> None:
        setter = getattr(self.teacher, "set_command_vector", None)
        if callable(setter):
            setter(command)

    def set_imitation_phase(self, imitation_phase: list[float]) -> None:
        setter = getattr(self.teacher, "set_imitation_phase", None)
        if callable(setter):
            setter(imitation_phase)

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        setter = getattr(self.teacher, "set_motor_targets", None)
        if callable(setter):
            setter(joint_names, positions)

    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        setter = getattr(self.teacher, "set_default_positions_by_name", None)
        if callable(setter):
            setter(positions_by_name)

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        bootstrap = getattr(self.teacher, "bootstrap_defaults_from_state", None)
        if callable(bootstrap):
            result = bootstrap(state)
            if isinstance(result, dict):
                return {str(k): float(v) for k, v in result.items()}
        return {}

    def reset_state(self) -> None:
        reset = getattr(self.teacher, "reset_state", None)
        if callable(reset):
            reset()


def _profile_input_size(profile: PolicyProfile) -> int:
    shape = list(profile.model.input_shape)
    if len(shape) != 2 or not isinstance(shape[-1], int) or int(shape[-1]) <= 0:
        raise ValueError(f"teacher profile input_shape must be [batch, positive_size], got {shape}")
    return int(shape[-1])
