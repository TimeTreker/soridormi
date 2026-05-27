from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from soridormi_runtime.linear_behavior_clone_policy import LinearBehaviorClonePolicy
from soridormi_runtime.onnx_policy import OnnxPolicy


ONNX_BACKENDS = {"", "onnx", "onnxruntime", "onnx_policy"}
LINEAR_BACKENDS = {"linear", "linear_npz", "linear_behavior_clone", "behavior_clone_linear"}


def normalize_policy_backend(value: str | None) -> str:
    text = (value or "onnx").strip().lower().replace("-", "_")
    if text in ONNX_BACKENDS:
        return "onnx"
    if text in LINEAR_BACKENDS:
        return "linear_behavior_clone"
    raise ValueError(
        f"Unknown SORIDORMI_POLICY_BACKEND={value!r}. Use one of: onnx, linear_behavior_clone."
    )


def policy_backend_from_env() -> str:
    return normalize_policy_backend(
        os.environ.get("SORIDORMI_POLICY_BACKEND") or os.environ.get("SORIDORMI_POLICY_KIND")
    )


def make_runtime_policy(
    *,
    policy_path: str | Path | None = None,
    robot_config_path: str | Path | None = None,
) -> Any:
    backend = policy_backend_from_env()
    if backend == "onnx":
        return OnnxPolicy(policy_path=policy_path, robot_config_path=robot_config_path)
    if backend == "linear_behavior_clone":
        return LinearBehaviorClonePolicy(policy_path=policy_path, robot_config_path=robot_config_path)
    raise AssertionError(f"unhandled policy backend: {backend}")
