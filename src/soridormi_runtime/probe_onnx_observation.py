from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.observation_builder import ObservationBuilder


DEFAULT_POLICY_PATH = Path("/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")


def make_dummy_state(joint_names: list[str]) -> RobotState:
    n = len(joint_names)

    return RobotState(
        time=0.0,
        joints=JointState(
            names=joint_names,
            positions=[0.0] * n,
            velocities=[0.0] * n,
            torques=[0.0] * n,
        ),
        imu=IMUState(
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gyro_xyz=[0.0, 0.0, 0.0],
            accel_xyz=[0.0, 0.0, 9.81],
        ),
    )


def choose_providers() -> list[str]:
    available = ort.get_available_providers()

    providers: list[str] = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    return providers


def main() -> None:
    policy_path = Path(os.environ.get("SORIDORMI_POLICY_PATH", DEFAULT_POLICY_PATH))

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    print("Soridormi ONNX observation probe")
    print("================================")
    print(f"Policy path: {policy_path}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    providers = choose_providers()
    print(f"Selected providers: {providers}")

    session = ort.InferenceSession(str(policy_path), providers=providers)

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(f"Expected exactly one ONNX input, got {len(inputs)}")

    input_info = inputs[0]
    output_info = outputs[0]

    print()
    print("ONNX model")
    print("----------")
    print(f"Input:  name={input_info.name!r} shape={input_info.shape} type={input_info.type}")
    print(f"Output: name={output_info.name!r} shape={output_info.shape} type={output_info.type}")

    builder = ObservationBuilder.from_robot_config()
    state = make_dummy_state(builder.config.joint_names)

    obs = builder.build_batch(state)

    print()
    print("Observation")
    print("-----------")
    print(f"Observation shape: {list(obs.shape)}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min/max: {float(obs.min()):.6f} / {float(obs.max()):.6f}")

    if obs.shape != (1, 101):
        raise RuntimeError(f"Expected observation shape [1, 101], got {obs.shape}")

    result = session.run(None, {input_info.name: obs})

    print()
    print("Outputs")
    print("-------")

    for info, value in zip(outputs, result):
        arr = np.asarray(value)
        print(
            f"{info.name}: shape={list(arr.shape)} "
            f"dtype={arr.dtype} min={float(arr.min()):.6f} max={float(arr.max()):.6f}"
        )

    first_action = np.asarray(result[0], dtype=np.float32)

    if first_action.shape != (1, 14):
        raise RuntimeError(f"Expected action shape [1, 14], got {first_action.shape}")

    builder.update_action_history(first_action)

    obs2 = builder.build_batch(state)

    print()
    print("Action history")
    print("--------------")
    print(f"Updated second observation shape: {list(obs2.shape)}")
    print("Probe OK")


if __name__ == "__main__":
    main()
