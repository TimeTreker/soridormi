from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.onnx_policy import DEFAULT_POLICY_PATH, OnnxPolicy


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


def main() -> None:
    policy_path = Path(os.environ.get("SORIDORMI_POLICY_PATH", DEFAULT_POLICY_PATH))

    print("Soridormi ONNX policy wrapper probe")
    print("====================================")
    print(f"Policy path: {policy_path}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    policy = OnnxPolicy(policy_path=policy_path)
    info = policy.describe()

    print()
    print("Policy session")
    print("--------------")
    print(f"Selected providers: {info['providers']}")
    print(f"Input:  {info['input_name']} shape={info['input_shape']}")
    print(f"Output: {info['output_name']} shape={info['output_shape']}")
    print(f"Joints: {len(info['joint_names'])}")

    state = make_dummy_state(policy.joint_names)
    action = policy.compute_action(state)

    print()
    print("Action")
    print("------")
    print(f"shape: {list(action.shape)}")
    print(f"dtype: {action.dtype}")
    print(f"min/max: {float(np.min(action)):.6f} / {float(np.max(action)):.6f}")

    second_action = policy.compute_action(state)
    print(f"second_action_shape: {list(second_action.shape)}")
    print("Probe OK")


if __name__ == "__main__":
    main()
