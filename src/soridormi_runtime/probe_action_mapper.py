from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.action_mapper import PolicyActionMapper
from soridormi_runtime.onnx_policy import OnnxPolicy, resolve_policy_path


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


def main() -> None:
    policy_path = resolve_policy_path(os.environ.get("SORIDORMI_POLICY_PATH") or DEFAULT_POLICY_PATH)

    print("Soridormi action mapper probe")
    print("=============================")
    print(f"Policy path: {policy_path}")

    policy = OnnxPolicy(policy_path=policy_path)
    mapper = PolicyActionMapper.from_robot_config()

    state = make_dummy_state(policy.joint_names)
    action = policy.compute_action(state)
    command = mapper.action_to_command(action, state=state)

    # Keep the observation builder's motor target history in sync with the mapper.
    policy.observation_builder.set_motor_targets(command.names, command.positions)

    print()
    print("Policy")
    print("------")
    print(f"providers: {policy.providers}")
    print(f"action shape: {list(action.shape)}")
    print(f"action min/max: {float(np.min(action)):.6f} / {float(np.max(action)):.6f}")

    print()
    print("MotorCommand")
    print("------------")
    print(f"joint count: {len(command.names)}")
    print(f"position min/max: {min(command.positions):.6f} / {max(command.positions):.6f}")
    print(f"kp: {command.kp[0]:.3f} ...")
    print(f"kd: {command.kd[0]:.3f} ...")

    print()
    print("First commands")
    print("--------------")
    for name, position in list(zip(command.names, command.positions))[:14]:
        print(f"{name:24s} {position:+.6f}")

    print()
    print("Probe OK")


if __name__ == "__main__":
    main()
