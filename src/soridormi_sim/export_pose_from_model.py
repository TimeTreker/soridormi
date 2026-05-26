from __future__ import annotations

import os
from pathlib import Path

import mujoco

from .robot_config import load_robot_config


def _mujoco_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    if name is None:
        raise RuntimeError(f"MuJoCo object {obj_type} id={obj_id} has no name")
    return name


def _actuator_joint_qpos(model: mujoco.MjModel, actuator_name: str) -> float:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"Actuator from config not found in model: {actuator_name}")

    joint_id = int(model.actuator_trnid[actuator_id][0])
    if joint_id < 0:
        raise ValueError(f"Actuator is not attached to a joint: {actuator_name}")

    qpos_addr = int(model.jnt_qposadr[joint_id])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return float(data.qpos[qpos_addr])


def main() -> None:
    config = load_robot_config()
    model_path = Path(os.environ.get("MUJOCO_MODEL_PATH") or config.model.path)

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    xyz_start, xyz_stop = config.base.qpos_xyz_slice
    quat_start, quat_stop = config.base.qpos_quat_wxyz_slice

    print("# Paste/merge this into configs/robots/open_duck_mini_v2.yaml")
    print("# Generated from:")
    print(f"#   {model_path}")
    print()
    print("reset_pose:")
    print("  base:")
    xyz = [float(x) for x in data.qpos[xyz_start:xyz_stop]]
    quat = [float(x) for x in data.qpos[quat_start:quat_stop]]
    print("    position_xyz: [" + ", ".join(f"{x:.6f}" for x in xyz) + "]")
    print("    quat_wxyz: [" + ", ".join(f"{x:.6f}" for x in quat) + "]")
    print("  joints:")

    qpos_by_actuator: dict[str, float] = {}
    for actuator_name in config.actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator from config not found in model: {actuator_name}")

        joint_id = int(model.actuator_trnid[actuator_id][0])
        joint_name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qpos = float(data.qpos[qpos_addr])
        qpos_by_actuator[actuator_name] = qpos
        print(f"    {joint_name}: {qpos:.6f}  # actuator={actuator_name}")

    print()
    print("default_pose:")
    print("  positions:")
    for actuator_name in config.actuator_names:
        print(f"    {actuator_name}: {qpos_by_actuator[actuator_name]:.6f}")

    print()
    print("  gains:")
    print("    kp_default: 10.0")
    print("    kd_default: 0.5")
    print()
    print("  torque_default: 0.0")


if __name__ == "__main__":
    main()
