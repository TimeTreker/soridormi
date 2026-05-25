from __future__ import annotations

import os
from pathlib import Path

import mujoco


DEFAULT_MODEL_PATH = (
    "/workspaces/Open_Duck_Playground/"
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
)


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"<unnamed:{obj_id}>"


def main() -> None:
    model_path = Path(os.environ.get("SORIDORMI_MJCF_PATH", DEFAULT_MODEL_PATH))

    if not model_path.exists():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {model_path}\n"
            "Check that Open_Duck_Playground is mounted and submodules are initialized."
        )

    print(f"Loading MuJoCo model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print()
    print("Model summary")
    print("-------------")
    print(f"nq:       {model.nq}")
    print(f"nv:       {model.nv}")
    print(f"nu:       {model.nu}")
    print(f"njnt:     {model.njnt}")
    print(f"nbody:    {model.nbody}")
    print(f"ngeom:    {model.ngeom}")
    print(f"timestep: {model.opt.timestep}")

    print()
    print("Joints")
    print("------")
    for i in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = int(model.jnt_type[i])
        qpos_addr = int(model.jnt_qposadr[i])
        dof_addr = int(model.jnt_dofadr[i])
        print(
            f"{i:02d}  name={name:32s} "
            f"type={joint_type} qpos_addr={qpos_addr} dof_addr={dof_addr}"
        )

    print()
    print("Actuators")
    print("---------")
    for i in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        ctrlrange = model.actuator_ctrlrange[i]
        print(
            f"{i:02d}  name={name:32s} "
            f"trnid={trnid.tolist()} ctrlrange={ctrlrange.tolist()}"
        )

    print()
    print("Bodies")
    print("------")
    for i in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:02d}  name={name}")


if __name__ == "__main__":
    main()
