from __future__ import annotations

import os

from soridormi_api.server import RobotApiServer

from .mujoco_backend import FakeMujocoBackend, MujocoBackend
from .robot_config import load_robot_config


def main() -> None:
    host = os.environ.get("SIM_HOST", "0.0.0.0")
    port = int(os.environ.get("SIM_PORT", "5555"))
    use_real_mujoco = os.environ.get("SORIDORMI_SIM_BACKEND", "fake") == "mujoco"
    config_path = os.environ.get("SORIDORMI_ROBOT_CONFIG")
    model_path = os.environ.get("MUJOCO_MODEL_PATH")

    if use_real_mujoco or model_path:
        backend = MujocoBackend(config_path=config_path, model_path=model_path)
    else:
        backend = FakeMujocoBackend(config=load_robot_config(config_path))

    RobotApiServer(backend=backend, host=host, port=port).serve_forever()


if __name__ == "__main__":
    main()
