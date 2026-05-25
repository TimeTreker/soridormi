from __future__ import annotations

import os

from soridormi_api.server import RobotApiServer

from .mujoco_backend import FakeMujocoBackend, MujocoBackend


def main() -> None:
    host = os.environ.get("SIM_HOST", "0.0.0.0")
    port = int(os.environ.get("SIM_PORT", "5555"))
    model_path = os.environ.get("MUJOCO_MODEL_PATH")

    if model_path:
        backend = MujocoBackend(model_path=model_path)
    else:
        backend = FakeMujocoBackend()

    RobotApiServer(backend=backend, host=host, port=port).serve_forever()


if __name__ == "__main__":
    main()
