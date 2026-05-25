from __future__ import annotations

import os
import time

from rich.console import Console

from .backends.hardware import HardwareRobot
from .backends.sim import SimRobot
from .controller import HoldPositionController
from .joint_sweep_controller import JointSweepController
from .standing_controller import StandingPoseController

console = Console()


def make_robot():
    backend = os.environ.get("SORIDORMI_BACKEND", "sim")

    if backend == "sim":
        host = os.environ.get("SIM_HOST", "127.0.0.1")
        port = int(os.environ.get("SIM_PORT", "5555"))
        return SimRobot(host=host, port=port)

    if backend == "hardware":
        return HardwareRobot()

    raise ValueError(f"Unknown SORIDORMI_BACKEND={backend!r}")


def make_controller():
    mode = os.environ.get("SORIDORMI_RUNTIME_MODE", "hold").strip().lower()

    if mode in {"hold", "hold_position"}:
        return HoldPositionController()

    if mode in {"stand", "standing", "default_pose"}:
        return StandingPoseController()

    if mode in {"joint_sweep", "sweep", "joint-test", "joint_test"}:
        return JointSweepController()

    raise ValueError(
        f"Unknown SORIDORMI_RUNTIME_MODE={mode!r}. "
        "Use one of: hold, stand, joint_sweep."
    )


def main() -> None:
    hz = float(os.environ.get("CONTROL_HZ", "50"))
    dt = 1.0 / hz

    robot = make_robot()
    controller = make_controller()

    console.print(f"[green]Soridormi runtime loop starting at {hz:.1f} Hz[/green]")
    console.print(f"Backend: {os.environ.get('SORIDORMI_BACKEND', 'sim')}")
    console.print(f"Runtime mode: {os.environ.get('SORIDORMI_RUNTIME_MODE', 'hold')}")

    while True:
        start = time.monotonic()

        state = robot.read_state()
        command = controller.compute(state)
        robot.send_motor_command(command)

        console.print(
            f"t={state.time:.3f} joints={len(state.joints.names)} ",
            end="\r",
        )

        elapsed = time.monotonic() - start
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
