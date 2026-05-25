from __future__ import annotations

import os
import time

from rich.console import Console

from .backends.hardware import HardwareRobot
from .backends.sim import SimRobot
from .controller import HoldPositionController

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


def main() -> None:
    hz = float(os.environ.get("CONTROL_HZ", "50"))
    dt = 1.0 / hz
    robot = make_robot()
    controller = HoldPositionController()

    console.print(f"[green]Soridormi runtime loop starting at {hz:.1f} Hz[/green]")
    console.print(f"Backend: {os.environ.get('SORIDORMI_BACKEND', 'sim')}")

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
