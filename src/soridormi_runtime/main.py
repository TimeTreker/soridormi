from __future__ import annotations

import os
import time

from rich.console import Console

from .backends.hardware import HardwareRobot
from .backends.sim import SimRobot
from .controller import HoldPositionController
from .joint_sweep_controller import JointSweepController
from .logging import make_runtime_logger_from_env
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

    backend = os.environ.get("SORIDORMI_BACKEND", "sim")
    mode = os.environ.get("SORIDORMI_RUNTIME_MODE", "hold")

    robot = make_robot()
    controller = make_controller()
    runtime_logger = make_runtime_logger_from_env(mode=mode, backend=backend)

    console.print(f"[green]Soridormi runtime loop starting at {hz:.1f} Hz[/green]")
    console.print(f"Backend: {backend}")
    console.print(f"Runtime mode: {mode}")
    if runtime_logger.path is not None:
        console.print(f"Runtime log: {runtime_logger.path}")

    step_index = 0

    try:
        while True:
            start = time.monotonic()

            state = robot.read_state()
            command = controller.compute(state)
            robot.send_motor_command(command)
            runtime_logger.log_step(
                step_index=step_index,
                state=state,
                command=command,
                mode=mode,
                backend=backend,
            )

            console.print(
                f"t={state.time:.3f} joints={len(state.joints.names)} ",
                end="\r",
            )

            step_index += 1
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        console.print("\n[yellow]Runtime loop stopped by user.[/yellow]")
    finally:
        runtime_logger.close()
        if runtime_logger.path is not None:
            console.print(f"Closed runtime log: {runtime_logger.path}")


if __name__ == "__main__":
    main()
