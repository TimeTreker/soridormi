from __future__ import annotations

import os
import time
from typing import Any

from rich.console import Console

from .backends.hardware import HardwareRobot
from .backends.sim import SimRobot
from .controller import HoldPositionController
from .joint_sweep_controller import JointSweepController
from .logging import make_runtime_logger_from_env
from .onnx_policy_controller import OnnxPolicyController
from .standing_controller import StandingPoseController
from .runtime_limits import runtime_limit_reached, runtime_limits_from_env

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

    if mode in {"onnx_policy", "policy", "walk_policy", "walking_policy"}:
        return OnnxPolicyController()

    raise ValueError(
        f"Unknown SORIDORMI_RUNTIME_MODE={mode!r}. "
        "Use one of: hold, stand, joint_sweep, onnx_policy."
    )


def _controller_policy_log_payload(controller: object) -> dict[str, Any]:
    getter = getattr(controller, "get_policy_log_payload", None)
    if not callable(getter):
        return {}

    payload = getter()
    if not isinstance(payload, dict):
        return {}
    return payload


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _step_command_and_get_state(robot: object, command: object):
    stepper = getattr(robot, "step_motor_command", None)
    if callable(stepper):
        return stepper(command)

    sender = getattr(robot, "send_motor_command")
    reader = getattr(robot, "read_state")
    sender(command)
    return reader()


def main() -> None:
    hz = float(os.environ.get("CONTROL_HZ", "50"))
    dt = 1.0 / hz

    backend = os.environ.get("SORIDORMI_BACKEND", "sim")
    mode = os.environ.get("SORIDORMI_RUNTIME_MODE", "hold")
    sync_step = _env_bool("SORIDORMI_SIM_SYNC_STEP", default=False)
    limits = runtime_limits_from_env()

    robot = make_robot()
    if _env_bool("SORIDORMI_RESET_AT_START", default=False):
        resetter = getattr(robot, "reset", None)
        if callable(resetter):
            message = resetter()
            console.print(f"[cyan]Requested backend reset at experiment start: {message}[/cyan]")
        else:
            console.print("[yellow]SORIDORMI_RESET_AT_START=1 but backend has no reset method.[/yellow]")
    controller = make_controller()
    runtime_logger = make_runtime_logger_from_env(mode=mode, backend=backend)

    console.print(f"[green]Soridormi runtime loop starting at {hz:.1f} Hz[/green]")
    console.print(f"Backend: {backend}")
    console.print(f"Runtime mode: {mode}")
    if hasattr(controller, "describe"):
        console.print(f"Controller: {controller.describe()}")
    if runtime_logger.path is not None:
        console.print(f"Runtime log: {runtime_logger.path}")
    if sync_step:
        console.print("Simulator synchronous step mode: enabled")

    step_index = 0
    loop_started_at = time.monotonic()
    state = robot.read_state() if sync_step else None
    if limits.max_steps is not None:
        console.print(f"Runtime max steps: {limits.max_steps}")
    if limits.max_seconds is not None:
        console.print(f"Runtime max seconds: {limits.max_seconds:.3f}")

    try:
        while True:
            start = time.monotonic()

            if sync_step:
                assert state is not None
                command = controller.compute(state)
                policy_log_payload = _controller_policy_log_payload(controller)
                next_state = _step_command_and_get_state(robot, command)
            else:
                state = robot.read_state()
                command = controller.compute(state)
                robot.send_motor_command(command)
                policy_log_payload = _controller_policy_log_payload(controller)
                next_state = None

            runtime_logger.log_step(
                step_index=step_index,
                state=state,
                command=command,
                mode=mode,
                backend=backend,
                policy_raw_action=policy_log_payload.get("policy_raw_action"),
                policy_action=policy_log_payload.get("policy_action"),
                policy_observation=policy_log_payload.get("policy_observation"),
                policy_debug=policy_log_payload.get("policy_debug"),
                policy_observation_stats=policy_log_payload.get("policy_observation_stats"),
            )

            console.print(
                f"t={state.time:.3f} joints={len(state.joints.names)} ",
                end="\r",
            )

            if sync_step:
                state = next_state

            step_index += 1
            now = time.monotonic()
            if runtime_limit_reached(completed_steps=step_index, started_at=loop_started_at, now=now, limits=limits):
                console.print(f"\n[green]Runtime limit reached after {step_index} steps.[/green]")
                break
            elapsed = now - start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        console.print("\n[yellow]Runtime loop stopped by user.[/yellow]")
    finally:
        runtime_logger.close()
        if runtime_logger.path is not None:
            console.print(f"Closed runtime log: {runtime_logger.path}")


if __name__ == "__main__":
    main()
