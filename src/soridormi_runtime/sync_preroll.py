from __future__ import annotations

from soridormi_api import MotorCommand, RobotState


def make_hold_current_command(state: RobotState) -> MotorCommand:
    """Build a position command that holds the backend's current actuator controls.

    The official Open Duck loop applies the MuJoCo home ctrl and then advances
    one policy-control interval before its first ONNX observation. During that
    interval the controls remain at the home targets. For Soridormi sync-step
    parity, the safest generic equivalent is to send a hold-current command
    using ``state.actuator_ctrl`` when the simulator exposes it, falling back to
    current joint positions on older/non-MuJoCo backends.
    """

    names = list(state.joints.names)
    if state.actuator_ctrl is not None and len(state.actuator_ctrl) == len(names):
        positions = [float(x) for x in state.actuator_ctrl]
    else:
        positions = [float(x) for x in state.joints.positions]

    n = len(names)
    return MotorCommand(
        names=names,
        positions=positions,
        velocities=[0.0] * n,
        kp=[10.0] * n,
        kd=[0.5] * n,
        torques=[0.0] * n,
    )


def preroll_sync_simulator(robot: object, state: RobotState, steps: int) -> RobotState:
    """Advance a synchronous simulator before the first policy observation.

    The robot object must expose ``step_motor_command(command) -> RobotState``.
    This helper intentionally has no ONNX/MuJoCo imports so it can be unit-tested
    in lightweight host environments.
    """

    if steps <= 0:
        return state

    stepper = getattr(robot, "step_motor_command", None)
    if not callable(stepper):
        raise RuntimeError("SORIDORMI_SIM_PREROLL_STEPS requires a sync-step simulator backend")

    current = state
    for _ in range(steps):
        current = stepper(make_hold_current_command(current))
    return current
