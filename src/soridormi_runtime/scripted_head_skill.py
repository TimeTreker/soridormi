"""Execute safe scripted head/neck social skills against a MuJoCo simulator.

This module is intentionally narrow. It executes manifest-backed scripted
keyframe plans such as ``neutral_head``, ``look_direction``, and ``express_attention`` by holding all non-head joints at the
simulator-reported positions while smoothly moving only the declared head/neck
actuators. Hardware execution is not exposed here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from soridormi_api.types import MotorCommand, RobotState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soridormi_api.client import RobotApiClient

from .skill_execution import SkillExecutionError, SkillExecutionRegistry, SkillPlan, _load_json_args
from .skill_manifest import DEFAULT_SKILL_MANIFEST


HEAD_JOINT_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
SUPPORTED_SCRIPTED_SKILLS = {"neutral_head", "look_direction", "nod_yes", "shake_no", "bow", "express_attention"}
NEUTRAL_HOME_GESTURE_SKILLS = {"nod_yes", "shake_no", "bow", "express_attention"}
MOVING_HEAD_JOINTS_BY_SKILL: dict[str, set[str]] = {
    "nod_yes": {"head_pitch"},
    "shake_no": {"head_yaw"},
    "bow": {"neck_pitch", "head_pitch"},
    "express_attention": {"head_pitch", "head_yaw"},
}
# Defaults are intentionally gentle for viewer validation. These scripted
# social skills are pose trajectories, not twitch tests; callers can still
# override both values from the CLI for faster debugging.
DEFAULT_TRANSITION_FRACTION = 0.40
DEFAULT_MAX_HEAD_VELOCITY_RADPS = 0.35


@dataclass(frozen=True)
class ScriptedHeadExecutionResult:
    skill_id: str
    backend: str
    executed: bool
    steps: int
    control_hz: float
    duration_s: float
    requested_duration_s: float
    effective_duration_s: float
    target_positions_by_name: dict[str, float]
    start_positions_by_name: dict[str, float]
    final_positions_by_name: dict[str, float]
    target_min_positions_by_name: dict[str, float]
    target_max_positions_by_name: dict[str, float]
    observed_min_positions_by_name: dict[str, float]
    observed_max_positions_by_name: dict[str, float]
    keyframe_targets: list[dict[str, Any]]
    transition_fraction: float
    max_head_velocity_radps: float | None
    auto_stretched_duration: bool
    observed_min_base_height_m: float | None = None
    observed_max_base_height_m: float | None = None
    final_base_height_m: float | None = None
    fall_height_m: float | None = None
    fallen: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoothstep(alpha: float) -> float:
    clamped = max(0.0, min(1.0, float(alpha)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def joint_positions_by_name(state: RobotState) -> dict[str, float]:
    return {name: float(position) for name, position in zip(state.joints.names, state.joints.positions)}


def command_positions_by_name(state: RobotState) -> dict[str, float]:
    """Return actuator control targets currently held by the backend.

    The MuJoCo API exposes both joint positions (qpos) and actuator controls
    (data.ctrl). A scripted head gesture should stream a head pose trajectory
    while leaving the rest of the robot on the stable actuator controls already
    holding the body. Preserving non-head joints from qpos can retarget the legs
    to a transient physical pose and make the robot fall even though only the
    head was meant to move.
    """

    names = list(state.joints.names)
    if state.actuator_ctrl is not None and len(state.actuator_ctrl) == len(names):
        return {name: float(value) for name, value in zip(names, state.actuator_ctrl)}
    return joint_positions_by_name(state)


def _head_subset(positions_by_name: Mapping[str, float]) -> dict[str, float]:
    return {name: float(positions_by_name.get(name, 0.0)) for name in HEAD_JOINT_NAMES}


def _base_height_range(states: Sequence[RobotState]) -> tuple[float | None, float | None, float | None]:
    heights = [
        float(state.base_position_xyz[2])
        for state in states
        if state.base_position_xyz is not None and len(state.base_position_xyz) == 3
    ]
    if not heights:
        return None, None, None
    return min(heights), max(heights), heights[-1]


def _head_ranges(history: Sequence[Mapping[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    if not history:
        empty = {name: 0.0 for name in HEAD_JOINT_NAMES}
        return empty, dict(empty)
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for name in HEAD_JOINT_NAMES:
        values = [float(sample.get(name, 0.0)) for sample in history]
        mins[name] = min(values)
        maxs[name] = max(values)
    return mins, maxs


def _keyframe_targets_for_report(plan: SkillPlan, targets: Sequence[Mapping[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (keyframe, target) in enumerate(zip(plan.keyframes, targets)):
        rows.append(
            {
                "index": index,
                "label": keyframe.label,
                "duration_s": float(keyframe.duration_s),
                "positions_by_name": _head_subset(target),
            }
        )
    return rows


def interpolate_positions(
    start_positions_by_name: Mapping[str, float],
    target_positions_by_name: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    eased = smoothstep(alpha)
    result = dict(start_positions_by_name)
    for name, target in target_positions_by_name.items():
        start = float(start_positions_by_name.get(name, 0.0))
        result[name] = start + (float(target) - start) * eased
    return result


def target_positions_for_segment_step(
    start_positions_by_name: Mapping[str, float],
    target_positions_by_name: Mapping[str, float],
    *,
    step_index: int,
    segment_steps: int,
    transition_fraction: float = DEFAULT_TRANSITION_FRACTION,
    strict_target_names: set[str] | None = None,
) -> dict[str, float]:
    """Return the command target for one step in a keyframe segment.

    Earlier M8F code ramped for the entire keyframe duration. For a gesture
    like ``shake_no`` this means the command reached the left/right extreme
    only on the last step, then immediately reversed. The sim's position
    actuator often never visibly reached the yaw/pitch extreme. This helper
    ramps only for the first fraction of a segment and holds the target for the
    remaining steps, so each keyframe is a visible gesture pose instead of a
    purely transitional waypoint.
    """

    if segment_steps <= 0:
        raise SkillExecutionError("segment_steps must be positive")
    if not 0.0 <= transition_fraction <= 1.0:
        raise SkillExecutionError("--transition-fraction must be between 0.0 and 1.0")

    strict_names = strict_target_names or set()

    if transition_fraction == 0.0 or segment_steps == 1:
        result = dict(target_positions_by_name)
    else:
        transition_steps = max(1, int(math.ceil(float(segment_steps) * float(transition_fraction))))
        if step_index + 1 >= transition_steps:
            result = dict(target_positions_by_name)
        else:
            alpha = float(step_index + 1) / float(transition_steps)
            result = interpolate_positions(start_positions_by_name, target_positions_by_name, alpha)

    # Social gestures are axis-specific: shake_no is yaw-only, nod_yes is
    # pitch-only, and bow is shallow neck/head pitch only. Do not let simulator
    # drift from a previous segment get blended back through non-moving head
    # joints; command those joints directly to their home target every step.
    # This keeps gestures legible and makes final neutral truly neutral.
    for name in strict_names:
        if name in target_positions_by_name:
            result[name] = float(target_positions_by_name[name])
    return result


def _gesture_home_positions(plan: SkillPlan, reference_positions_by_name: Mapping[str, float]) -> dict[str, float]:
    if plan.skill_id in NEUTRAL_HOME_GESTURE_SKILLS:
        # Social gestures should be legible from a straight head pose: start at
        # neutral, move only the gesture axis, and return to neutral. Using the
        # current simulator pose as an offset can preserve a small downward head
        # drift and make shake_no look like "head down" instead of left/right.
        return {name: 0.0 for name in HEAD_JOINT_NAMES}
    return {name: float(reference_positions_by_name.get(name, 0.0)) for name in HEAD_JOINT_NAMES}


def strict_head_target_names_for_plan(plan: SkillPlan) -> set[str]:
    moving = MOVING_HEAD_JOINTS_BY_SKILL.get(plan.skill_id)
    if moving is None:
        return set()
    return set(HEAD_JOINT_NAMES) - set(moving)


def resolve_keyframe_targets_for_execution(
    plan: SkillPlan,
    reference_positions_by_name: Mapping[str, float],
) -> list[dict[str, float]]:
    """Resolve dry-run keyframes into live simulator targets.

    ``look_direction`` uses absolute head targets. ``nod_yes``, ``shake_no``,
    and ``bow`` are neutral-home gestures: they first align to a straight head pose, move
    only their intended axis, and return to that neutral pose. This matches the
    social expectation that each gesture starts straight, moves only its intended
    head/neck axes, and ends straight without carrying over prior drift.
    """

    home = _gesture_home_positions(plan, reference_positions_by_name)
    targets: list[dict[str, float]] = []
    for keyframe in plan.keyframes:
        if plan.skill_id in NEUTRAL_HOME_GESTURE_SKILLS:
            resolved = {
                name: float(home.get(name, 0.0)) + float(keyframe.positions_by_name.get(name, 0.0))
                for name in HEAD_JOINT_NAMES
            }
        else:
            resolved = {
                name: float(keyframe.positions_by_name.get(name, 0.0))
                for name in HEAD_JOINT_NAMES
            }
        targets.append(resolved)
    return targets


def head_trajectory_axis_path_length_rad(targets: Sequence[Mapping[str, float]]) -> float:
    """Return the longest per-axis path length in a head pose trajectory.

    For axis-specific gestures such as shake_no and nod_yes, this is the total
    left/right or down/up distance that must be traversed. It lets the executor
    stretch too-short requested durations instead of twitching the head quickly.
    """

    if len(targets) < 2:
        return 0.0
    totals = {name: 0.0 for name in HEAD_JOINT_NAMES}
    previous = targets[0]
    for target in targets[1:]:
        for name in HEAD_JOINT_NAMES:
            totals[name] += abs(float(target.get(name, 0.0)) - float(previous.get(name, 0.0)))
        previous = target
    return max(totals.values())


def effective_duration_for_trajectory(
    *,
    requested_duration_s: float,
    targets: Sequence[Mapping[str, float]],
    max_head_velocity_radps: float | None,
    auto_stretch_duration: bool,
    keyframe_durations: Sequence[float] | None = None,
) -> float:
    if requested_duration_s <= 0.0:
        raise SkillExecutionError('scripted duration must be positive')
    if not auto_stretch_duration or max_head_velocity_radps is None or max_head_velocity_radps <= 0.0:
        return float(requested_duration_s)
    if len(targets) < 2:
        return float(requested_duration_s)

    durations = list(keyframe_durations or [])
    if not durations:
        path_length = head_trajectory_axis_path_length_rad(targets)
        if path_length <= 0.0:
            return float(requested_duration_s)
        return max(float(requested_duration_s), path_length / float(max_head_velocity_radps))
    if len(durations) != len(targets):
        raise SkillExecutionError('keyframe duration/target length mismatch')
    if any(float(duration) <= 0.0 for duration in durations):
        raise SkillExecutionError('scripted keyframe durations must be positive')

    requested = sum(float(duration) for duration in durations)
    previous = targets[0]
    required_duration = float(requested_duration_s)
    for target, duration in zip(targets[1:], durations[1:]):
        segment_distance = max(
            abs(float(target.get(name, 0.0)) - float(previous.get(name, 0.0)))
            for name in HEAD_JOINT_NAMES
        )
        if segment_distance > 0.0:
            segment_fraction = float(duration) / requested
            required_duration = max(
                required_duration,
                ((segment_distance / float(max_head_velocity_radps)) / segment_fraction) * 1.25,
            )
        previous = target
    return required_duration


def keyframe_steps_for_durations(durations: Sequence[float], control_hz: float) -> list[int]:
    if control_hz <= 0:
        raise SkillExecutionError('--control-hz must be positive')
    if not durations:
        return []
    if any(float(duration) <= 0.0 for duration in durations):
        raise SkillExecutionError('scripted keyframe durations must be positive')

    total_duration = sum(float(duration) for duration in durations)
    target_total_steps = max(len(durations), int(math.ceil(total_duration * float(control_hz))))
    raw_steps = [float(duration) / total_duration * float(target_total_steps) for duration in durations]
    steps = [max(1, int(math.floor(value))) for value in raw_steps]

    while sum(steps) < target_total_steps:
        fractions = [value - math.floor(value) for value in raw_steps]
        index = max(range(len(steps)), key=lambda item: (fractions[item], raw_steps[item], -item))
        steps[index] += 1
    while sum(steps) > target_total_steps:
        index = max(range(len(steps)), key=lambda item: (steps[item], raw_steps[item], -item))
        if steps[index] <= 1:
            break
        steps[index] -= 1

    return steps


def scaled_keyframe_durations(plan: SkillPlan, effective_duration_s: float) -> list[float]:
    requested = sum(float(keyframe.duration_s) for keyframe in plan.keyframes)
    if requested <= 0.0:
        raise SkillExecutionError('scripted keyframe durations must be positive')
    scale = float(effective_duration_s) / requested
    return [float(keyframe.duration_s) * scale for keyframe in plan.keyframes]


def limit_head_target_velocity(
    previous_positions_by_name: Mapping[str, float],
    target_positions_by_name: Mapping[str, float],
    *,
    dt: float,
    max_velocity_radps: float | None,
) -> dict[str, float]:
    """Clamp per-step target changes before a pose command is streamed.

    This does not change the intended keyframes; it only ensures the generated
    trajectory between those keyframes is slow enough for viewer validation and
    safer for the balancing posture. Use None or 0 to disable the limiter.
    """

    result = dict(target_positions_by_name)
    if max_velocity_radps is None or max_velocity_radps <= 0.0:
        return result
    if dt <= 0.0:
        raise SkillExecutionError('dt must be positive when limiting head velocity')

    max_delta = float(max_velocity_radps) * float(dt)
    for name in HEAD_JOINT_NAMES:
        if name not in result:
            continue
        previous = float(previous_positions_by_name.get(name, result[name]))
        desired = float(result[name])
        delta = desired - previous
        if delta > max_delta:
            result[name] = previous + max_delta
        elif delta < -max_delta:
            result[name] = previous - max_delta
    return result


def plan_head_pose_trajectory(
    plan: SkillPlan,
    resolved_targets: Sequence[Mapping[str, float]],
    keyframe_steps: Sequence[int],
    *,
    start_positions_by_name: Mapping[str, float],
    control_hz: float,
    transition_fraction: float,
    max_head_velocity_radps: float | None,
) -> list[dict[str, float]]:
    """Generate the per-step head pose trajectory to stream to the sim.

    The social-skill execution model is deliberately: plan head-pose trajectory
    first, then send one pose command per control step. This keeps the scripted
    behavior inspectable and prevents the command loop from inventing poses
    from transient simulator state.
    """

    if len(resolved_targets) != len(keyframe_steps):
        raise SkillExecutionError('keyframe target/step length mismatch')
    if control_hz <= 0.0:
        raise SkillExecutionError('--control-hz must be positive')

    strict_target_names = strict_head_target_names_for_plan(plan)
    segment_start_positions = dict(start_positions_by_name)
    previous_commanded_positions = _head_subset(start_positions_by_name)
    dt = 1.0 / float(control_hz)
    trajectory: list[dict[str, float]] = []

    for target_positions, segment_steps in zip(resolved_targets, keyframe_steps):
        for step_index in range(segment_steps):
            target_for_step = target_positions_for_segment_step(
                segment_start_positions,
                target_positions,
                step_index=step_index,
                segment_steps=int(segment_steps),
                transition_fraction=transition_fraction,
                strict_target_names=strict_target_names,
            )
            target_for_step = limit_head_target_velocity(
                previous_commanded_positions,
                target_for_step,
                dt=dt,
                max_velocity_radps=max_head_velocity_radps,
            )
            target_for_step = _head_subset(target_for_step)
            trajectory.append(target_for_step)
            previous_commanded_positions = target_for_step
        segment_start_positions = dict(previous_commanded_positions)

    return trajectory


def motor_command_from_targets(
    state: RobotState,
    target_positions_by_name: Mapping[str, float],
    *,
    kp: float = 10.0,
    kd: float = 0.35,
) -> MotorCommand:
    """Build a MotorCommand from a planned head pose trajectory point.

    Head joints are overridden by ``target_positions_by_name``. All other
    joints are held at the backend's current actuator controls when available,
    matching the existing sync-preroll parity convention.
    """

    controls = command_positions_by_name(state)
    names = list(state.joints.names)
    positions = [float(target_positions_by_name.get(name, controls[name])) for name in names]
    n = len(names)
    return MotorCommand(
        names=names,
        positions=positions,
        velocities=[0.0] * n,
        kp=[float(kp)] * n,
        kd=[float(kd)] * n,
        torques=[0.0] * n,
    )


def validate_scripted_head_plan(plan: SkillPlan) -> None:
    if plan.skill_id not in SUPPORTED_SCRIPTED_SKILLS:
        raise SkillExecutionError(f"unsupported scripted head skill: {plan.skill_id}")
    if plan.execution != "scripted_keyframe":
        raise SkillExecutionError(f"skill {plan.skill_id} is not a scripted_keyframe skill")
    if plan.commands:
        raise SkillExecutionError(f"skill {plan.skill_id} unexpectedly produced velocity commands")
    if not plan.keyframes:
        raise SkillExecutionError(f"skill {plan.skill_id} must produce at least one head keyframe")
    for index, keyframe in enumerate(plan.keyframes):
        unknown = set(keyframe.positions_by_name) - set(HEAD_JOINT_NAMES)
        if unknown:
            raise SkillExecutionError(
                f"skill {plan.skill_id} keyframe {index} targets non-head joints: {sorted(unknown)}"
            )
        if keyframe.duration_s <= 0:
            raise SkillExecutionError(f"skill {plan.skill_id} keyframe {index} duration must be positive")


def _load_robot_api_client_class() -> type["RobotApiClient"]:
    try:
        from soridormi_api.client import RobotApiClient
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
        if exc.name == "zmq":
            raise RuntimeError(
                "Live MuJoCo scripted social skills require pyzmq for the Soridormi API client. "
                "Install project dependencies with `python -m pip install -e .` or install pyzmq "
                "directly with `python -m pip install pyzmq`, then rerun the command. "
                "Use --dry-run to validate the skill plan without connecting to the simulator."
            ) from exc
        raise
    return RobotApiClient


def _read_initial_state(client: "RobotApiClient") -> RobotState:
    try:
        return client.read_state()
    except Exception as exc:  # pragma: no cover - depends on live simulator
        raise RuntimeError(
            "Could not read MuJoCo state. Start the simulator first with: "
            "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera"
        ) from exc



def keyframe_steps_for_plan(plan: SkillPlan, control_hz: float) -> list[int]:
    return keyframe_steps_for_durations([float(keyframe.duration_s) for keyframe in plan.keyframes], control_hz)

def execute_scripted_head_plan(
    plan: SkillPlan,
    *,
    backend: str = "mujoco",
    host: str = "127.0.0.1",
    port: int = 5555,
    control_hz: float = 50.0,
    dry_run: bool = False,
    kp: float = 10.0,
    kd: float = 0.35,
    transition_fraction: float = DEFAULT_TRANSITION_FRACTION,
    max_head_velocity_radps: float | None = DEFAULT_MAX_HEAD_VELOCITY_RADPS,
    auto_stretch_duration: bool = True,
    fall_height_m: float = 0.14,
) -> ScriptedHeadExecutionResult:
    validate_scripted_head_plan(plan)
    if backend != "mujoco":
        raise SkillExecutionError("scripted head skills are sim-only; use --backend mujoco")
    if control_hz <= 0:
        raise SkillExecutionError("--control-hz must be positive")
    if not 0.0 <= transition_fraction <= 1.0:
        raise SkillExecutionError("--transition-fraction must be between 0.0 and 1.0")
    if max_head_velocity_radps is not None and max_head_velocity_radps < 0.0:
        raise SkillExecutionError("--max-head-velocity-radps must be non-negative")
    if fall_height_m <= 0.0:
        raise SkillExecutionError("--fall-height-m must be positive")

    requested_duration_s = sum(float(keyframe.duration_s) for keyframe in plan.keyframes)
    dry_run_targets = resolve_keyframe_targets_for_execution(
        plan,
        {name: 0.0 for name in HEAD_JOINT_NAMES},
    )
    effective_duration_s = effective_duration_for_trajectory(
        requested_duration_s=requested_duration_s,
        targets=dry_run_targets,
        max_head_velocity_radps=max_head_velocity_radps,
        auto_stretch_duration=auto_stretch_duration,
        keyframe_durations=[float(keyframe.duration_s) for keyframe in plan.keyframes],
    )
    keyframe_durations = scaled_keyframe_durations(plan, effective_duration_s)
    keyframe_steps = keyframe_steps_for_durations(keyframe_durations, control_hz)
    steps = sum(keyframe_steps)

    dry_trajectory = plan_head_pose_trajectory(
        plan,
        dry_run_targets,
        keyframe_steps,
        start_positions_by_name={name: 0.0 for name in HEAD_JOINT_NAMES},
        control_hz=control_hz,
        transition_fraction=transition_fraction,
        max_head_velocity_radps=max_head_velocity_radps,
    )
    dry_target_min, dry_target_max = _head_ranges(dry_trajectory)

    if dry_run:
        final_target_positions = dry_run_targets[-1]
        return ScriptedHeadExecutionResult(
            skill_id=plan.skill_id,
            backend=backend,
            executed=False,
            steps=steps,
            control_hz=float(control_hz),
            duration_s=effective_duration_s,
            requested_duration_s=requested_duration_s,
            effective_duration_s=effective_duration_s,
            target_positions_by_name=_head_subset(final_target_positions),
            start_positions_by_name={},
            final_positions_by_name=_head_subset(final_target_positions),
            target_min_positions_by_name=dry_target_min,
            target_max_positions_by_name=dry_target_max,
            observed_min_positions_by_name={},
            observed_max_positions_by_name={},
            keyframe_targets=_keyframe_targets_for_report(plan, dry_run_targets),
            transition_fraction=float(transition_fraction),
            max_head_velocity_radps=max_head_velocity_radps,
            auto_stretched_duration=effective_duration_s > requested_duration_s + 1e-9,
            fall_height_m=float(fall_height_m),
            fallen=None,
        )

    robot_api_client_class = _load_robot_api_client_class()
    client = robot_api_client_class(host=host, port=port)
    try:
        state = _read_initial_state(client)
        initial_positions = joint_positions_by_name(state)
        initial_controls = command_positions_by_name(state)
        resolved_targets = resolve_keyframe_targets_for_execution(plan, initial_positions)
        effective_duration_s = effective_duration_for_trajectory(
            requested_duration_s=requested_duration_s,
            targets=resolved_targets,
            max_head_velocity_radps=max_head_velocity_radps,
            auto_stretch_duration=auto_stretch_duration,
            keyframe_durations=[float(keyframe.duration_s) for keyframe in plan.keyframes],
        )
        keyframe_durations = scaled_keyframe_durations(plan, effective_duration_s)
        keyframe_steps = keyframe_steps_for_durations(keyframe_durations, control_hz)
        steps = sum(keyframe_steps)
        final_target_positions = resolved_targets[-1]
        trajectory = plan_head_pose_trajectory(
            plan,
            resolved_targets,
            keyframe_steps,
            start_positions_by_name=initial_controls,
            control_hz=control_hz,
            transition_fraction=transition_fraction,
            max_head_velocity_radps=max_head_velocity_radps,
        )
        commanded_history: list[dict[str, float]] = []
        observed_history: list[dict[str, float]] = [_head_subset(initial_positions)]
        state_history: list[RobotState] = [state]
        final_positions = initial_positions
        for target_for_step in trajectory:
            commanded_history.append(_head_subset(target_for_step))
            command = motor_command_from_targets(state, target_for_step, kp=kp, kd=kd)
            state = client.step_motor_command(command)
            final_positions = joint_positions_by_name(state)
            observed_history.append(_head_subset(final_positions))
            state_history.append(state)
        commanded_min, commanded_max = _head_ranges(commanded_history)
        observed_min, observed_max = _head_ranges(observed_history)
        min_base_height, max_base_height, final_base_height = _base_height_range(state_history)
        fallen = bool(min_base_height is not None and min_base_height < float(fall_height_m))
        return ScriptedHeadExecutionResult(
            skill_id=plan.skill_id,
            backend=backend,
            executed=True,
            steps=steps,
            control_hz=float(control_hz),
            duration_s=effective_duration_s,
            requested_duration_s=requested_duration_s,
            effective_duration_s=effective_duration_s,
            target_positions_by_name=_head_subset(final_target_positions),
            start_positions_by_name=_head_subset(initial_positions),
            final_positions_by_name=_head_subset(final_positions),
            target_min_positions_by_name=commanded_min,
            target_max_positions_by_name=commanded_max,
            observed_min_positions_by_name=observed_min,
            observed_max_positions_by_name=observed_max,
            keyframe_targets=_keyframe_targets_for_report(plan, resolved_targets),
            transition_fraction=float(transition_fraction),
            max_head_velocity_radps=max_head_velocity_radps,
            auto_stretched_duration=effective_duration_s > requested_duration_s + 1e-9,
            observed_min_base_height_m=min_base_height,
            observed_max_base_height_m=max_base_height,
            final_base_height_m=final_base_height,
            fall_height_m=float(fall_height_m),
            fallen=fallen,
        )
    finally:
        client.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute safe scripted head/neck social skills against an already-running MuJoCo sim."
    )
    parser.add_argument("skill", help="Scripted social skill id, e.g. neutral_head, look_direction, nod_yes, shake_no, bow, or express_attention.")
    parser.add_argument("--manifest", default=str(DEFAULT_SKILL_MANIFEST), help="Path to skill manifest JSON.")
    parser.add_argument("--args", default="{}", help="Skill parameter JSON object.")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco"], help="Execution backend; hardware is not exposed.")
    parser.add_argument("--host", default="127.0.0.1", help="MuJoCo API host.")
    parser.add_argument("--port", type=int, default=5555, help="MuJoCo API port.")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Scripted control frequency.")
    parser.add_argument(
        "--transition-fraction",
        type=float,
        default=DEFAULT_TRANSITION_FRACTION,
        help=f"Fraction of each keyframe segment spent ramping before holding the target pose. default: {DEFAULT_TRANSITION_FRACTION}",
    )
    parser.add_argument(
        "--max-head-velocity-radps",
        type=float,
        default=DEFAULT_MAX_HEAD_VELOCITY_RADPS,
        help=f"Maximum planned head target speed in rad/s. Use 0 to disable. default: {DEFAULT_MAX_HEAD_VELOCITY_RADPS}",
    )
    parser.add_argument(
        "--no-auto-stretch-duration",
        action="store_true",
        help="Do not automatically extend too-short nod/shake durations to satisfy the head velocity limit.",
    )
    parser.add_argument(
        "--fall-height-m",
        type=float,
        default=0.14,
        help="Base-height fall threshold used for live telemetry (default: 0.14).",
    )
    parser.add_argument("--kp", type=float, default=10.0, help="Position gain for the scripted command.")
    parser.add_argument("--kd", type=float, default=0.35, help="Velocity damping for the scripted command.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without connecting to MuJoCo.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _print_human(plan: SkillPlan, result: ScriptedHeadExecutionResult) -> None:
    print("Soridormi scripted social skill")
    print("================================")
    print(plan.summary)
    print(f"Backend: {result.backend}")
    print(f"Executed: {result.executed}")
    print(f"Steps: {result.steps}")
    print(f"Control Hz: {result.control_hz:.1f}")
    if abs(result.effective_duration_s - result.requested_duration_s) > 1e-9:
        print(f"Requested duration: {result.requested_duration_s:.2f}s")
        print(f"Effective duration: {result.effective_duration_s:.2f}s (auto-stretched for head speed limit)")
    else:
        print(f"Duration: {result.duration_s:.2f}s")
    print(f"Transition fraction: {result.transition_fraction:.2f}")
    if result.max_head_velocity_radps is not None and result.max_head_velocity_radps > 0.0:
        print(f"Max head target speed: {result.max_head_velocity_radps:.2f} rad/s")
    print("Keyframe target head positions:")
    for row in result.keyframe_targets:
        label = row.get("label") or f"keyframe_{row.get('index')}"
        duration = float(row.get("duration_s", 0.0))
        positions = row.get("positions_by_name", {})
        head_pitch = float(positions.get("head_pitch", 0.0))
        head_yaw = float(positions.get("head_yaw", 0.0))
        print(f"- {label}: head_yaw={head_yaw:.3f}, head_pitch={head_pitch:.3f}, duration={duration:.2f}s")
    print("Final target head positions:")
    for name, value in sorted(result.target_positions_by_name.items()):
        print(f"- {name}: {value:.3f}")
    print("Commanded target head range:")
    for name in HEAD_JOINT_NAMES:
        lo = float(result.target_min_positions_by_name.get(name, 0.0))
        hi = float(result.target_max_positions_by_name.get(name, 0.0))
        print(f"- {name}: min={lo:.3f}, max={hi:.3f}")
    if result.executed:
        print("Observed simulator head range:")
        for name in HEAD_JOINT_NAMES:
            lo = float(result.observed_min_positions_by_name.get(name, 0.0))
            hi = float(result.observed_max_positions_by_name.get(name, 0.0))
            print(f"- {name}: min={lo:.3f}, max={hi:.3f}")
        if result.observed_min_base_height_m is not None:
            print("Base-height stability:")
            print(f"- fall_height_m: {float(result.fall_height_m or 0.0):.3f}")
            print(f"- min_base_height_m: {float(result.observed_min_base_height_m):.3f}")
            if result.final_base_height_m is not None:
                print(f"- final_base_height_m: {float(result.final_base_height_m):.3f}")
            print(f"- fallen: {bool(result.fallen)}")
    else:
        print("Dry-run only; no simulator or hardware command was executed.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        registry = SkillExecutionRegistry.from_manifest_path(args.manifest)
        parameters = _load_json_args(args.args)
        plan = registry.create_plan(args.skill, parameters)
        result = execute_scripted_head_plan(
            plan,
            backend=args.backend,
            host=args.host,
            port=args.port,
            control_hz=args.control_hz,
            dry_run=args.dry_run,
            kp=args.kp,
            kd=args.kd,
            transition_fraction=args.transition_fraction,
            max_head_velocity_radps=args.max_head_velocity_radps,
            auto_stretch_duration=not args.no_auto_stretch_duration,
            fall_height_m=args.fall_height_m,
        )
    except (SkillExecutionError, RuntimeError) as exc:
        payload = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Scripted social skill failed: {exc}")
        return 2

    payload = {"ok": True, "plan": plan.to_dict(), "result": result.to_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(plan, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
