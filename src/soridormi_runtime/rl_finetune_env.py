from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.action_mapper import PolicyActionMapper
from soridormi_runtime.action_postprocessor import ActionPostprocessor
from soridormi_runtime.backends.sim import SimRobot
from soridormi_runtime.policy_command import GaitPhaseGenerator, PolicyCommand
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.walking_reward import WalkingRewardConfig, compute_walking_reward


ACTION_SIZE = 14
DEFAULT_OUTPUT = Path("/data/rl_finetune_env/m617_smoke.json")


class PolicyLike(Protocol):
    def compute_action(self, state: RobotState) -> np.ndarray:
        ...

    def set_command_vector(self, command: list[float]) -> None:
        ...

    def set_imitation_phase(self, imitation_phase: list[float]) -> None:
        ...

    def set_motor_targets(self, joint_names: list[str], positions: list[float] | np.ndarray) -> None:
        ...

    def bootstrap_defaults_from_state(self, state: RobotState) -> dict[str, float]:
        ...


class MapperLike(Protocol):
    def action_to_command(
        self,
        action: np.ndarray | list[float],
        state: RobotState | None = None,
        dt: float | None = None,
    ) -> MotorCommand:
        ...

    def set_default_positions_by_name(self, positions_by_name: dict[str, float]) -> None:
        ...

    def reset_targets(self) -> None:
        ...


class RobotLike(Protocol):
    def read_state(self) -> RobotState:
        ...

    def step_motor_command(self, command: MotorCommand) -> RobotState:
        ...

    def reset(self) -> str:
        ...


@dataclass(frozen=True)
class ResidualActionConfig:
    """Safety envelope for residual policy actions.

    Residual fine-tuning starts from a trusted teacher policy and learns a small
    correction. The external RL agent supplies an unconstrained 14D residual in
    roughly [-1, 1]. Soridormi clips that value, multiplies it by
    residual_scale, adds it to the teacher action, then optionally clips the
    final action before mapping to joint targets.
    """

    residual_scale: float = 0.05
    residual_clip_abs: float = 1.0
    final_action_clip_abs: float | None = None

    def apply(self, teacher_action: np.ndarray, residual_action: np.ndarray | list[float] | None) -> tuple[np.ndarray, np.ndarray]:
        teacher = _action_array(teacher_action, "teacher_action")
        if residual_action is None:
            raw_residual = np.zeros(ACTION_SIZE, dtype=np.float32)
        else:
            raw_residual = _action_array(residual_action, "residual_action")
        clipped_residual = np.clip(raw_residual, -self.residual_clip_abs, self.residual_clip_abs)
        applied_residual = clipped_residual * float(self.residual_scale)
        final_action = teacher + applied_residual
        if self.final_action_clip_abs is not None and self.final_action_clip_abs > 0.0:
            final_action = np.clip(final_action, -self.final_action_clip_abs, self.final_action_clip_abs)
        return applied_residual.astype(np.float32), final_action.astype(np.float32)


@dataclass(frozen=True)
class RlFineTuneStep:
    step_index: int
    state_time: float
    next_state_time: float
    teacher_action: list[float]
    residual_action: list[float]
    final_action: list[float]
    motor_command: dict[str, Any]
    metrics: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]


@dataclass(frozen=True)
class RlFineTuneRunResult:
    ok: bool
    profile: str
    steps_requested: int
    steps_completed: int
    residual_config: dict[str, Any]
    output_path: str | None
    reward_config: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterable[None]:
    old_values: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class RlFineTuneEnv:
    """Synchronous MuJoCo residual fine-tuning environment boundary.

    This class is intentionally small: it does not implement an RL algorithm.
    It exposes the backbone primitive needed by M6.18/M6.19:

        state -> teacher action -> residual correction -> MotorCommand -> sim step

    The default implementation uses the same RobotState/MotorCommand API as the
    runtime and therefore stays compatible with the sim-to-real boundary.
    """

    def __init__(
        self,
        *,
        profile: str | PolicyProfile = "open_duck_forward",
        robot: RobotLike | None = None,
        policy: PolicyLike | None = None,
        mapper: MapperLike | None = None,
        command: PolicyCommand | None = None,
        phase_generator: GaitPhaseGenerator | None = None,
        residual_config: ResidualActionConfig | None = None,
        reward_config: WalkingRewardConfig | None = None,
        control_hz: float | None = None,
        host: str = "127.0.0.1",
        port: int = 5555,
        reset_on_start: bool = True,
    ) -> None:
        self.profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
        self.profile_env = self.profile.env()
        self.control_hz = float(control_hz or self.profile_env.get("CONTROL_HZ", "50"))
        self.dt = 1.0 / self.control_hz
        self.robot = robot or SimRobot(host=host, port=port)
        self.residual_config = residual_config or ResidualActionConfig()
        self.reward_config = reward_config or WalkingRewardConfig()
        self.previous_final_action: np.ndarray | None = None
        self.reset_on_start = bool(reset_on_start)
        self.step_index = 0
        self.current_state: RobotState | None = None
        self.policy_defaults_bootstrapped = False

        with _temporary_env(self.profile_env):
            if policy is None:
                from soridormi_runtime.policy_factory import make_runtime_policy

                self.policy = make_runtime_policy(
                    policy_path=self.profile.model.path,
                    robot_config_path=os.environ.get("SORIDORMI_ROBOT_CONFIG"),
                )
            else:
                self.policy = policy
            self.mapper = mapper or PolicyActionMapper.from_robot_config(
                path=os.environ.get("SORIDORMI_ROBOT_CONFIG"),
                use_env_overrides=True,
            )
            self.command = command or PolicyCommand.from_env()
            self.phase_generator = phase_generator or GaitPhaseGenerator.from_env()
            self.postprocessor = ActionPostprocessor.from_env()

    def reset(self) -> RobotState:
        if self.reset_on_start:
            reset = getattr(self.robot, "reset", None)
            if callable(reset):
                reset()
        self.step_index = 0
        self.policy_defaults_bootstrapped = False
        resetter = getattr(self.mapper, "reset_targets", None)
        if callable(resetter):
            resetter()
        self.current_state = self.robot.read_state()
        self.previous_final_action = None
        return self.current_state

    def step(self, residual_action: np.ndarray | list[float] | None = None) -> RlFineTuneStep:
        state = self.current_state or self.robot.read_state()
        self._bootstrap_defaults_once(state)

        self._set_policy_inputs()
        teacher_action = _action_array(self.policy.compute_action(state), "teacher_action")
        teacher_action = _action_array(
            self.postprocessor.apply(teacher_action, list(state.joints.names)),
            "teacher_action",
        )
        residual_applied, final_action = self.residual_config.apply(teacher_action, residual_action)

        command = self.mapper.action_to_command(final_action, state=state, dt=self.dt)
        setter = getattr(self.policy, "set_motor_targets", None)
        if callable(setter):
            setter(command.names, command.positions)

        next_state = self.robot.step_motor_command(command)
        metrics = _transition_metrics(state, next_state, command)
        reward = compute_walking_reward(
            state,
            next_state,
            command=self.command,
            motor_command=command,
            final_action=final_action,
            residual_action=residual_applied,
            previous_final_action=self.previous_final_action,
            config=self.reward_config,
        )
        metrics["reward"] = float(reward.reward)
        metrics["reward_terms"] = dict(reward.terms)
        metrics["reward_diagnostics"] = dict(reward.diagnostics)
        metrics["terminated"] = bool(reward.terminated)
        transition = RlFineTuneStep(
            step_index=self.step_index,
            state_time=float(state.time),
            next_state_time=float(next_state.time),
            teacher_action=_float_list(teacher_action),
            residual_action=_float_list(residual_applied),
            final_action=_float_list(final_action),
            motor_command=_motor_command_summary(command),
            metrics=metrics,
            state_before=_state_summary(state),
            state_after=_state_summary(next_state),
        )
        self.previous_final_action = final_action.copy()
        self.current_state = next_state
        self.step_index += 1
        return transition

    def _bootstrap_defaults_once(self, state: RobotState) -> None:
        if self.policy_defaults_bootstrapped:
            return
        bootstrap = getattr(self.policy, "bootstrap_defaults_from_state", None)
        defaults: dict[str, float] = {}
        if callable(bootstrap):
            returned = bootstrap(state)
            if isinstance(returned, dict):
                defaults = {str(k): float(v) for k, v in returned.items()}
        if defaults:
            setter = getattr(self.mapper, "set_default_positions_by_name", None)
            if callable(setter):
                setter(defaults)
        self.policy_defaults_bootstrapped = True

    def _set_policy_inputs(self) -> None:
        command_vector = self.command.as_list()
        phase_vector = list(self.phase_generator.advance_and_as_list())
        command_setter = getattr(self.policy, "set_command_vector", None)
        if callable(command_setter):
            command_setter(command_vector)
        phase_setter = getattr(self.policy, "set_imitation_phase", None)
        if callable(phase_setter):
            phase_setter(phase_vector)


def run_zero_residual_smoke(
    *,
    profile: str,
    steps: int,
    residual_config: ResidualActionConfig,
    output_path: Path | None,
    reward_config: WalkingRewardConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 5555,
    reset_on_start: bool = True,
) -> RlFineTuneRunResult:
    env = RlFineTuneEnv(
        profile=profile,
        host=host,
        port=port,
        residual_config=residual_config,
        reward_config=reward_config,
        reset_on_start=reset_on_start,
    )
    transitions: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        env.reset()
        for _ in range(max(0, int(steps))):
            transitions.append(asdict(env.step(np.zeros(ACTION_SIZE, dtype=np.float32))))
    except Exception as exc:  # pragma: no cover - depends on live simulator
        errors.append(repr(exc))

    result = RlFineTuneRunResult(
        ok=not errors,
        profile=profile,
        steps_requested=int(steps),
        steps_completed=len(transitions),
        residual_config=asdict(residual_config),
        reward_config=asdict(reward_config or WalkingRewardConfig()),
        output_path=str(output_path) if output_path is not None else None,
        transitions=transitions,
        errors=errors,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _action_array(values: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (1, ACTION_SIZE):
        arr = arr.reshape(ACTION_SIZE)
    if arr.shape != (ACTION_SIZE,):
        raise ValueError(f"{name} must have shape ({ACTION_SIZE},) or (1, {ACTION_SIZE}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _float_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]


def _state_summary(state: RobotState) -> dict[str, Any]:
    return {
        "time": float(state.time),
        "base_position_xyz": None if state.base_position_xyz is None else [float(x) for x in state.base_position_xyz],
        "base_quat_wxyz": None if state.base_quat_wxyz is None else [float(x) for x in state.base_quat_wxyz],
        "feet_contacts": None if state.feet_contacts is None else [float(x) for x in state.feet_contacts],
        "joint_position_abs_max": max((abs(float(x)) for x in state.joints.positions), default=0.0),
        "joint_velocity_abs_max": max((abs(float(x)) for x in state.joints.velocities), default=0.0),
    }


def _motor_command_summary(command: MotorCommand) -> dict[str, Any]:
    return {
        "joint_count": len(command.names),
        "names": list(command.names),
        "position_abs_max": max((abs(float(x)) for x in command.positions), default=0.0),
        "kp_mean": float(np.mean(command.kp)) if command.kp else 0.0,
        "kd_mean": float(np.mean(command.kd)) if command.kd else 0.0,
    }


def _transition_metrics(before: RobotState, after: RobotState, command: MotorCommand) -> dict[str, Any]:
    before_pos = before.base_position_xyz or [0.0, 0.0, 0.0]
    after_pos = after.base_position_xyz or before_pos
    delta = [float(a) - float(b) for a, b in zip(after_pos, before_pos)]
    dt = max(0.0, float(after.time) - float(before.time))
    return {
        "dt": dt,
        "base_delta_xyz": delta,
        "forward_delta_x": delta[0] if len(delta) > 0 else 0.0,
        "lateral_delta_y": delta[1] if len(delta) > 1 else 0.0,
        "vertical_delta_z": delta[2] if len(delta) > 2 else 0.0,
        "command_position_abs_max": max((abs(float(x)) for x in command.positions), default=0.0),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M6.17 MuJoCo residual fine-tuning environment smoke loop.")
    parser.add_argument("--profile", default="open_duck_forward", help="Teacher policy profile name or YAML path")
    parser.add_argument("--steps", type=int, default=20, help="Number of synchronous sim steps to run")
    parser.add_argument("--residual-scale", type=float, default=0.05, help="Scale applied to clipped residual action")
    parser.add_argument("--residual-clip-abs", type=float, default=1.0, help="Clip raw residual action to ±this value")
    parser.add_argument("--final-action-clip-abs", type=float, default=0.0, help="Optional final action clip; 0 disables")
    parser.add_argument("--host", default=os.environ.get("SIM_HOST", "127.0.0.1"), help="Simulator API host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "5555")), help="Simulator API port")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path")
    parser.add_argument("--target-height", type=float, default=0.30, help="Nominal base height for reward shaping")
    parser.add_argument("--fall-height", type=float, default=0.14, help="Base height below which reward terminates")
    parser.add_argument("--min-upright", type=float, default=0.65, help="Minimum upright score before reward terminates")
    parser.add_argument("--forward-velocity-sigma", type=float, default=0.20, help="Velocity tracking sigma for commanded x velocity")
    parser.add_argument("--no-reset", action="store_true", help="Do not call simulator reset before stepping")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without connecting to simulator")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    final_clip = args.final_action_clip_abs if args.final_action_clip_abs > 0 else None
    residual_config = ResidualActionConfig(
        residual_scale=args.residual_scale,
        residual_clip_abs=args.residual_clip_abs,
        final_action_clip_abs=final_clip,
    )
    reward_config = WalkingRewardConfig(
        target_height=args.target_height,
        fall_height=args.fall_height,
        min_upright=args.min_upright,
        forward_velocity_sigma=args.forward_velocity_sigma,
    )

    if args.dry_run:
        payload = {
            "profile": args.profile,
            "steps": args.steps,
            "host": args.host,
            "port": args.port,
            "output": str(args.output),
            "reset_on_start": not args.no_reset,
            "residual_config": asdict(residual_config),
            "reward_config": asdict(reward_config),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    result = run_zero_residual_smoke(
        profile=args.profile,
        steps=args.steps,
        residual_config=residual_config,
        reward_config=reward_config,
        output_path=args.output,
        host=args.host,
        port=args.port,
        reset_on_start=not args.no_reset,
    )
    print("Soridormi RL fine-tuning environment smoke")
    print("===========================================")
    print(f"Profile: {result.profile}")
    print(f"Steps completed: {result.steps_completed}/{result.steps_requested}")
    print(f"Output: {result.output_path}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
