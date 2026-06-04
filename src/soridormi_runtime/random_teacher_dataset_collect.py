from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scenario_curriculum import (
    DEFAULT_SCENARIO_MANIFEST,
    ScenarioCurriculumError,
    ScenarioDefinition,
    get_scenario_definition,
    list_scenarios,
    validate_scenario_for_teacher_collection,
)
from soridormi_runtime.rl_finetune_env import ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.teacher_dataset_collect import _manifest_path_for, _sample_from_step
from soridormi_runtime.training_dataset import DATASET_SCHEMA_VERSION, sha256_file
from soridormi_runtime.walking_reward import WalkingRewardConfig


DEFAULT_RANDOM_OUTPUT = Path("/data/training_datasets/teacher_policy_random_walk.jsonl")


def _reset_env_with_retries(
    env: Any,
    *,
    attempts: int = 1,
    sleep_s: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[bool, int, str | None]:
    """Reset a simulator environment with bounded transient-error retries."""

    max_attempts = max(1, int(attempts))
    delay = max(0.0, float(sleep_s))
    last_error: str | None = None
    for attempt_index in range(max_attempts):
        try:
            env.reset()
            return True, attempt_index + 1, None
        except Exception as exc:  # pragma: no cover - concrete exception type varies by transport/backend
            last_error = repr(exc)
            if attempt_index + 1 >= max_attempts:
                break
            if delay > 0.0:
                sleep_fn(delay)
    return False, max_attempts, last_error


@dataclass(frozen=True)
class CommandRange:
    minimum: float
    maximum: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(float(self.minimum), float(self.maximum)))

    def describe(self) -> dict[str, float]:
        return {"minimum": float(self.minimum), "maximum": float(self.maximum)}


@dataclass(frozen=True)
class HoldStepRange:
    minimum: int
    maximum: int

    def sample(self, rng: np.random.Generator) -> int:
        low = max(1, int(self.minimum))
        high = max(low, int(self.maximum))
        # numpy's integer upper bound is exclusive.
        return int(rng.integers(low, high + 1))

    def describe(self) -> dict[str, int]:
        return {"minimum": int(self.minimum), "maximum": int(self.maximum)}



@dataclass
class CommandCoverageStats:
    count: int = 0
    vx_min: float | None = None
    vx_max: float | None = None
    vy_min: float | None = None
    vy_max: float | None = None
    yaw_min: float | None = None
    yaw_max: float | None = None
    vx_sum: float = 0.0
    vy_sum: float = 0.0
    yaw_sum: float = 0.0
    stop_like_count: int = 0

    def add(self, command: PolicyCommand) -> None:
        vx = float(command.x_velocity)
        vy = float(command.y_velocity)
        yaw = float(command.yaw_velocity)
        self.count += 1
        self.vx_min = vx if self.vx_min is None else min(self.vx_min, vx)
        self.vx_max = vx if self.vx_max is None else max(self.vx_max, vx)
        self.vy_min = vy if self.vy_min is None else min(self.vy_min, vy)
        self.vy_max = vy if self.vy_max is None else max(self.vy_max, vy)
        self.yaw_min = yaw if self.yaw_min is None else min(self.yaw_min, yaw)
        self.yaw_max = yaw if self.yaw_max is None else max(self.yaw_max, yaw)
        self.vx_sum += vx
        self.vy_sum += vy
        self.yaw_sum += yaw
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(yaw) < 1e-6:
            self.stop_like_count += 1

    def describe(self) -> dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "vx": {},
                "vy": {},
                "yaw": {},
                "stop_like_count": 0,
                "stop_like_ratio": 0.0,
            }
        count = float(self.count)
        return {
            "count": int(self.count),
            "vx": {
                "minimum": float(self.vx_min if self.vx_min is not None else 0.0),
                "maximum": float(self.vx_max if self.vx_max is not None else 0.0),
                "mean": float(self.vx_sum / count),
            },
            "vy": {
                "minimum": float(self.vy_min if self.vy_min is not None else 0.0),
                "maximum": float(self.vy_max if self.vy_max is not None else 0.0),
                "mean": float(self.vy_sum / count),
            },
            "yaw": {
                "minimum": float(self.yaw_min if self.yaw_min is not None else 0.0),
                "maximum": float(self.yaw_max if self.yaw_max is not None else 0.0),
                "mean": float(self.yaw_sum / count),
            },
            "stop_like_count": int(self.stop_like_count),
            "stop_like_ratio": float(self.stop_like_count / count),
        }


def interpolate_command(start: PolicyCommand, target: PolicyCommand, alpha: float) -> PolicyCommand:
    clamped = min(1.0, max(0.0, float(alpha)))

    def blend(a: float, b: float) -> float:
        return float(a + (b - a) * clamped)

    return PolicyCommand(
        x_velocity=blend(start.x_velocity, target.x_velocity),
        y_velocity=blend(start.y_velocity, target.y_velocity),
        yaw_velocity=blend(start.yaw_velocity, target.yaw_velocity),
        neck_pitch=blend(start.neck_pitch, target.neck_pitch),
        head_pitch=blend(start.head_pitch, target.head_pitch),
        head_yaw=blend(start.head_yaw, target.head_yaw),
        head_roll=blend(start.head_roll, target.head_roll),
    )


def ramped_segment_command(
    *,
    previous_command: PolicyCommand,
    target_command: PolicyCommand,
    segment_step_index: int,
    ramp_steps: int,
) -> tuple[PolicyCommand, float]:
    if ramp_steps <= 0:
        return target_command, 1.0
    alpha = min(1.0, float(int(segment_step_index) + 1) / float(max(1, int(ramp_steps))))
    return interpolate_command(previous_command, target_command, alpha), alpha


@dataclass(frozen=True)
class RandomCommandSegment:
    segment_index: int
    start_step: int
    hold_steps: int
    command: PolicyCommand
    segment_id: str

    @property
    def end_step_exclusive(self) -> int:
        return int(self.start_step) + int(self.hold_steps)

    def describe(self) -> dict[str, Any]:
        payload = self.command.describe()
        payload.update(
            {
                "segment_index": int(self.segment_index),
                "start_step": int(self.start_step),
                "hold_steps": int(self.hold_steps),
                "end_step_exclusive": int(self.end_step_exclusive),
                "segment_id": self.segment_id,
            }
        )
        return payload


@dataclass(frozen=True)
class RandomTeacherDatasetCollectResult:
    ok: bool
    profile: str
    output_path: str
    manifest_path: str
    episodes_requested: int
    steps_per_episode: int
    sample_count: int
    skipped_steps: int = 0
    terminated_episodes: int = 0
    dataset_sha256: str | None = None
    seed: int = 0
    vx_range: dict[str, float] = field(default_factory=dict)
    vy_range: dict[str, float] = field(default_factory=dict)
    yaw_range: dict[str, float] = field(default_factory=dict)
    command_hold_steps: dict[str, int] = field(default_factory=dict)
    stop_probability: float = 0.0
    command_ramp_steps: int = 0
    segment_count: int = 0
    command_coverage: dict[str, Any] = field(default_factory=dict)
    scenario_id: str | None = None
    scenario_status: str | None = None
    scenario_family: str | None = None
    skill_id: str | None = None
    scenario_dataset_tags: list[str] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)
    environment_context: dict[str, Any] = field(default_factory=dict)
    command_space: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _parse_range(text: str, *, name: str) -> CommandRange:
    parts = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be MIN,MAX")
    try:
        minimum = float(parts[0])
        maximum = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain floats") from exc
    if minimum > maximum:
        raise argparse.ArgumentTypeError(f"{name} minimum must be <= maximum")
    return CommandRange(minimum=minimum, maximum=maximum)


def _parse_hold_range(text: str) -> HoldStepRange:
    parts = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--command-hold-steps must be MIN,MAX")
    try:
        minimum = int(parts[0])
        maximum = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--command-hold-steps must contain integers") from exc
    if minimum <= 0 or maximum <= 0:
        raise argparse.ArgumentTypeError("--command-hold-steps values must be positive")
    if minimum > maximum:
        raise argparse.ArgumentTypeError("--command-hold-steps minimum must be <= maximum")
    return HoldStepRange(minimum=minimum, maximum=maximum)


def _segment_id(command: PolicyCommand, segment_index: int) -> str:
    values = command.as_list()
    safe = []
    for value in values[:3]:
        safe.append(f"{float(value):+.3f}".replace("+", "p").replace("-", "m").replace(".", "_"))
    return f"segment_{segment_index}_{'_'.join(safe)}"


def generate_random_command_schedule(
    *,
    steps_per_episode: int,
    vx_range: CommandRange,
    vy_range: CommandRange,
    yaw_range: CommandRange,
    command_hold_steps: HoldStepRange,
    rng: np.random.Generator,
    stop_probability: float = 0.10,
) -> list[RandomCommandSegment]:
    """Generate a piecewise-constant command schedule for one rollout episode."""

    total_steps = max(0, int(steps_per_episode))
    if total_steps <= 0:
        return []
    stop_p = min(1.0, max(0.0, float(stop_probability)))
    segments: list[RandomCommandSegment] = []
    step = 0
    segment_index = 0
    while step < total_steps:
        hold = min(command_hold_steps.sample(rng), total_steps - step)
        if rng.random() < stop_p:
            command = PolicyCommand()
        else:
            command = PolicyCommand(
                x_velocity=vx_range.sample(rng),
                y_velocity=vy_range.sample(rng),
                yaw_velocity=yaw_range.sample(rng),
            )
        segment = RandomCommandSegment(
            segment_index=segment_index,
            start_step=step,
            hold_steps=hold,
            command=command,
            segment_id=_segment_id(command, segment_index),
        )
        segments.append(segment)
        step += hold
        segment_index += 1
    return segments


def collect_random_teacher_dataset(
    *,
    profile: str,
    output_path: str | Path = DEFAULT_RANDOM_OUTPUT,
    manifest_path: str | Path | None = None,
    episodes: int = 1,
    steps_per_episode: int = 1000,
    vx_range: CommandRange = CommandRange(-0.03, 0.15),
    vy_range: CommandRange = CommandRange(-0.03, 0.03),
    yaw_range: CommandRange = CommandRange(-0.20, 0.20),
    command_hold_steps: HoldStepRange = HoldStepRange(80, 250),
    stop_probability: float = 0.10,
    command_ramp_steps: int = 20,
    seed: int = 123,
    reward_config: WalkingRewardConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 5555,
    stop_on_terminated: bool = True,
    reset_on_start: bool = True,
    reset_attempts: int = 1,
    reset_retry_sleep: float = 0.0,
    env_factory: Callable[..., Any] | None = None,
    scenario: ScenarioDefinition | None = None,
    initial_warnings: list[str] | None = None,
) -> RandomTeacherDatasetCollectResult:
    """Collect teacher BC samples under random piecewise velocity commands.

    Unlike ``teacher_dataset_collect.collect_teacher_dataset``, this collector
    changes the policy command several times inside each episode. This is the
    M6B data path for continuous command-conditioned free walking: one rollout
    contains stand, stop, turn, curve, lateral, forward, and backward command
    changes sampled from conservative ranges. By default, command changes are
    ramped over a short number of control steps so the BC dataset contains
    smooth speed transitions instead of only abrupt step commands.
    """

    output = Path(output_path)
    manifest = _manifest_path_for(output, Path(manifest_path) if manifest_path is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    reward_cfg = reward_config or WalkingRewardConfig()
    rng = np.random.default_rng(int(seed))
    factory = env_factory or RlFineTuneEnv
    errors: list[str] = []
    warnings: list[str] = list(initial_warnings or [])
    sample_count = 0
    skipped_steps = 0
    terminated_episodes = 0
    segment_count = 0
    coverage = CommandCoverageStats()
    global_step_index = 0
    scenario_metadata = scenario.dataset_metadata() if scenario is not None else {}
    resolved_scenario_id = (
        str(scenario_metadata.get("scenario_id"))
        if scenario_metadata.get("scenario_id")
        else f"{profile}:random_walk_seed_{int(seed)}"
    )

    with output.open("w", encoding="utf-8") as f:
        try:
            env = factory(
                profile=profile,
                host=host,
                port=port,
                command=PolicyCommand(),
                residual_config=ResidualActionConfig(residual_scale=0.0),
                reward_config=reward_cfg,
                reset_on_start=reset_on_start,
            )
        except Exception as exc:
            errors.append(f"could not construct training environment: {exc!r}")
            env = None

        if env is not None:
            for episode_index in range(max(0, int(episodes))):
                schedule = generate_random_command_schedule(
                    steps_per_episode=steps_per_episode,
                    vx_range=vx_range,
                    vy_range=vy_range,
                    yaw_range=yaw_range,
                    command_hold_steps=command_hold_steps,
                    stop_probability=stop_probability,
                    rng=rng,
                )
                segment_count += len(schedule)
                if not schedule:
                    warnings.append(f"episode {episode_index}: empty command schedule")
                    continue

                reset_ok, reset_tries_used, reset_error = _reset_env_with_retries(
                    env,
                    attempts=reset_attempts,
                    sleep_s=reset_retry_sleep,
                )
                if not reset_ok:
                    errors.append(
                        f"episode {episode_index}: reset failed after {reset_tries_used} "
                        f"attempt(s): {reset_error}"
                    )
                    break
                if reset_tries_used > 1:
                    warnings.append(
                        f"episode {episode_index}: reset succeeded on attempt {reset_tries_used}"
                    )

                previous_command = PolicyCommand()
                for segment in schedule:
                    for segment_step_index in range(segment.hold_steps):
                        episode_step_index = int(segment.start_step) + int(segment_step_index)
                        applied_command, ramp_alpha = ramped_segment_command(
                            previous_command=previous_command,
                            target_command=segment.command,
                            segment_step_index=segment_step_index,
                            ramp_steps=command_ramp_steps,
                        )
                        setattr(env, "command", applied_command)
                        coverage.add(applied_command)
                        try:
                            transition = env.step(None)
                        except Exception as exc:
                            errors.append(
                                f"episode {episode_index} segment {segment.segment_index} "
                                f"step {episode_step_index}: step failed: {exc!r}"
                            )
                            break

                        sample, skip_reason = _sample_from_step(
                            profile=profile,
                            command_index=segment.segment_index,
                            episode_index=episode_index,
                            episode_step_index=episode_step_index,
                            global_step_index=global_step_index,
                            transition=transition,
                            command_vector=applied_command.as_list(),
                        )
                        global_step_index += 1
                        if sample is None:
                            skipped_steps += 1
                            if skip_reason is not None and len(warnings) < 50:
                                warnings.append(skip_reason)
                        else:
                            rollout_id = f"{resolved_scenario_id}:episode_{episode_index}"
                            command_ramp_name = (
                                "linear_segment_ramp"
                                if int(max(0, command_ramp_steps)) > 0
                                else "instant_segment_hold"
                            )
                            sample.update(
                                {
                                    "source_log": f"live_teacher_random_rollout:{rollout_id}",
                                    "scenario_id": resolved_scenario_id,
                                    "rollout_id": rollout_id,
                                    "mode": "teacher_policy_random_command_collection",
                                    "command_segment_index": int(segment.segment_index),
                                    "command_segment_id": segment.segment_id,
                                    "command_segment_step_index": int(segment_step_index),
                                    "command_segment_start_step": int(segment.start_step),
                                    "command_segment_hold_steps": int(segment.hold_steps),
                                    "command_ramp_steps": int(max(0, command_ramp_steps)),
                                    "command_ramp_alpha": float(ramp_alpha),
                                    "command_ramp_name": command_ramp_name,
                                    "policy_command_target": segment.command.as_list(),
                                    "desired_command": segment.command.describe(),
                                    "applied_command": applied_command.describe(),
                                    "command_schedule_seed": int(seed),
                                }
                            )
                            if scenario_metadata:
                                sample.update(scenario_metadata)
                                sample["scenario_id"] = resolved_scenario_id
                            sample.setdefault("policy_debug", {})
                            if isinstance(sample["policy_debug"], dict):
                                sample["policy_debug"].update(
                                    {
                                        "collector": "soridormi_runtime.random_teacher_dataset_collect",
                                        "command_segment": segment.describe(),
                                        "applied_command": applied_command.describe(),
                                        "target_command": segment.command.describe(),
                                        "command_ramp_alpha": float(ramp_alpha),
                                        "command_ramp_name": command_ramp_name,
                                    }
                                )
                            f.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
                            sample_count += 1

                        metrics = getattr(transition, "metrics", {}) or {}
                        if bool(metrics.get("terminated", False)):
                            terminated_episodes += 1
                            if stop_on_terminated:
                                break
                    else:
                        previous_command = segment.command
                        continue
                    break

    if sample_count == 0 and not errors:
        errors.append("No random teacher samples were collected. Check simulator connectivity.")

    dataset_sha = sha256_file(output) if output.exists() else None
    result = RandomTeacherDatasetCollectResult(
        ok=not errors,
        profile=profile,
        output_path=str(output),
        manifest_path=str(manifest),
        episodes_requested=int(episodes),
        steps_per_episode=int(steps_per_episode),
        sample_count=sample_count,
        skipped_steps=skipped_steps,
        terminated_episodes=terminated_episodes,
        dataset_sha256=dataset_sha,
        seed=int(seed),
        vx_range=vx_range.describe(),
        vy_range=vy_range.describe(),
        yaw_range=yaw_range.describe(),
        command_hold_steps=command_hold_steps.describe(),
        stop_probability=float(stop_probability),
        command_ramp_steps=int(max(0, command_ramp_steps)),
        segment_count=segment_count,
        command_coverage=coverage.describe(),
        scenario_id=resolved_scenario_id if scenario is not None else None,
        scenario_status=str(scenario_metadata.get("scenario_status")) if scenario_metadata else None,
        scenario_family=str(scenario_metadata.get("scenario_family")) if scenario_metadata else None,
        skill_id=str(scenario_metadata.get("skill_id")) if scenario_metadata.get("skill_id") else None,
        scenario_dataset_tags=list(scenario_metadata.get("scenario_dataset_tags", [])),
        task_context=dict(scenario_metadata.get("task_context", {})),
        environment_context=dict(scenario_metadata.get("environment_context", {})),
        command_space=dict(scenario_metadata.get("command_space", {})),
        errors=errors,
        warnings=warnings,
    )
    manifest_payload = asdict(result)
    manifest_payload["schema_version"] = DATASET_SCHEMA_VERSION
    manifest_payload["dataset_type"] = "soridormi.policy_supervision.random_command.v1"
    manifest_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _print_summary(result: RandomTeacherDatasetCollectResult) -> None:
    print("Soridormi random-command teacher dataset collector")
    print("==================================================")
    print(f"Profile: {result.profile}")
    print(f"Output: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Seed: {result.seed}")
    print(f"Episodes: {result.episodes_requested}")
    print(f"Steps per episode: {result.steps_per_episode}")
    print(f"Segments: {result.segment_count}")
    print(f"Command ramp steps: {result.command_ramp_steps}")
    if result.scenario_id:
        print(f"Scenario: {result.scenario_id} ({result.scenario_status})")
        if result.skill_id:
            print(f"Skill: {result.skill_id}")
    if result.command_coverage:
        coverage = result.command_coverage
        vx = coverage.get("vx", {})
        vy = coverage.get("vy", {})
        yaw = coverage.get("yaw", {})
        print(
            "Applied command coverage: "
            f"vx=[{vx.get('minimum', 0.0):.3f},{vx.get('maximum', 0.0):.3f}] "
            f"vy=[{vy.get('minimum', 0.0):.3f},{vy.get('maximum', 0.0):.3f}] "
            f"yaw=[{yaw.get('minimum', 0.0):.3f},{yaw.get('maximum', 0.0):.3f}]"
        )
    print(f"Samples: {result.sample_count}")
    print(f"Skipped steps: {result.skipped_steps}")
    print(f"Terminated episodes: {result.terminated_episodes}")
    if result.dataset_sha256:
        print(f"Dataset SHA256: {result.dataset_sha256}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:20]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


RANGE_OPTIONS = frozenset({"--vx-range", "--vy-range", "--yaw-range"})


def _normalize_negative_range_args(argv: list[str]) -> list[str]:
    """Allow ``--vx-range -0.03,0.15`` style negative CSV values.

    ``argparse`` treats values that start with ``-`` as possible options. That
    breaks common range values such as ``-0.03,0.15`` when they are supplied as
    the next token after ``--vx-range``. Convert those pairs into the equivalent
    ``--vx-range=-0.03,0.15`` spelling before parsing.
    """

    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in RANGE_OPTIONS and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and "," in value:
                normalized.append(f"{token}={value}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect random command-conditioned teacher BC samples from MuJoCo."
    )
    parser.add_argument("--profile", default="open_duck_forward")
    parser.add_argument("--output", type=Path, default=DEFAULT_RANDOM_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=1000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--scenario", default=None, help="Scenario id from the Soridormi scenario curriculum")
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=DEFAULT_SCENARIO_MANIFEST,
        help="Path to configs/scenarios/open_duck_mini_v2_scenarios.json",
    )
    parser.add_argument(
        "--allow-planned-scenario",
        action="store_true",
        help="Allow planned scenarios for metadata-only collection before MuJoCo eval promotion",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="List known scenario ids and exit")
    parser.add_argument(
        "--vx-range",
        default=None,
        help="MIN,MAX random forward velocity range; defaults to scenario command_space or -0.03,0.15",
    )
    parser.add_argument(
        "--vy-range",
        default=None,
        help="MIN,MAX random lateral velocity range; defaults to scenario command_space or -0.03,0.03",
    )
    parser.add_argument(
        "--yaw-range",
        default=None,
        help="MIN,MAX random yaw velocity range; defaults to scenario command_space or -0.20,0.20",
    )
    parser.add_argument("--command-hold-steps", default="80,250", help="MIN,MAX steps per command segment")
    parser.add_argument(
        "--command-ramp-steps",
        type=int,
        default=20,
        help="control steps used to ramp from the previous command to each new target command",
    )
    parser.add_argument("--stop-probability", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-height", type=float, default=0.30)
    parser.add_argument("--fall-height", type=float, default=0.14)
    parser.add_argument("--min-upright", type=float, default=0.65)
    parser.add_argument("--forward-velocity-sigma", type=float, default=0.20)
    parser.add_argument("--continue-after-terminated", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument(
        "--reset-attempts",
        type=int,
        default=5,
        help="Retry transient simulator reset failures this many times per episode (default: 5)",
    )
    parser.add_argument(
        "--reset-retry-sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between simulator reset attempts (default: 0.25)",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _range_text_from_args(
    *,
    cli_value: str | None,
    scenario: ScenarioDefinition | None,
    scenario_field: str,
    default: str,
) -> str:
    if cli_value is not None:
        return cli_value
    if scenario is not None:
        return scenario.command_range_text(scenario_field)
    return default


def _print_scenario_list(scenario_manifest: Path, *, as_json: bool) -> None:
    scenarios = list_scenarios(scenario_manifest)
    payload = [
        {
            "id": item.id,
            "title": item.title,
            "status": item.status,
            "family": item.family,
            "priority": item.priority,
            "skills": item.skills,
        }
        for item in scenarios
    ]
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for item in payload:
        print(f"{item['id']}\t{item['status']}\t{item['family']}\t{','.join(item['skills'])}")


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = _build_parser().parse_args(_normalize_negative_range_args(raw_argv))
    if args.list_scenarios:
        _print_scenario_list(args.scenario_manifest, as_json=args.json)
        return 0

    scenario = None
    scenario_warnings: list[str] = []
    if args.scenario:
        try:
            scenario = get_scenario_definition(args.scenario, args.scenario_manifest)
            scenario_warnings = validate_scenario_for_teacher_collection(
                scenario,
                allow_planned=args.allow_planned_scenario,
            )
        except ScenarioCurriculumError as exc:
            print(f"Scenario error: {exc}", file=sys.stderr)
            return 2

    reward_config = WalkingRewardConfig(
        target_height=args.target_height,
        fall_height=args.fall_height,
        min_upright=args.min_upright,
        forward_velocity_sigma=args.forward_velocity_sigma,
    )
    result = collect_random_teacher_dataset(
        profile=args.profile,
        output_path=args.output,
        manifest_path=args.manifest,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        vx_range=_parse_range(
            _range_text_from_args(
                cli_value=args.vx_range,
                scenario=scenario,
                scenario_field="vx_mps",
                default="-0.03,0.15",
            ),
            name="--vx-range",
        ),
        vy_range=_parse_range(
            _range_text_from_args(
                cli_value=args.vy_range,
                scenario=scenario,
                scenario_field="vy_mps",
                default="-0.03,0.03",
            ),
            name="--vy-range",
        ),
        yaw_range=_parse_range(
            _range_text_from_args(
                cli_value=args.yaw_range,
                scenario=scenario,
                scenario_field="yaw_radps",
                default="-0.20,0.20",
            ),
            name="--yaw-range",
        ),
        command_hold_steps=_parse_hold_range(args.command_hold_steps),
        stop_probability=args.stop_probability,
        command_ramp_steps=max(0, int(args.command_ramp_steps)),
        seed=args.seed,
        reward_config=reward_config,
        host=args.host,
        port=args.port,
        stop_on_terminated=not args.continue_after_terminated,
        reset_on_start=not args.no_reset,
        reset_attempts=args.reset_attempts,
        reset_retry_sleep=args.reset_retry_sleep,
        scenario=scenario,
        initial_warnings=scenario_warnings,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
