from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.rl_finetune_env import ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.teacher_dataset_collect import _manifest_path_for, _sample_from_step
from soridormi_runtime.training_dataset import DATASET_SCHEMA_VERSION, sha256_file
from soridormi_runtime.walking_reward import WalkingRewardConfig


DEFAULT_RANDOM_OUTPUT = Path("/data/training_datasets/teacher_policy_random_walk.jsonl")


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
    segment_count: int = 0
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
    seed: int = 123,
    reward_config: WalkingRewardConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 5555,
    stop_on_terminated: bool = True,
    reset_on_start: bool = True,
    env_factory: Callable[..., Any] | None = None,
) -> RandomTeacherDatasetCollectResult:
    """Collect teacher BC samples under random piecewise velocity commands.

    Unlike ``teacher_dataset_collect.collect_teacher_dataset``, this collector
    changes the policy command several times inside each episode. This is the
    M6B data path for command-conditioned free walking: one rollout contains
    stand, stop, turn, curve, lateral, forward, and backward command changes
    sampled from conservative ranges.
    """

    output = Path(output_path)
    manifest = _manifest_path_for(output, Path(manifest_path) if manifest_path is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    reward_cfg = reward_config or WalkingRewardConfig()
    rng = np.random.default_rng(int(seed))
    factory = env_factory or RlFineTuneEnv
    errors: list[str] = []
    warnings: list[str] = []
    sample_count = 0
    skipped_steps = 0
    terminated_episodes = 0
    segment_count = 0
    global_step_index = 0

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

                try:
                    env.reset()
                except Exception as exc:
                    errors.append(f"episode {episode_index}: reset failed: {exc!r}")
                    break

                for segment in schedule:
                    setattr(env, "command", segment.command)
                    for segment_step_index in range(segment.hold_steps):
                        episode_step_index = int(segment.start_step) + int(segment_step_index)
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
                            command_vector=segment.command.as_list(),
                        )
                        global_step_index += 1
                        if sample is None:
                            skipped_steps += 1
                            if skip_reason is not None and len(warnings) < 50:
                                warnings.append(skip_reason)
                        else:
                            scenario_id = f"{profile}:random_walk_seed_{int(seed)}"
                            rollout_id = f"{scenario_id}:episode_{episode_index}"
                            sample.update(
                                {
                                    "source_log": f"live_teacher_random_rollout:{rollout_id}",
                                    "scenario_id": scenario_id,
                                    "rollout_id": rollout_id,
                                    "mode": "teacher_policy_random_command_collection",
                                    "command_segment_index": int(segment.segment_index),
                                    "command_segment_id": segment.segment_id,
                                    "command_segment_step_index": int(segment_step_index),
                                    "command_segment_start_step": int(segment.start_step),
                                    "command_segment_hold_steps": int(segment.hold_steps),
                                    "command_schedule_seed": int(seed),
                                }
                            )
                            sample.setdefault("policy_debug", {})
                            if isinstance(sample["policy_debug"], dict):
                                sample["policy_debug"].update(
                                    {
                                        "collector": "soridormi_runtime.random_teacher_dataset_collect",
                                        "command_segment": segment.describe(),
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
        segment_count=segment_count,
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
    parser.add_argument("--vx-range", default="-0.03,0.15", help="MIN,MAX random forward velocity range")
    parser.add_argument("--vy-range", default="-0.03,0.03", help="MIN,MAX random lateral velocity range")
    parser.add_argument("--yaw-range", default="-0.20,0.20", help="MIN,MAX random yaw velocity range")
    parser.add_argument("--command-hold-steps", default="80,250", help="MIN,MAX steps per command segment")
    parser.add_argument("--stop-probability", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-height", type=float, default=0.30)
    parser.add_argument("--fall-height", type=float, default=0.14)
    parser.add_argument("--min-upright", type=float, default=0.65)
    parser.add_argument("--forward-velocity-sigma", type=float, default=0.20)
    parser.add_argument("--continue-after-terminated", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = _build_parser().parse_args(_normalize_negative_range_args(raw_argv))
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
        vx_range=_parse_range(args.vx_range, name="--vx-range"),
        vy_range=_parse_range(args.vy_range, name="--vy-range"),
        yaw_range=_parse_range(args.yaw_range, name="--yaw-range"),
        command_hold_steps=_parse_hold_range(args.command_hold_steps),
        stop_probability=args.stop_probability,
        seed=args.seed,
        reward_config=reward_config,
        host=args.host,
        port=args.port,
        stop_on_terminated=not args.continue_after_terminated,
        reset_on_start=not args.no_reset,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
