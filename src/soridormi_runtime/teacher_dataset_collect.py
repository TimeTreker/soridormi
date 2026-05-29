from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.rl_finetune_env import ACTION_SIZE, ResidualActionConfig, RlFineTuneEnv
from soridormi_runtime.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_OBSERVATION_SIZE,
    sha256_file,
)
from soridormi_runtime.walking_reward import WalkingRewardConfig


DEFAULT_OUTPUT = Path("/data/training_datasets/teacher_policy_live.jsonl")


@dataclass(frozen=True)
class TeacherDatasetCollectResult:
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
    command: dict[str, float] = field(default_factory=dict)
    commands: list[dict[str, float]] = field(default_factory=list)
    command_count: int = 1
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _float_list(values: Any, *, size: int, field_name: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (1, size):
        arr = arr.reshape(size)
    if arr.shape != (size,):
        raise ValueError(f"{field_name} must have shape ({size},) or (1, {size}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must contain only finite values")
    return [float(x) for x in arr.tolist()]


def _manifest_path_for(output_path: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    suffix = output_path.suffix or ".jsonl"
    return output_path.with_suffix(suffix + ".manifest.json")


def _command_id(command_vector: list[float] | None) -> str:
    if not command_vector:
        return "none"
    safe = []
    for value in command_vector:
        item = float(value)
        safe.append(f"{item:+.3f}".replace("+", "p").replace("-", "m").replace(".", "_"))
    return "_".join(safe)


def _sample_from_step(
    *,
    profile: str,
    command_index: int,
    episode_index: int,
    episode_step_index: int,
    global_step_index: int,
    transition: Any,
    command_vector: list[float] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    observation = getattr(transition, "observation", None)
    if observation is None:
        return (
            None,
            f"episode {episode_index} step {episode_step_index}: "
            "teacher policy did not expose observation",
        )
    try:
        observation_values = _float_list(
            observation,
            size=DEFAULT_OBSERVATION_SIZE,
            field_name="observation",
        )
        action_values = _float_list(
            getattr(transition, "teacher_action"),
            size=ACTION_SIZE,
            field_name="teacher_action",
        )
    except Exception as exc:
        return None, f"episode {episode_index} step {episode_step_index}: {exc}"

    metrics = getattr(transition, "metrics", {}) or {}
    command_id = _command_id(command_vector)
    scenario_id = f"{profile}:command_{command_index}_{command_id}"
    rollout_id = f"{scenario_id}:episode_{episode_index}"
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "sample_type": "soridormi.policy_supervision.v1",
        "source_log": f"live_teacher_rollout:{rollout_id}",
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "command_id": command_id,
        "step_index": int(global_step_index),
        "command_index": int(command_index),
        "episode_index": int(episode_index),
        "episode_step_index": int(episode_step_index),
        "robot_time": float(getattr(transition, "state_time", 0.0)),
        "next_robot_time": float(getattr(transition, "next_state_time", 0.0)),
        "mode": "teacher_policy_live_collection",
        "backend": "sim",
        "observation": observation_values,
        "action": action_values,
        "raw_action": action_values,
        "policy_command": command_vector,
        "motor_command": getattr(transition, "motor_command", None),
        "state": getattr(transition, "state_before", None),
        "next_state": getattr(transition, "state_after", None),
        "policy_debug": {
            "profile": profile,
            "collector": "soridormi_runtime.teacher_dataset_collect",
            "reward": metrics.get("reward"),
            "reward_terms": metrics.get("reward_terms"),
            "terminated": bool(metrics.get("terminated", False)),
        },
    }, None


def collect_teacher_dataset(
    *,
    profile: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path | None = None,
    episodes: int = 1,
    steps_per_episode: int = 1000,
    command: PolicyCommand | None = None,
    commands: Iterable[PolicyCommand] | None = None,
    reward_config: WalkingRewardConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 5555,
    stop_on_terminated: bool = True,
    reset_on_start: bool = True,
    env_factory: Callable[..., Any] | None = None,
) -> TeacherDatasetCollectResult:
    """Collect supervised teacher-policy samples directly from the live MuJoCo sim.

    This bypasses the older two-step "run a policy log, then export from the log"
    workflow. Each written JSONL row is already compatible with
    ``prepare_training_dataset.py`` and the linear/neural BC trainers.
    """

    output = Path(output_path)
    manifest = _manifest_path_for(
        output,
        Path(manifest_path) if manifest_path is not None else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    command_list = list(commands) if commands is not None else [command or PolicyCommand.from_env()]
    if not command_list:
        command_list = [command or PolicyCommand.from_env()]
    reward_cfg = reward_config or WalkingRewardConfig()
    errors: list[str] = []
    warnings: list[str] = []
    sample_count = 0
    skipped_steps = 0
    terminated_episodes = 0
    global_step_index = 0

    factory = env_factory or RlFineTuneEnv
    with output.open("w", encoding="utf-8") as f:
        for command_index, command_obj in enumerate(command_list):
            try:
                env = factory(
                    profile=profile,
                    host=host,
                    port=port,
                    command=command_obj,
                    residual_config=ResidualActionConfig(residual_scale=0.0),
                    reward_config=reward_cfg,
                    reset_on_start=reset_on_start,
                )
            except Exception as exc:
                errors.append(
                    f"command {command_index}: could not construct training environment: {exc!r}"
                )
                continue

            for episode_index in range(max(0, int(episodes))):
                try:
                    env.reset()
                except Exception as exc:
                    errors.append(
                        f"command {command_index} episode {episode_index}: reset failed: {exc!r}"
                    )
                    break

                for episode_step_index in range(max(0, int(steps_per_episode))):
                    try:
                        transition = env.step(None)
                    except Exception as exc:
                        errors.append(
                            "command "
                            f"{command_index} episode {episode_index} "
                            f"step {episode_step_index}: step failed: {exc!r}"
                        )
                        break

                    sample, skip_reason = _sample_from_step(
                        profile=profile,
                        command_index=command_index,
                        episode_index=episode_index,
                        episode_step_index=episode_step_index,
                        global_step_index=global_step_index,
                        transition=transition,
                        command_vector=command_obj.as_list(),
                    )
                    global_step_index += 1
                    if sample is None:
                        skipped_steps += 1
                        if skip_reason is not None and len(warnings) < 50:
                            warnings.append(skip_reason)
                    else:
                        f.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
                        sample_count += 1

                    metrics = getattr(transition, "metrics", {}) or {}
                    if bool(metrics.get("terminated", False)):
                        terminated_episodes += 1
                        if stop_on_terminated:
                            break

    if sample_count == 0 and not errors:
        errors.append(
            "No teacher samples were collected. "
            "Check simulator connectivity and policy observation exposure."
        )

    dataset_sha = sha256_file(output) if output.exists() else None
    result = TeacherDatasetCollectResult(
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
        command=command_list[0].describe(),
        commands=[item.describe() for item in command_list],
        command_count=len(command_list),
        errors=errors,
        warnings=warnings,
    )
    manifest_payload = asdict(result)
    manifest_payload["schema_version"] = DATASET_SCHEMA_VERSION
    manifest_payload["dataset_type"] = "soridormi.policy_supervision.v1"
    manifest_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _print_summary(result: TeacherDatasetCollectResult) -> None:
    print("Soridormi live teacher dataset collector")
    print("=========================================")
    print(f"Profile: {result.profile}")
    print(f"Output: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Command count: {result.command_count}")
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


def _parse_float_values(text: str | None, *, default: float) -> list[float]:
    if text is None or str(text).strip() == "":
        return [float(default)]
    values: list[float] = []
    for chunk in str(text).split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _command_grid_from_args(args: argparse.Namespace) -> list[PolicyCommand]:
    x_values = _parse_float_values(args.command_x_values, default=args.command_x)
    y_values = _parse_float_values(args.command_y_values, default=args.command_y)
    yaw_values = _parse_float_values(args.command_yaw_values, default=args.command_yaw)
    return [
        PolicyCommand(
            x_velocity=x,
            y_velocity=y,
            yaw_velocity=yaw,
            neck_pitch=args.neck_pitch,
            head_pitch=args.head_pitch,
            head_yaw=args.head_yaw,
            head_roll=args.head_roll,
        )
        for x, y, yaw in product(x_values, y_values, yaw_values)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect BC training samples by rolling out a teacher policy in MuJoCo."
    )
    parser.add_argument(
        "--profile",
        default="open_duck_forward",
        help="Teacher policy profile name or YAML path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output supervised JSONL dataset",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=1000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--command-x", type=float, default=0.0)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument(
        "--command-x-values",
        default=None,
        help=(
            "Comma-separated forward command grid, e.g. "
            "'0.00,0.05,0.10,0.15'. Overrides --command-x."
        ),
    )
    parser.add_argument(
        "--command-y-values",
        default=None,
        help="Comma-separated lateral command grid. Overrides --command-y.",
    )
    parser.add_argument(
        "--command-yaw-values",
        default=None,
        help="Comma-separated yaw command grid. Overrides --command-yaw.",
    )
    parser.add_argument("--neck-pitch", type=float, default=0.0)
    parser.add_argument("--head-pitch", type=float, default=0.0)
    parser.add_argument("--head-yaw", type=float, default=0.0)
    parser.add_argument("--head-roll", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=0.30)
    parser.add_argument("--fall-height", type=float, default=0.14)
    parser.add_argument("--min-upright", type=float, default=0.65)
    parser.add_argument("--forward-velocity-sigma", type=float, default=0.20)
    parser.add_argument("--continue-after-terminated", action="store_true")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not call simulator reset at episode starts",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command_grid = _command_grid_from_args(args)
    reward_config = WalkingRewardConfig(
        target_height=args.target_height,
        fall_height=args.fall_height,
        min_upright=args.min_upright,
        forward_velocity_sigma=args.forward_velocity_sigma,
    )
    result = collect_teacher_dataset(
        profile=args.profile,
        output_path=args.output,
        manifest_path=args.manifest,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        commands=command_grid,
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
