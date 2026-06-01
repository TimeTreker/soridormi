from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.random_teacher_dataset_collect import (
    CommandRange,
    HoldStepRange,
    _normalize_negative_range_args,
    _range_text_from_args,
    collect_random_teacher_dataset,
    generate_random_command_schedule,
    ramped_segment_command,
)
from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.scenario_curriculum import get_scenario_definition
from soridormi_runtime.rl_finetune_env import RlFineTuneStep
from soridormi_runtime.training_dataset_prepare import load_and_validate_dataset


class FakeRandomTeacherEnv:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.index = 0
        self.reset_count = 0
        self.seen_commands: list[tuple[float, float, float]] = []

    def reset(self) -> object:
        self.index = 0
        self.reset_count += 1
        return object()

    def step(self, residual_action=None) -> RlFineTuneStep:
        idx = self.index
        self.index += 1
        cmd = self.command
        if cmd is not None:
            self.seen_commands.append((cmd.x_velocity, cmd.y_velocity, cmd.yaw_velocity))
        return RlFineTuneStep(
            step_index=idx,
            state_time=0.02 * idx,
            next_state_time=0.02 * (idx + 1),
            observation=[float(idx)] * 101,
            teacher_action=[0.1] * 14,
            residual_action=[0.0] * 14,
            final_action=[0.1] * 14,
            motor_command={"joint_count": 14, "names": [f"joint_{i}" for i in range(14)]},
            metrics={"reward": 1.0, "reward_terms": {}, "terminated": False},
            state_before={"time": 0.02 * idx},
            state_after={"time": 0.02 * (idx + 1)},
        )




def test_ramped_segment_command_interpolates_velocity_commands() -> None:
    start = PolicyCommand(x_velocity=0.0, y_velocity=0.0, yaw_velocity=0.0)
    target = PolicyCommand(x_velocity=0.2, y_velocity=-0.04, yaw_velocity=0.3)

    first, alpha_first = ramped_segment_command(
        previous_command=start,
        target_command=target,
        segment_step_index=0,
        ramp_steps=4,
    )
    last, alpha_last = ramped_segment_command(
        previous_command=start,
        target_command=target,
        segment_step_index=3,
        ramp_steps=4,
    )

    assert alpha_first == 0.25
    assert first.x_velocity == 0.05
    assert first.y_velocity == -0.01
    assert first.yaw_velocity == 0.075
    assert alpha_last == 1.0
    assert last.x_velocity == target.x_velocity
    assert last.y_velocity == target.y_velocity
    assert last.yaw_velocity == target.yaw_velocity


def test_ramped_segment_command_can_be_disabled() -> None:
    target = PolicyCommand(x_velocity=0.2)
    command, alpha = ramped_segment_command(
        previous_command=PolicyCommand(),
        target_command=target,
        segment_step_index=0,
        ramp_steps=0,
    )

    assert alpha == 1.0
    assert command.x_velocity == target.x_velocity

def test_generate_random_command_schedule_is_seeded_and_covers_episode() -> None:
    import numpy as np

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    kwargs = dict(
        steps_per_episode=25,
        vx_range=CommandRange(-0.03, 0.15),
        vy_range=CommandRange(-0.03, 0.03),
        yaw_range=CommandRange(-0.20, 0.20),
        command_hold_steps=HoldStepRange(4, 8),
        stop_probability=0.0,
    )

    schedule_a = generate_random_command_schedule(rng=rng_a, **kwargs)
    schedule_b = generate_random_command_schedule(rng=rng_b, **kwargs)

    assert [item.describe() for item in schedule_a] == [item.describe() for item in schedule_b]
    assert schedule_a[0].start_step == 0
    assert schedule_a[-1].end_step_exclusive == 25
    assert all(4 <= item.hold_steps <= 8 for item in schedule_a[:-1])
    assert all(-0.03 <= item.command.x_velocity <= 0.15 for item in schedule_a)
    assert all(-0.03 <= item.command.y_velocity <= 0.03 for item in schedule_a)
    assert all(-0.20 <= item.command.yaw_velocity <= 0.20 for item in schedule_a)


def test_collect_random_teacher_dataset_writes_segment_metadata(tmp_path: Path) -> None:
    output = tmp_path / "random_teacher.jsonl"
    result = collect_random_teacher_dataset(
        profile="teacher_profile",
        output_path=output,
        episodes=2,
        steps_per_episode=12,
        vx_range=CommandRange(-0.03, 0.15),
        vy_range=CommandRange(-0.03, 0.03),
        yaw_range=CommandRange(-0.20, 0.20),
        command_hold_steps=HoldStepRange(3, 4),
        stop_probability=0.0,
        command_ramp_steps=2,
        seed=7,
        env_factory=FakeRandomTeacherEnv,
    )

    assert result.ok
    assert result.sample_count == 24
    assert result.segment_count >= 6
    assert output.exists()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sample_type"] == "soridormi.policy_supervision.v1"
    assert rows[0]["mode"] == "teacher_policy_random_command_collection"
    assert rows[0]["source_log"].startswith("live_teacher_random_rollout:")
    assert rows[0]["scenario_id"] == "teacher_profile:random_walk_seed_7"
    assert rows[0]["rollout_id"].endswith("episode_0")
    assert "command_segment_index" in rows[0]
    assert "command_segment_id" in rows[0]
    assert "command_segment_hold_steps" in rows[0]
    assert rows[0]["command_ramp_steps"] == 2
    assert 0.0 <= rows[0]["command_ramp_alpha"] <= 1.0
    assert "policy_command_target" in rows[0]
    assert rows[-1]["rollout_id"].endswith("episode_1")

    unique_commands = {tuple(row["policy_command"][:3]) for row in rows}
    assert len(unique_commands) > 2

    samples, summary = load_and_validate_dataset(output)
    assert summary.ok
    assert len(samples) == 24


def test_collect_random_teacher_dataset_manifest_records_ranges(tmp_path: Path) -> None:
    output = tmp_path / "random_teacher.jsonl"
    result = collect_random_teacher_dataset(
        profile="teacher_profile",
        output_path=output,
        episodes=1,
        steps_per_episode=5,
        vx_range=CommandRange(0.0, 0.1),
        vy_range=CommandRange(-0.01, 0.01),
        yaw_range=CommandRange(-0.2, 0.2),
        command_hold_steps=HoldStepRange(2, 3),
        command_ramp_steps=2,
        seed=1234,
        env_factory=FakeRandomTeacherEnv,
    )

    manifest = Path(result.manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset_type"] == "soridormi.policy_supervision.random_command.v1"
    assert payload["seed"] == 1234
    assert payload["vx_range"] == {"minimum": 0.0, "maximum": 0.1}
    assert payload["command_hold_steps"] == {"minimum": 2, "maximum": 3}
    assert payload["command_ramp_steps"] == 2
    assert payload["command_coverage"]["count"] == result.sample_count
    assert payload["command_coverage"]["vx"]["maximum"] <= 0.1




def test_collect_random_teacher_dataset_writes_scenario_context_metadata(tmp_path: Path) -> None:
    scenario = get_scenario_definition("flat_walk_varied_speed_v1")
    output = tmp_path / "scenario_teacher.jsonl"
    result = collect_random_teacher_dataset(
        profile="teacher_profile",
        output_path=output,
        episodes=1,
        steps_per_episode=8,
        vx_range=CommandRange(*scenario.command_range("vx_mps")),
        vy_range=CommandRange(*scenario.command_range("vy_mps")),
        yaw_range=CommandRange(*scenario.command_range("yaw_radps")),
        command_hold_steps=HoldStepRange(3, 4),
        stop_probability=0.0,
        command_ramp_steps=2,
        seed=9,
        env_factory=FakeRandomTeacherEnv,
        scenario=scenario,
    )

    assert result.ok
    assert result.scenario_id == "flat_walk_varied_speed_v1"
    assert result.skill_id == "walk_velocity"
    assert result.vx_range == {"minimum": -0.03, "maximum": 0.25}
    assert result.scenario_status == "mujoco_registry_ready"

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    row = rows[0]
    assert row["scenario_id"] == "flat_walk_varied_speed_v1"
    assert row["scenario_status"] == "mujoco_registry_ready"
    assert row["scenario_family"] == "locomotion_flat"
    assert row["skill_id"] == "walk_velocity"
    assert row["task_context"]["skill_family"] == "locomotion"
    assert row["environment_context"]["terrain_type"] == "flat"
    assert row["command_space"]["vx_mps"] == [-0.03, 0.25]
    assert row["command_ramp_name"] == "linear_segment_ramp"
    assert row["desired_command"] == row["policy_debug"]["target_command"]
    assert row["applied_command"] == row["policy_debug"]["applied_command"]

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["scenario_id"] == "flat_walk_varied_speed_v1"
    assert manifest["skill_id"] == "walk_velocity"
    assert manifest["task_context"]["skill_family"] == "locomotion"


def test_scenario_range_defaults_are_used_when_cli_range_is_absent() -> None:
    scenario = get_scenario_definition("flat_walk_varied_speed_v1")

    assert (
        _range_text_from_args(
            cli_value=None,
            scenario=scenario,
            scenario_field="vx_mps",
            default="-0.03,0.15",
        )
        == "-0.03,0.25"
    )
    assert (
        _range_text_from_args(
            cli_value="0.01,0.02",
            scenario=scenario,
            scenario_field="vx_mps",
            default="-0.03,0.15",
        )
        == "0.01,0.02"
    )

def test_negative_range_args_accept_two_token_cli_form() -> None:
    normalized = _normalize_negative_range_args(
        [
            "--profile",
            "open_duck_forward",
            "--vx-range",
            "-0.03,0.15",
            "--vy-range",
            "-0.03,0.03",
            "--yaw-range",
            "-0.20,0.20",
            "--command-hold-steps",
            "80,250",
        ]
    )

    assert "--vx-range=-0.03,0.15" in normalized
    assert "--vy-range=-0.03,0.03" in normalized
    assert "--yaw-range=-0.20,0.20" in normalized
    assert "-0.03,0.15" not in normalized
    assert "--command-hold-steps" in normalized
    assert "80,250" in normalized


def test_negative_range_args_keep_equals_form() -> None:
    normalized = _normalize_negative_range_args(["--vx-range=-0.03,0.15", "--yaw-range=0.0,0.2"])

    assert normalized == ["--vx-range=-0.03,0.15", "--yaw-range=0.0,0.2"]

def test_collect_random_teacher_dataset_script_documents_mujoco_viewer_flags() -> None:
    import subprocess

    proc = subprocess.run(
        ["bash", "scripts/collect_random_teacher_dataset.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--backend NAME" in proc.stdout
    assert "--viewer" in proc.stdout
    assert "--no-viewer" in proc.stdout
    assert "--command-ramp-steps" in proc.stdout
    assert "--scenario ID" in proc.stdout
    assert "--list-scenarios" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera" in proc.stdout
