from __future__ import annotations

import json
from pathlib import Path

from soridormi_runtime.policy_command import PolicyCommand
from soridormi_runtime.rl_finetune_env import RlFineTuneStep
from soridormi_runtime.teacher_dataset_collect import collect_teacher_dataset
from soridormi_runtime.training_dataset_prepare import load_and_validate_dataset


class FakeTeacherEnv:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.index = 0
        self.reset_count = 0

    def reset(self) -> object:
        self.index = 0
        self.reset_count += 1
        return object()

    def step(self, residual_action=None) -> RlFineTuneStep:
        idx = self.index
        self.index += 1
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


def test_collect_teacher_dataset_writes_preparable_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "teacher.jsonl"
    result = collect_teacher_dataset(
        profile="teacher_profile",
        output_path=output,
        episodes=2,
        steps_per_episode=3,
        command=PolicyCommand(x_velocity=0.15),
        env_factory=FakeTeacherEnv,
    )

    assert result.ok
    assert result.sample_count == 6
    assert result.skipped_steps == 0
    assert output.exists()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sample_type"] == "soridormi.policy_supervision.v1"
    assert rows[0]["policy_command"][0] == 0.15
    assert rows[0]["scenario_id"].startswith("teacher_profile:command_")
    assert rows[0]["rollout_id"].endswith("episode_0")
    assert rows[0]["source_log"].startswith("live_teacher_rollout:")
    assert rows[0]["observation"] == [0.0] * 101
    assert rows[-1]["episode_index"] == 1
    assert rows[-1]["rollout_id"].endswith("episode_1")

    samples, summary = load_and_validate_dataset(output)
    assert summary.ok
    assert len(samples) == 6


def test_collect_teacher_dataset_reports_missing_observation(tmp_path: Path) -> None:
    class MissingObservationEnv(FakeTeacherEnv):
        def step(self, residual_action=None) -> RlFineTuneStep:
            step = super().step(residual_action)
            return RlFineTuneStep(
                step_index=step.step_index,
                state_time=step.state_time,
                next_state_time=step.next_state_time,
                observation=None,
                teacher_action=step.teacher_action,
                residual_action=step.residual_action,
                final_action=step.final_action,
                motor_command=step.motor_command,
                metrics=step.metrics,
                state_before=step.state_before,
                state_after=step.state_after,
            )

    result = collect_teacher_dataset(
        profile="teacher_profile",
        output_path=tmp_path / "missing.jsonl",
        episodes=1,
        steps_per_episode=1,
        env_factory=MissingObservationEnv,
    )

    assert not result.ok
    assert result.sample_count == 0
    assert result.skipped_steps == 1
    assert "No teacher samples" in result.errors[0]


def test_collect_teacher_dataset_supports_command_grid_metadata(tmp_path: Path) -> None:
    output = tmp_path / "teacher_grid.jsonl"
    result = collect_teacher_dataset(
        profile="teacher_profile",
        output_path=output,
        episodes=2,
        steps_per_episode=2,
        commands=[
            PolicyCommand(x_velocity=0.05),
            PolicyCommand(x_velocity=0.15, yaw_velocity=0.2),
        ],
        env_factory=FakeTeacherEnv,
    )

    assert result.ok
    assert result.command_count == 2
    assert result.sample_count == 8
    assert len(result.commands) == 2

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["command_index"] for row in rows} == {0, 1}
    assert {row["scenario_id"] for row in rows} == {
        "teacher_profile:command_0_p0_050_p0_000_p0_000_p0_000_p0_000_p0_000_p0_000",
        "teacher_profile:command_1_p0_150_p0_000_p0_200_p0_000_p0_000_p0_000_p0_000",
    }
    assert len({row["rollout_id"] for row in rows}) == 4
    assert rows[0]["policy_command"][0] == 0.05
    assert rows[-1]["policy_command"][0] == 0.15
    assert rows[-1]["policy_command"][2] == 0.2
