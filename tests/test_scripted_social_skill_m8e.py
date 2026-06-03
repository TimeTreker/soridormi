from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from soridormi_api import IMUState, JointState, RobotState
from soridormi_runtime.scripted_head_skill import (
    HEAD_JOINT_NAMES,
    command_positions_by_name,
    effective_duration_for_trajectory,
    execute_scripted_head_plan,
    interpolate_positions,
    limit_head_target_velocity,
    motor_command_from_targets,
    plan_head_pose_trajectory,
    resolve_keyframe_targets_for_execution,
    smoothstep,
    target_positions_for_segment_step,
    validate_scripted_head_plan,
)
import soridormi_runtime.scripted_head_skill as scripted_head_skill
from soridormi_runtime.skill_execution import SkillExecutionError, SkillExecutionRegistry, plan_shell_exports
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest


def _registry() -> SkillExecutionRegistry:
    return SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))


def _state() -> RobotState:
    names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]
    positions = [float(index) * 0.01 for index, _ in enumerate(names)]
    return RobotState(
        time=0.0,
        joints=JointState(
            names=names,
            positions=positions,
            velocities=[0.0] * len(names),
            torques=[0.0] * len(names),
        ),
        imu=IMUState(),
    )


def test_look_direction_manifest_skill_is_experimental_and_scripted() -> None:
    skill = _registry().skills["look_direction"]

    assert skill["status"] == "available_sim_experimental"
    assert skill["execution"] == "scripted_keyframe"
    assert set(skill["required_actuator_groups"]) == {"head_neck"}


def test_look_direction_plan_targets_only_head_neck_keyframe() -> None:
    plan = _registry().create_plan(
        "look_direction",
        {"head_yaw_rad": 0.25, "head_pitch_rad": -0.1, "duration_s": 1.2},
    )

    assert plan.commands == ()
    assert len(plan.keyframes) == 1
    keyframe = plan.keyframes[0]
    assert keyframe.duration_s == pytest.approx(1.2)
    assert set(keyframe.positions_by_name) == set(HEAD_JOINT_NAMES)
    assert keyframe.positions_by_name["head_yaw"] == pytest.approx(0.25)
    assert keyframe.positions_by_name["head_pitch"] == pytest.approx(-0.1)


def test_velocity_shell_exports_reject_scripted_keyframe_plans() -> None:
    plan = _registry().create_plan("look_direction", {"head_yaw_rad": 0.2})

    with pytest.raises(SkillExecutionError, match="exactly one segment"):
        plan_shell_exports(plan)


def test_interpolation_is_smooth_and_preserves_non_targets() -> None:
    start = {"left_hip_yaw": 0.4, "head_yaw": 0.0, "head_pitch": 0.0}
    target = {"head_yaw": 0.2, "head_pitch": -0.1}
    halfway = interpolate_positions(start, target, 0.5)

    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert halfway["left_hip_yaw"] == pytest.approx(0.4)
    assert halfway["head_yaw"] == pytest.approx(0.1)
    assert halfway["head_pitch"] == pytest.approx(-0.05)


def test_motor_command_preserves_joint_order_and_only_overrides_targets() -> None:
    state = _state()
    command = motor_command_from_targets(state, {"head_yaw": 0.3, "head_pitch": -0.2})

    assert command.names == state.joints.names
    by_name = dict(zip(command.names, command.positions))
    current = dict(zip(state.joints.names, state.joints.positions))
    assert by_name["head_yaw"] == pytest.approx(0.3)
    assert by_name["head_pitch"] == pytest.approx(-0.2)
    assert by_name["left_hip_yaw"] == pytest.approx(current["left_hip_yaw"])
    assert by_name["right_ankle"] == pytest.approx(current["right_ankle"])


def test_motor_command_preserves_actuator_ctrl_for_non_head_joints() -> None:
    base = _state()
    actuator_ctrl = [1.0 + 0.01 * index for index in range(len(base.joints.names))]
    state = base.model_copy(update={"actuator_ctrl": actuator_ctrl})
    command = motor_command_from_targets(state, {"head_yaw": 0.3})

    controls = command_positions_by_name(state)
    by_name = dict(zip(command.names, command.positions))
    qpos = dict(zip(state.joints.names, state.joints.positions))
    assert controls["left_hip_pitch"] != pytest.approx(qpos["left_hip_pitch"])
    assert by_name["left_hip_pitch"] == pytest.approx(controls["left_hip_pitch"])
    assert by_name["right_ankle"] == pytest.approx(controls["right_ankle"])
    assert by_name["head_yaw"] == pytest.approx(0.3)


def test_execute_scripted_head_plan_dry_run_never_connects_to_sim() -> None:
    plan = _registry().create_plan(
        "look_direction",
        {"head_yaw_rad": 0.2, "head_pitch_rad": -0.05, "duration_s": 0.5},
    )
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0)

    assert result.executed is False
    assert result.steps == 10
    assert result.target_positions_by_name["head_yaw"] == pytest.approx(0.2)


def test_validate_scripted_head_plan_rejects_locomotion_plan() -> None:
    plan = _registry().create_plan("walk_velocity", {"vx_mps": 0.1})

    with pytest.raises(SkillExecutionError, match="unsupported scripted head skill"):
        validate_scripted_head_plan(plan)


def test_scripted_head_cli_dry_run_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "look_direction",
            "--args",
            json.dumps({"head_yaw_rad": 0.15, "duration_s": 0.4}),
            "--backend",
            "mujoco",
            "--control-hz",
            "25",
            "--dry-run",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["result"]["executed"] is False
    assert payload["result"]["steps"] == 10
    assert payload["plan"]["keyframes"][0]["positions_by_name"]["head_yaw"] == pytest.approx(0.15)


def test_scripted_social_shell_wrapper_help() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_scripted_social_skill_in_sim.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert "run_sim_server.sh --backend mujoco" in proc.stdout
    assert "look_direction" in proc.stdout


def test_scripted_head_dry_run_imports_without_pyzmq() -> None:
    script = r'''
import builtins
import json

real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "zmq" or name.startswith("zmq."):
        raise ModuleNotFoundError("No module named 'zmq'", name="zmq")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import

from soridormi_api import IMUState, JointState, MotorCommand, RobotState
from soridormi_runtime.scripted_head_skill import execute_scripted_head_plan, smoothstep
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest

registry = SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))
plan = registry.create_plan("look_direction", {"head_yaw_rad": 0.2, "duration_s": 0.2})
result = execute_scripted_head_plan(plan, dry_run=True, control_hz=10.0)
print(json.dumps({"ok": True, "steps": result.steps, "smoothstep": smoothstep(0.5)}))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["steps"] == 2
    assert payload["smoothstep"] == pytest.approx(0.5)


def test_scripted_head_live_without_pyzmq_returns_actionable_error() -> None:
    script = r'''
import builtins
import json

real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "zmq" or name.startswith("zmq."):
        raise ModuleNotFoundError("No module named 'zmq'", name="zmq")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import

from soridormi_runtime.scripted_head_skill import execute_scripted_head_plan
from soridormi_runtime.skill_execution import SkillExecutionRegistry
from soridormi_runtime.skill_manifest import DEFAULT_SKILL_MANIFEST, load_skill_manifest

registry = SkillExecutionRegistry(load_skill_manifest(DEFAULT_SKILL_MANIFEST))
plan = registry.create_plan("look_direction", {"head_yaw_rad": 0.2, "duration_s": 0.2})
try:
    execute_scripted_head_plan(plan, dry_run=False, control_hz=10.0)
except RuntimeError as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
else:
    raise AssertionError("expected missing-pyzmq RuntimeError")
'''
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is False
    assert "pyzmq" in payload["error"]
    assert "--dry-run" in payload["error"]



def test_scripted_social_shell_wrapper_runs_inside_runtime_container(tmp_path: Path) -> None:
    capture_path = tmp_path / "docker_args.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${SORIDORMI_FAKE_DOCKER_CAPTURE}\"\n"
        "exit 0\n"
    )
    fake_docker.chmod(0o755)

    env_path = Path(".env")
    old_env = env_path.read_text() if env_path.exists() else None
    env_path.write_text("UID=1000\nGID=1000\nCONTAINER_USER=chromie\nSIM_HOST=127.0.0.1\nSIM_PORT=5555\n")
    try:
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["SORIDORMI_FAKE_DOCKER_CAPTURE"] = str(capture_path)
        subprocess.run(
            [
                "bash",
                "scripts/run_scripted_social_skill_in_sim.sh",
                "look_direction",
                "--args",
                json.dumps({"head_yaw_rad": 0.2, "duration_s": 0.2}),
                "--backend",
                "mujoco",
                "--dry-run",
                "--json",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
    finally:
        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(old_env)

    docker_args = capture_path.read_text().splitlines()
    assert docker_args[:6] == ["compose", "-f", "compose.sim.yaml", "run", "--rm", "runtime"]
    assert "bash" in docker_args
    assert "-lc" in docker_args
    assert any("python -m soridormi_runtime.scripted_head_skill \"$@\"" in line for line in docker_args)
    separator_index = docker_args.index("_")
    assert docker_args[separator_index:] == [
        "_",
        "look_direction",
        "--args",
        json.dumps({"head_yaw_rad": 0.2, "duration_s": 0.2}),
        "--backend",
        "mujoco",
        "--dry-run",
        "--json",
    ]


def test_nod_yes_and_shake_no_are_experimental_scripted_head_skills() -> None:
    registry = _registry()
    for skill_id in ["nod_yes", "shake_no"]:
        skill = registry.skills[skill_id]
        assert skill["status"] == "available_sim_experimental"
        assert skill["execution"] == "scripted_keyframe"
        assert set(skill["required_actuator_groups"]) == {"head_neck"}
        assert skill["safety"]["hardware_enabled"] is False


def test_nod_yes_plan_uses_multi_keyframe_head_pitch_sequence() -> None:
    plan = _registry().create_plan("nod_yes", {"count": 2, "amplitude": "medium", "duration_s": 1.5})

    assert plan.commands == ()
    assert len(plan.keyframes) == 6
    assert plan.total_duration_s == pytest.approx(1.5)
    assert all(set(keyframe.positions_by_name) == set(HEAD_JOINT_NAMES) for keyframe in plan.keyframes)
    assert plan.keyframes[0].label == "nod_yes_neutral_start"
    assert plan.keyframes[0].positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert plan.keyframes[1].positions_by_name["head_pitch"] < -0.25
    assert plan.keyframes[2].positions_by_name["head_pitch"] > 0.17
    assert plan.keyframes[-1].label == "nod_yes_neutral_end"
    assert plan.keyframes[-1].positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert all(keyframe.positions_by_name["head_yaw"] == pytest.approx(0.0) for keyframe in plan.keyframes)


def test_shake_no_plan_uses_multi_keyframe_head_yaw_sequence() -> None:
    plan = _registry().create_plan("shake_no", {"count": 2, "amplitude": "small", "duration_s": 1.0})

    assert plan.commands == ()
    assert len(plan.keyframes) == 6
    assert plan.total_duration_s == pytest.approx(1.0)
    assert plan.keyframes[0].label == "shake_no_neutral_start"
    assert plan.keyframes[0].positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert plan.keyframes[1].positions_by_name["head_yaw"] > 0.27
    assert plan.keyframes[2].positions_by_name["head_yaw"] < -0.27
    assert plan.keyframes[-1].label == "shake_no_neutral_end"
    assert plan.keyframes[-1].positions_by_name["head_yaw"] == pytest.approx(0.0)
    assert all(keyframe.positions_by_name["head_pitch"] == pytest.approx(0.0) for keyframe in plan.keyframes)


def test_scripted_multi_keyframe_dry_run_sums_segment_steps() -> None:
    plan = _registry().create_plan("shake_no", {"count": 2, "duration_s": 1.0})
    result = execute_scripted_head_plan(plan, dry_run=True, control_hz=20.0, max_head_velocity_radps=0.0)

    assert result.executed is False
    assert len(plan.keyframes) == 6
    assert result.steps == 20
    assert result.duration_s == pytest.approx(1.0)
    assert result.final_positions_by_name["head_yaw"] == pytest.approx(0.0)


def test_scripted_social_rejects_non_integer_count() -> None:
    with pytest.raises(SkillExecutionError, match="integer"):
        _registry().create_plan("nod_yes", {"count": 2.5})


def test_scripted_social_requires_at_least_two_cycles() -> None:
    with pytest.raises(SkillExecutionError, match="min 2"):
        _registry().create_plan("shake_no", {"count": 1})


def test_scripted_social_allows_longer_sim_exploration_counts() -> None:
    plan = _registry().create_plan("shake_no", {"count": 5, "duration_s": 8.0})

    assert len(plan.keyframes) == 12
    assert plan.total_duration_s == pytest.approx(8.0)


def test_repeated_gestures_resolve_to_neutral_home_not_prior_head_drift() -> None:
    registry = _registry()
    reference = {"neck_pitch": 0.03, "head_pitch": -0.05, "head_yaw": 0.11, "head_roll": -0.02}

    nod = registry.create_plan("nod_yes", {"count": 2, "amplitude": "small"})
    nod_targets = resolve_keyframe_targets_for_execution(nod, reference)
    assert nod_targets[0] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
    assert nod_targets[1]["head_pitch"] == pytest.approx(-0.18)
    assert nod_targets[2]["head_pitch"] == pytest.approx(0.12)
    assert nod_targets[-1]["head_pitch"] == pytest.approx(0.0)
    assert all(target["head_yaw"] == pytest.approx(0.0) for target in nod_targets)

    shake = registry.create_plan("shake_no", {"count": 2, "amplitude": "small"})
    shake_targets = resolve_keyframe_targets_for_execution(shake, reference)
    assert shake_targets[0] == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
    assert shake_targets[1]["head_yaw"] == pytest.approx(0.28)
    assert shake_targets[2]["head_yaw"] == pytest.approx(-0.28)
    assert shake_targets[-1]["head_yaw"] == pytest.approx(0.0)
    assert all(target["head_pitch"] == pytest.approx(0.0) for target in shake_targets)


def test_scripted_social_cli_dry_run_json_for_nod_yes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "nod_yes",
            "--args",
            json.dumps({"count": 2, "duration_s": 1.0}),
            "--backend",
            "mujoco",
            "--control-hz",
            "20",
            "--dry-run",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["result"]["executed"] is False
    assert payload["result"]["steps"] > 20
    assert payload["result"]["auto_stretched_duration"] is True
    assert payload["plan"]["skill_id"] == "nod_yes"
    assert len(payload["plan"]["keyframes"]) == 6

def test_segment_target_reaches_extreme_before_segment_end() -> None:
    start = {"head_yaw": 0.0, "head_pitch": 0.0}
    target = {"head_yaw": 0.4, "head_pitch": 0.0}

    early = target_positions_for_segment_step(
        start,
        target,
        step_index=0,
        segment_steps=20,
        transition_fraction=0.25,
    )
    held = target_positions_for_segment_step(
        start,
        target,
        step_index=5,
        segment_steps=20,
        transition_fraction=0.25,
    )
    last = target_positions_for_segment_step(
        start,
        target,
        step_index=19,
        segment_steps=20,
        transition_fraction=0.25,
    )

    assert 0.0 < early["head_yaw"] < 0.4
    assert held["head_yaw"] == pytest.approx(0.4)
    assert last["head_yaw"] == pytest.approx(0.4)



def test_strict_target_names_do_not_blend_previous_pitch_drift() -> None:
    target = {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.4, "head_roll": 0.0}
    step_target = target_positions_for_segment_step(
        {"neck_pitch": -0.03, "head_pitch": -0.02, "head_yaw": 0.0, "head_roll": 0.01},
        target,
        step_index=0,
        segment_steps=20,
        transition_fraction=0.25,
        strict_target_names={"neck_pitch", "head_pitch", "head_roll"},
    )

    assert step_target["neck_pitch"] == pytest.approx(0.0)
    assert step_target["head_pitch"] == pytest.approx(0.0)
    assert step_target["head_roll"] == pytest.approx(0.0)
    assert 0.0 < step_target["head_yaw"] < 0.4

def test_segment_target_can_hold_extreme_immediately() -> None:
    target = {"head_yaw": -0.4, "head_pitch": 0.0}

    first = target_positions_for_segment_step(
        {"head_yaw": 0.0, "head_pitch": 0.0},
        target,
        step_index=0,
        segment_steps=20,
        transition_fraction=0.0,
    )

    assert first["head_yaw"] == pytest.approx(-0.4)


def test_head_velocity_limit_clamps_per_step_target_change() -> None:
    limited = limit_head_target_velocity(
        {"head_yaw": 0.0, "head_pitch": 0.0},
        {"head_yaw": 0.4, "head_pitch": -0.2},
        dt=0.02,
        max_velocity_radps=0.8,
    )

    assert limited["head_yaw"] == pytest.approx(0.016)
    assert limited["head_pitch"] == pytest.approx(-0.016)


def test_effective_duration_auto_stretches_fast_shake_no() -> None:
    plan = _registry().create_plan("shake_no", {"count": 2, "amplitude": "medium", "duration_s": 2.0})
    targets = resolve_keyframe_targets_for_execution(plan, {name: 0.0 for name in HEAD_JOINT_NAMES})

    effective = effective_duration_for_trajectory(
        requested_duration_s=plan.total_duration_s,
        targets=targets,
        max_head_velocity_radps=0.8,
        auto_stretch_duration=True,
        keyframe_durations=[keyframe.duration_s for keyframe in plan.keyframes],
    )

    assert effective == pytest.approx(7.5)


def test_plan_head_pose_trajectory_is_axis_specific_and_slow() -> None:
    plan = _registry().create_plan("shake_no", {"count": 2, "amplitude": "medium", "duration_s": 2.0})
    targets = resolve_keyframe_targets_for_execution(plan, {name: 0.0 for name in HEAD_JOINT_NAMES})
    effective = effective_duration_for_trajectory(
        requested_duration_s=plan.total_duration_s,
        targets=targets,
        max_head_velocity_radps=0.8,
        auto_stretch_duration=True,
        keyframe_durations=[keyframe.duration_s for keyframe in plan.keyframes],
    )
    from soridormi_runtime.scripted_head_skill import keyframe_steps_for_durations, scaled_keyframe_durations

    steps = keyframe_steps_for_durations(scaled_keyframe_durations(plan, effective), 50.0)
    trajectory = plan_head_pose_trajectory(
        plan,
        targets,
        steps,
        start_positions_by_name={name: 0.0 for name in HEAD_JOINT_NAMES},
        control_hz=50.0,
        transition_fraction=0.25,
        max_head_velocity_radps=0.8,
    )

    assert len(trajectory) == 375
    assert min(point["head_yaw"] for point in trajectory) == pytest.approx(-0.4)
    assert max(point["head_yaw"] for point in trajectory) == pytest.approx(0.4)
    assert {round(point["head_pitch"], 6) for point in trajectory} == {0.0}


def test_live_fake_shake_reports_commanded_and_observed_yaw_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, host: str, port: int) -> None:
            self.state = _state()

        def read_state(self) -> RobotState:
            return self.state

        def step_motor_command(self, command):  # type: ignore[no-untyped-def]
            current = dict(zip(self.state.joints.names, self.state.joints.positions))
            command_positions = dict(zip(command.names, command.positions))
            names = list(self.state.joints.names)
            positions = [float(command_positions.get(name, current[name])) for name in names]
            self.state = RobotState(
                time=self.state.time + 0.02,
                joints=JointState(
                    names=names,
                    positions=positions,
                    velocities=[0.0] * len(names),
                    torques=[0.0] * len(names),
                ),
                imu=IMUState(),
            )
            return self.state

        def close(self) -> None:
            return None

    monkeypatch.setattr(scripted_head_skill, "_load_robot_api_client_class", lambda: FakeClient)

    plan = _registry().create_plan("shake_no", {"count": 2, "amplitude": "medium", "duration_s": 2.0})
    result = execute_scripted_head_plan(
        plan,
        dry_run=False,
        control_hz=50.0,
        transition_fraction=0.25,
        max_head_velocity_radps=0.0,
    )

    assert result.executed is True
    assert result.start_positions_by_name["head_yaw"] != pytest.approx(0.0)
    assert result.target_min_positions_by_name["head_yaw"] == pytest.approx(-0.4)
    assert result.target_max_positions_by_name["head_yaw"] == pytest.approx(0.4)
    assert result.observed_min_positions_by_name["head_yaw"] == pytest.approx(-0.4)
    assert result.observed_max_positions_by_name["head_yaw"] == pytest.approx(0.4)
    assert result.keyframe_targets[0]["label"] == "shake_no_neutral_start"
    assert result.keyframe_targets[1]["label"] == "shake_no_right_1"
    assert result.target_min_positions_by_name["head_pitch"] == pytest.approx(0.0)
    assert result.target_max_positions_by_name["head_pitch"] == pytest.approx(0.0)


def test_scripted_head_cli_dry_run_json_reports_yaw_range() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.scripted_head_skill",
            "shake_no",
            "--args",
            json.dumps({"count": 2, "amplitude": "medium", "duration_s": 2.0}),
            "--backend",
            "mujoco",
            "--control-hz",
            "50",
            "--dry-run",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)

    result = payload["result"]
    assert result["target_min_positions_by_name"]["head_yaw"] == pytest.approx(-0.4)
    assert result["target_max_positions_by_name"]["head_yaw"] == pytest.approx(0.4)
    assert result["requested_duration_s"] == pytest.approx(2.0)
    assert result["effective_duration_s"] == pytest.approx(17.142857142857146)
    assert result["steps"] == 858
    assert result["auto_stretched_duration"] is True
    assert result["max_head_velocity_radps"] == pytest.approx(0.35)
    assert result["transition_fraction"] == pytest.approx(0.40)
    assert payload["plan"]["keyframes"][0]["label"] == "shake_no_neutral_start"
    assert payload["plan"]["keyframes"][1]["label"] == "shake_no_right_1"
