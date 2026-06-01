from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_run_sim_server_help_documents_backend_profile_and_viewer_flags() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_sim_server.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--backend NAME" in proc.stdout
    assert "default: mujoco" in proc.stdout
    assert "--profile PROFILE" in proc.stdout
    assert "--viewer" in proc.stdout
    assert "--no-viewer" in proc.stdout
    assert "--follow-camera" in proc.stdout
    assert "--no-follow-camera" in proc.stdout
    assert "--camera-distance N" in proc.stdout
    assert "--camera-azimuth DEG" in proc.stdout
    assert "--camera-elevation DEG" in proc.stdout
    assert "--rough-ground" in proc.stdout
    assert "--rough-stone-height M" in proc.stdout
    assert "--rough-stone-count N" in proc.stdout
    assert "--rough-stone-radius M" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera" in proc.stdout
    assert "--follow-camera --rough-ground" in proc.stdout


def test_run_sim_server_script_defaults_to_mujoco_without_viewer() -> None:
    text = open("scripts/run_sim_server.sh", encoding="utf-8").read()

    assert 'SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"' in text
    assert 'VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"' in text
    assert 'FOLLOW_CAMERA="${SORIDORMI_MUJOCO_FOLLOW_CAMERA:-0}"' in text
    assert 'CAMERA_DISTANCE="${SORIDORMI_MUJOCO_CAMERA_DISTANCE:-1.4}"' in text
    assert 'CAMERA_AZIMUTH="${SORIDORMI_MUJOCO_CAMERA_AZIMUTH:-135}"' in text
    assert 'CAMERA_ELEVATION="${SORIDORMI_MUJOCO_CAMERA_ELEVATION:--20}"' in text
    assert 'SIM_POLICY_PROFILE="${SORIDORMI_SIM_POLICY_PROFILE:-}"' in text
    assert 'ROUGH_GROUND="${SORIDORMI_MUJOCO_ROUGH_GROUND:-0}"' in text
    assert 'ROUGH_STONE_HEIGHT="${SORIDORMI_MUJOCO_ROUGH_STONE_HEIGHT:-0.008}"' in text
    assert 'export SORIDORMI_SIM_BACKEND="${SIM_BACKEND}"' in text
    assert 'export SORIDORMI_MUJOCO_VIEWER="${VIEWER_ENABLED}"' in text
    assert 'export SORIDORMI_MUJOCO_FOLLOW_CAMERA="${FOLLOW_CAMERA}"' in text
    assert 'export SORIDORMI_MUJOCO_CAMERA_DISTANCE="${CAMERA_DISTANCE}"' in text
    assert 'export SORIDORMI_MUJOCO_CAMERA_AZIMUTH="${CAMERA_AZIMUTH}"' in text
    assert 'export SORIDORMI_MUJOCO_CAMERA_ELEVATION="${CAMERA_ELEVATION}"' in text
    assert 'export SORIDORMI_SIM_POLICY_PROFILE="${SIM_POLICY_PROFILE}"' in text
    assert 'export SORIDORMI_MUJOCO_ROUGH_GROUND="${ROUGH_GROUND}"' in text
    assert 'SORIDORMI_SIM_BACKEND_OVERRIDE' in text
    assert 'SORIDORMI_MUJOCO_VIEWER_OVERRIDE' in text
    assert 'SORIDORMI_MUJOCO_FOLLOW_CAMERA_OVERRIDE' in text
    assert 'SORIDORMI_MUJOCO_CAMERA_DISTANCE_OVERRIDE' in text
    assert 'SORIDORMI_MUJOCO_ROUGH_GROUND_OVERRIDE' in text
    assert 'ROUGH_MODEL="$(dirname "${BASE_MODEL}")/soridormi_rough_ground_scene.xml"' in text
    assert 'MuJoCo\n      # resolves mesh and texture paths relative to the top-level XML/compiler' in text
    assert 'python -m soridormi_sim.rough_ground_scene' in text


def test_run_sim_server_resolves_profile_inside_sim_container_before_server_start() -> None:
    text = open("scripts/run_sim_server.sh", encoding="utf-8").read()

    assert 'python -m soridormi_runtime.policy_profiles "${SORIDORMI_SIM_POLICY_PROFILE}" --shell' in text
    assert 'SORIDORMI_MUJOCO_USE_HOME_KEYFRAME' in text
    assert 'SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE' in text
    assert 'SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE' in text
    assert 'SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE' in text


def test_policy_smoke_help_tells_user_to_start_profiled_mujoco_server() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_policy_rollout_smoke.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "matching MuJoCo compatibility" in proc.stdout
    assert "--profile open_duck_forward" in proc.stdout


def test_policy_smoke_help_documents_runtime_log_overrides() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_policy_rollout_smoke.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--log-format FORMAT" in proc.stdout
    assert "--log-prefix PREFIX" in proc.stdout
    assert "--log-dir DIR" in proc.stdout
    assert "--log-every-n N" in proc.stdout
    assert "--no-log" in proc.stdout


def test_policy_smoke_exports_logging_override_envs_for_profile_parity() -> None:
    text = open("scripts/run_policy_rollout_smoke.sh", encoding="utf-8").read()

    assert "SORIDORMI_RUNTIME_LOG_FORMAT_OVERRIDE" in text
    assert "SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE" in text
    assert "SORIDORMI_RUNTIME_LOG_DIR_OVERRIDE" in text
    assert "SORIDORMI_RUNTIME_LOG_EVERY_N_OVERRIDE" in text
    assert "open_duck_forward defaults to MCAP logging" in text


def test_policy_experiment_reapplies_logging_overrides_after_profile_resolution() -> None:
    text = open("scripts/run_policy_experiment.sh", encoding="utf-8").read()

    assert "LOG_FORMAT_OVERRIDE" in text
    assert 'eval "$(python -m soridormi_runtime.policy_profiles "${PROFILE}" --shell)"' in text
    assert 'export SORIDORMI_RUNTIME_LOG_FORMAT="${LOG_FORMAT_OVERRIDE}"' in text
    assert "Runtime log: enabled=" in text


def test_compose_runtime_passes_logging_override_envs() -> None:
    compose_path = Path("compose.sim.yaml")
    if not compose_path.exists():
        pytest.skip("compose.sim.yaml is not available in this runtime test environment")

    text = compose_path.read_text(encoding="utf-8")

    assert "SORIDORMI_RUNTIME_LOG_FORMAT_OVERRIDE: ${SORIDORMI_RUNTIME_LOG_FORMAT_OVERRIDE:-}" in text
    assert "SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE: ${SORIDORMI_RUNTIME_LOG_PREFIX_OVERRIDE:-}" in text
    assert "SORIDORMI_RUNTIME_LOG_DIR_OVERRIDE: ${SORIDORMI_RUNTIME_LOG_DIR_OVERRIDE:-}" in text
    assert "SORIDORMI_RUNTIME_LOG_EVERY_N_OVERRIDE: ${SORIDORMI_RUNTIME_LOG_EVERY_N_OVERRIDE:-}" in text
