from __future__ import annotations

import subprocess


def test_run_sim_server_help_documents_backend_and_viewer_flags() -> None:
    proc = subprocess.run(
        ["bash", "scripts/run_sim_server.sh", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--backend NAME" in proc.stdout
    assert "default: mujoco" in proc.stdout
    assert "--viewer" in proc.stdout
    assert "--no-viewer" in proc.stdout
    assert "./scripts/run_sim_server.sh --backend mujoco --viewer" in proc.stdout


def test_run_sim_server_script_defaults_to_mujoco_without_viewer() -> None:
    text = open("scripts/run_sim_server.sh", encoding="utf-8").read()

    assert 'SIM_BACKEND="${SORIDORMI_SIM_BACKEND:-mujoco}"' in text
    assert 'VIEWER_ENABLED="${SORIDORMI_MUJOCO_VIEWER:-0}"' in text
    assert 'export SORIDORMI_SIM_BACKEND="${SIM_BACKEND}"' in text
    assert 'export SORIDORMI_MUJOCO_VIEWER="${VIEWER_ENABLED}"' in text
