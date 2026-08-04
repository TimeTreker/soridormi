from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_ci_static_check_script_smoke_without_pytest() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "ci_static_check.sh"
    if not script.exists():
        pytest.skip("ci_static_check.sh is not present in this runtime-dev image checkout")
    env = os.environ.copy()
    env["SORIDORMI_CI_SKIP_PYTEST"] = "1"
    env["SORIDORMI_CI_STATIC_CHECK_USE_DOCKER"] = "0"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo / 'src'}" + (f":{existing_pythonpath}" if existing_pythonpath else "")

    result = subprocess.run(
        [str(script)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "Soridormi CI static check" in result.stdout
    assert "Validating policy profile suite" in result.stdout
    assert "Exporting canonical policy manifest" in result.stdout
    assert "Checking replacement-profile scaffold workflow" in result.stdout
    assert "Checking replacement-policy package workflow" in result.stdout
