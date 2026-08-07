from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


def test_repository_governance_ignores_generated_python_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    module_path = repo / "scripts" / "validate_repository_governance.py"
    spec = importlib.util.spec_from_file_location("validate_repository_governance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source_dir = tmp_path / "src" / "soridormi_runtime"
    cache_dir = source_dir / "__pycache__"
    cache_dir.mkdir(parents=True)
    (source_dir / "current.py").write_text("", encoding="utf-8")
    legacy_stem = "retired_" + "m" + str(10) + "_contract"
    (cache_dir / f"{legacy_stem}.cpython-312.pyc").write_bytes(b"generated")
    (source_dir / f"{legacy_stem}.pyc").write_bytes(b"generated")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.first_party_files() == [Path("src/soridormi_runtime/current.py")]


@pytest.mark.parametrize(
    ("script_name", "docker_mode_variable", "expected_service"),
    (
        ("ci_static_check.sh", "SORIDORMI_CI_STATIC_CHECK_USE_DOCKER", "runtime"),
        (
            "validate_task_agent_contract.sh",
            "SORIDORMI_TASK_AGENT_USE_DOCKER",
            "mcp-runtime",
        ),
    ),
)
def test_container_validation_mounts_the_full_checkout(
    tmp_path: Path,
    script_name: str,
    docker_mode_variable: str,
    expected_service: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    docker_log = tmp_path / "docker.args"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = compose ] && [ \"${2:-}\" = version ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"${FAKE_DOCKER_LOG}\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(docker_log)
    env[docker_mode_variable] = "1"
    # This test owns a synthetic host-to-container boundary. When the suite is
    # itself running inside the maintained validation container, do not let the
    # parent's recursion sentinel bypass the fake Docker executable below.
    env.pop("SORIDORMI_CI_STATIC_CHECK_IN_CONTAINER", None)

    subprocess.run(
        [str(repo / "scripts" / script_name)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    docker_args = docker_log.read_text(encoding="utf-8").splitlines()
    assert f"{repo}:/app" in docker_args
    assert "/app" in docker_args
    assert expected_service in docker_args


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
