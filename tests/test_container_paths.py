from pathlib import Path
import subprocess


def _run_bash(script: str) -> str:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def test_container_data_path_translation_handles_host_data_paths() -> None:
    output = _run_bash(
        r'''
        set -euo pipefail
        SORIDORMI_REPO_ROOT="$PWD"
        source scripts/lib/container_paths.sh
        soridormi_to_container_data_path data/policy_packages/example.policy.tar.gz
        soridormi_to_container_data_path "$PWD/data/policy_packages/example.policy.tar.gz"
        soridormi_to_container_data_path /data/policy_packages/example.policy.tar.gz
        soridormi_to_container_data_path configs/policies/example.yaml
        '''
    ).splitlines()

    assert output == [
        "/data/policy_packages/example.policy.tar.gz",
        "/data/policy_packages/example.policy.tar.gz",
        "/data/policy_packages/example.policy.tar.gz",
        "configs/policies/example.yaml",
    ]


def test_container_data_arg_translation_preserves_flags() -> None:
    output = _run_bash(
        r'''
        set -euo pipefail
        SORIDORMI_REPO_ROOT="$PWD"
        source scripts/lib/container_paths.sh
        translated=()
        soridormi_translate_container_data_args translated \
          --output-dir data/policy_packages \
          --require-provider CUDAExecutionProvider \
          data/policy_packages/example.policy.tar.gz
        printf '%s\n' "${translated[@]}"
        '''
    ).splitlines()

    assert output == [
        "--output-dir",
        "/data/policy_packages",
        "--require-provider",
        "CUDAExecutionProvider",
        "/data/policy_packages/example.policy.tar.gz",
    ]


def test_mcp_image_keeps_repo_config_manifests_available() -> None:
    repo = Path(__file__).resolve().parents[1]
    dockerfile = (repo / "docker" / "mcp" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY configs /app/configs" in dockerfile
    assert 'pip install --no-cache-dir -e ".[mcp]"' in dockerfile
