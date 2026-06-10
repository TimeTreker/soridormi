from __future__ import annotations

import subprocess
import sys


def test_onnx_modules_import_without_onnxruntime_installed() -> None:
    code = """
from soridormi_runtime.onnx_policy import OnnxPolicy
from soridormi_runtime.onnx_policy_controller import OnnxPolicyController
from soridormi_runtime.residual_policy import ResidualOnnxPolicy
print(OnnxPolicy.__name__, OnnxPolicyController.__name__, ResidualOnnxPolicy.__name__)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "OnnxPolicy" in proc.stdout


def test_mcp_http_tests_collect_when_optional_deps_are_missing_or_run_when_present() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_mcp_http_server.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "skipped" in proc.stdout.lower() or "passed" in proc.stdout.lower()
