from __future__ import annotations

from pathlib import Path

import pytest

from soridormi_runtime.policy_profiles import PolicyProfile, resolve_policy_profile_path


def write_profile(path: Path) -> None:
    path.write_text(
        """
name: test_profile
description: unit test profile
model:
  path: /models/test.onnx
  input_name: obs
  output_name: continuous_actions
  input_shape: [1, 101]
  output_shape: [1, 14]
runtime:
  mode: onnx_policy
  backend: sim
  control_hz: 50
command:
  x: 0.07
  y: -0.01
  yaw: 0.02
  ramp_seconds: 0.5
phase:
  mode: step
  period_steps: 50
  step_increment: 1
  enabled: true
action_mapping:
  action_scale: 0.25
  max_motor_velocity: 5.24
observation:
  accel_bias_xyz: [1.3, 0.0, 0.0]
  use_state_feet_contacts: true
  bootstrap_policy_defaults_from_state: true
logging:
  enabled: true
  format: mcap
  every_n: 1
  prefix: unit_policy
""",
        encoding="utf-8",
    )


def test_policy_profile_loads_env_contract(tmp_path: Path) -> None:
    profile_path = tmp_path / "test_profile.yaml"
    write_profile(profile_path)

    env = PolicyProfile.load(profile_path).env()

    assert env["SORIDORMI_POLICY_PATH"] == "/models/test.onnx"
    assert env["SORIDORMI_POLICY_INPUT_NAME"] == "obs"
    assert env["SORIDORMI_POLICY_OUTPUT_NAME"] == "continuous_actions"
    assert env["SORIDORMI_POLICY_EXPECTED_INPUT_SHAPE"] == "1,101"
    assert env["SORIDORMI_POLICY_EXPECTED_OUTPUT_SHAPE"] == "1,14"
    assert env["SORIDORMI_COMMAND_X"] == "0.07"
    assert env["SORIDORMI_COMMAND_Y"] == "-0.01"
    assert env["SORIDORMI_COMMAND_YAW"] == "0.02"
    assert env["SORIDORMI_ACTION_SCALE"] == "0.25"
    assert env["SORIDORMI_PHASE_REFERENCE_DATA"].endswith("polynomial_coefficients.pkl")
    assert env["SORIDORMI_RUNTIME_LOG_PREFIX"] == "unit_policy"


def test_policy_profile_shell_exports_are_safe(tmp_path: Path) -> None:
    profile_path = tmp_path / "test_profile.yaml"
    write_profile(profile_path)

    exports = PolicyProfile.load(profile_path).shell_exports()

    assert "export SORIDORMI_POLICY_PATH=/models/test.onnx" in exports
    assert "export SORIDORMI_RUNTIME_MODE=onnx_policy" in exports


def test_resolve_policy_profile_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = tmp_path / "test_profile.yaml"
    write_profile(profile_path)
    monkeypatch.setenv("SORIDORMI_POLICY_PROFILE_FILE", str(profile_path))
    monkeypatch.delenv("SORIDORMI_POLICY_PROFILE", raising=False)

    assert resolve_policy_profile_path(None) == profile_path
