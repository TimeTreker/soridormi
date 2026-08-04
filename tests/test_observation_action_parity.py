from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from soridormi_runtime.compare_observation_action_parity import (
    compare_observation_action_parity,
    load_official_trace,
    load_soridormi_trace,
)


def _write_trace(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _record(step: int, *, obs_delta: float = 0.0, action_delta: float = 0.0) -> dict:
    obs = [0.0] * 101
    obs[0] = float(step)
    obs[3] = 1.3 + obs_delta
    obs[97] = 1.0
    obs[98] = 1.0
    obs[99] = 1.0
    obs[100] = 0.0
    action = [0.1 + action_delta] * 14
    return {
        "step_index": step,
        "robot_time": step * 0.02,
        "observation": obs,
        "action": action,
        "motor_targets": [0.2] * 14,
        "joint_positions": [0.0] * 14,
        "joint_velocities": [0.0] * 14,
        "contacts": [1.0, 1.0],
        "phase": [1.0, 0.0],
        "command": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "base_position_xyz": [step * 0.01, 0.0, 0.15],
    }


def test_observation_action_parity_reports_segment_differences(tmp_path: Path) -> None:
    official_path = tmp_path / "official.trace.jsonl"
    soridormi_path = tmp_path / "soridormi.trace.jsonl"
    _write_trace(official_path, [_record(0), _record(1)])
    _write_trace(soridormi_path, [_record(0, obs_delta=2.0), _record(1, obs_delta=2.0)])

    summary = compare_observation_action_parity(
        load_official_trace(official_path),
        load_soridormi_trace(soridormi_path),
        steps=2,
    )

    assert summary["steps_compared"] == 2
    assert summary["metrics"]["observation"]["count"] == 2
    assert summary["worst_observation_segments"][0]["name"] == "accelerometer_xyz"
    assert any("Observation parity" in item for item in summary["diagnosis"])


def test_observation_action_parity_can_rerun_fake_onnx(tmp_path: Path, monkeypatch) -> None:
    official_path = tmp_path / "official.trace.jsonl"
    soridormi_path = tmp_path / "soridormi.trace.jsonl"
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"fake")

    rec = _record(0)
    rec["action"] = [0.0] * 14
    _write_trace(official_path, [rec])
    _write_trace(soridormi_path, [rec])

    fake = types.ModuleType("onnxruntime")

    class FakeInfo:
        def __init__(self, name: str, shape: list[int]):
            self.name = name
            self.shape = shape
            self.type = "tensor(float)"

    class FakeSession:
        def __init__(self, path: str, providers=None):
            self.path = path
            self.providers = providers or ["CPUExecutionProvider"]

        def get_inputs(self):
            return [FakeInfo("obs", [1, 101])]

        def get_outputs(self):
            return [FakeInfo("continuous_actions", [1, 14])]

        def get_providers(self):
            return self.providers

        def run(self, outputs, feeds):
            return [np.zeros((1, 14), dtype=np.float32)]

    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    summary = compare_observation_action_parity(
        load_official_trace(official_path),
        load_soridormi_trace(soridormi_path),
        steps=1,
        policy_path=policy_path,
    )

    assert summary["onnx_rerun"]["enabled"] is True
    assert summary["onnx_rerun"]["official_obs_vs_official_action"]["max_abs_diff"] == 0.0
    assert any("ONNX inference is parity-compatible" in item for item in summary["diagnosis"])
