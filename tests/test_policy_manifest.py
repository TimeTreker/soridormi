from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import yaml

from soridormi_runtime.policy_manifest import build_policy_manifest, inspect_model_artifact


def _write_temp_profile(tmp_path: Path, *, model_path: str | Path, name: str = "tmp_replacement") -> Path:
    profile_payload = yaml.safe_load(Path("configs/policies/open_duck_forward.yaml").read_text())
    profile_payload["name"] = name
    profile_payload["model"]["path"] = str(model_path)
    profile_path = tmp_path / f"{name}.yaml"
    profile_path.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")
    return profile_path


def test_manifest_static_does_not_require_missing_model(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_replacement.onnx"
    profile_path = _write_temp_profile(tmp_path, model_path=missing_model, name="tmp_missing_static")

    result = build_policy_manifest(profile_path, robot_config_path="configs/robots/open_duck_mini_v2.yaml")

    assert result.ok
    assert result.profile_name == "tmp_missing_static"
    assert result.contract["ok"] is True
    assert result.model_check is None
    assert not result.model_artifact.exists
    assert any("Model artifact not found" in warning for warning in result.warnings)


def test_manifest_can_require_model_file(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_replacement.onnx"
    profile_path = _write_temp_profile(tmp_path, model_path=missing_model, name="tmp_missing_required")

    result = build_policy_manifest(
        profile_path,
        robot_config_path="configs/robots/open_duck_mini_v2.yaml",
        require_model=True,
    )

    assert not result.ok
    assert any("Model artifact not found" in error for error in result.errors)


def test_manifest_hashes_existing_model_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "replacement.onnx"
    payload = b"fake but pinned model bytes"
    model_path.write_bytes(payload)

    profile_path = _write_temp_profile(tmp_path, model_path=model_path)

    result = build_policy_manifest(profile_path, robot_config_path="configs/robots/open_duck_mini_v2.yaml")

    assert result.ok
    assert result.model_artifact.exists
    assert result.model_artifact.size_bytes == len(payload)
    assert result.model_artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert asdict(result)["model_artifact"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_inspect_model_artifact_can_skip_hash(tmp_path: Path) -> None:
    model_path = tmp_path / "replacement.onnx"
    model_path.write_bytes(b"abc")

    info = inspect_model_artifact(model_path, hash_model=False)

    assert info.exists
    assert info.size_bytes == 3
    assert info.sha256 is None
