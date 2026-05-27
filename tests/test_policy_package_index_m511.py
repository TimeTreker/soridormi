from __future__ import annotations

import json
import tarfile
from pathlib import Path

from soridormi_runtime.policy_package import index_policy_packages, package_policy_profile

from test_policy_package_m59 import write_profile, write_robot_config


def test_policy_package_index_lists_and_verifies_packages(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="indexed_profile")

    package = package_policy_profile(profile, robot_config_path=robot_config, output_dir=tmp_path / "packages")
    assert package.ok

    result = index_policy_packages(tmp_path / "packages")

    assert result.ok
    assert len(result.packages) == 1
    entry = result.packages[0]
    assert entry.ok
    assert entry.profile_name == "indexed_profile"
    assert entry.package_version == 1
    assert entry.files_checked >= 5
    assert entry.sha256 == _sha256(Path(package.package_path))


def test_policy_package_index_reports_tampered_packages(tmp_path: Path) -> None:
    robot_config = tmp_path / "robot.yaml"
    profile = tmp_path / "profile.yaml"
    write_robot_config(robot_config)
    write_profile(profile, name="indexed_tamper_profile")
    package = package_policy_profile(profile, robot_config_path=robot_config, output_dir=tmp_path / "packages")
    assert package.ok

    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(package.package_path, "r:gz") as tar:
        tar.extractall(unpacked, filter="data")
    (unpacked / "profile.yaml").write_text("name: changed\n", encoding="utf-8")
    tampered = tmp_path / "packages" / "tampered.policy.tar.gz"
    with tarfile.open(tampered, "w:gz") as tar:
        for path in sorted(unpacked.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(unpacked).as_posix())

    result = index_policy_packages(tmp_path / "packages")

    assert not result.ok
    failed = [item for item in result.packages if not item.ok]
    assert failed
    assert any("sha256 mismatch" in error for item in failed for error in item.errors)


def test_policy_package_index_json_shape(tmp_path: Path) -> None:
    result = index_policy_packages(tmp_path / "missing")
    payload = json.loads(json.dumps(result, default=lambda value: value.__dict__))

    assert payload["ok"] is True
    assert payload["packages"] == []
    assert payload["warnings"]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
