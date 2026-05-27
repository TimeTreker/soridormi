from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soridormi_runtime.onnx_providers import parse_provider_csv
from soridormi_runtime.policy_acceptance import accept_policy_profile
from soridormi_runtime.policy_profiles import PolicyProfile


@dataclass(frozen=True)
class PolicyPackageFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PolicyPackageResult:
    ok: bool
    package_version: int
    generated_at_utc: str
    profile_name: str
    package_path: str
    include_model: bool
    files: list[PolicyPackageFile]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyPackageVerificationResult:
    ok: bool
    package_path: str
    package_version: int | None
    profile_name: str | None
    files_checked: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("._-")
    return value or "policy"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> PolicyPackageFile:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return PolicyPackageFile(path=str(dst), size_bytes=dst.stat().st_size, sha256=_sha256_file(dst))


def _write_json(path: Path, payload: Any) -> PolicyPackageFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PolicyPackageFile(path=str(path), size_bytes=path.stat().st_size, sha256=_sha256_file(path))


def _relative_file_info(root: Path, info: PolicyPackageFile) -> PolicyPackageFile:
    path = Path(info.path)
    return PolicyPackageFile(path=path.relative_to(root).as_posix(), size_bytes=info.size_bytes, sha256=info.sha256)


def _artifact_path(path_text: str) -> Path:
    return Path(path_text).expanduser()


def package_policy_profile(
    profile: str | Path | PolicyProfile,
    *,
    output_dir: str | Path = "data/policy_packages",
    robot_config_path: str | Path | None = None,
    include_model: bool = False,
    require_model: bool = False,
    check_model: bool = False,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
    hash_model: bool = True,
    profile_only: bool = True,
    force: bool = False,
) -> PolicyPackageResult:
    """Build a portable tar.gz handoff package for one replacement profile.

    The package is source-control friendly by default: it always includes the
    profile YAML and M5 acceptance artifacts, but it only embeds the ONNX file
    when ``include_model`` is requested. Use ``require_model`` for release
    packages where the artifact must be present.
    """
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
    stamp = _utc_stamp()
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    package_path = output_root / f"{_slug(policy_profile.name)}_{stamp}.policy.tar.gz"
    if package_path.exists() and not force:
        raise FileExistsError(f"Policy package already exists: {package_path}")

    errors: list[str] = []
    warnings: list[str] = []

    with tempfile.TemporaryDirectory(prefix="soridormi_policy_package_") as tmp:
        tmp_root = Path(tmp)
        package_root = tmp_root / "package"
        package_root.mkdir()

        acceptance = accept_policy_profile(
            policy_profile,
            output_dir=tmp_root / "acceptance_runs",
            robot_config_path=robot_config_path,
            check_model=check_model,
            require_model=require_model,
            providers=providers,
            require_providers=require_providers,
            prefer_cuda=prefer_cuda,
            hash_model=hash_model,
            profile_only=profile_only,
            force=force,
        )
        errors.extend(acceptance.errors)
        warnings.extend(acceptance.warnings)

        files: list[PolicyPackageFile] = []
        files.append(_copy_file(Path(policy_profile.path), package_root / "profile.yaml"))

        for name, artifact_path in (
            ("contract.json", acceptance.artifacts.contract_json),
            ("manifest.json", acceptance.artifacts.manifest_json),
            ("profile_suite.json", acceptance.artifacts.suite_validation_json),
            ("acceptance.json", Path(acceptance.artifacts.directory) / "acceptance.json"),
            ("acceptance_report.md", acceptance.artifacts.report_markdown),
        ):
            files.append(_copy_file(_artifact_path(str(artifact_path)), package_root / "artifacts" / name))

        model_artifact: dict[str, Any] | None = None
        model_path = Path(policy_profile.model.path).expanduser()
        if include_model:
            if model_path.exists():
                copied = _copy_file(model_path, package_root / "model" / model_path.name)
                files.append(copied)
                model_artifact = {
                    "declared_path": str(policy_profile.model.path),
                    "packaged_path": Path(copied.path).relative_to(package_root).as_posix(),
                    "size_bytes": copied.size_bytes,
                    "sha256": copied.sha256,
                }
            else:
                message = f"Model artifact not found for package inclusion: {policy_profile.model.path}"
                if require_model:
                    errors.append(message)
                else:
                    warnings.append(message)

        manifest_without_files: dict[str, Any] = {
            "package_version": 1,
            "generated_at_utc": _utc_iso(),
            "profile_name": policy_profile.name,
            "profile_path": str(policy_profile.path),
            "robot_config_path": str(robot_config_path) if robot_config_path is not None else None,
            "include_model": include_model,
            "require_model": require_model,
            "check_model": check_model,
            "acceptance_ok": acceptance.ok,
            "model_artifact": model_artifact,
            "errors": errors,
            "warnings": warnings,
        }
        relative_files = [_relative_file_info(package_root, item) for item in files]
        package_manifest = dict(manifest_without_files)
        package_manifest["files"] = [asdict(item) for item in relative_files]
        manifest_file = _write_json(package_root / "package_manifest.json", package_manifest)
        files.append(manifest_file)

        with tarfile.open(package_path, "w:gz") as tar:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(package_root).as_posix())

        return PolicyPackageResult(
            ok=acceptance.ok and not errors,
            package_version=1,
            generated_at_utc=manifest_without_files["generated_at_utc"],
            profile_name=policy_profile.name,
            package_path=str(package_path),
            include_model=include_model,
            files=[_relative_file_info(package_root, item) for item in files],
            errors=errors,
            warnings=warnings,
        )


def verify_policy_package(package_path: str | Path) -> PolicyPackageVerificationResult:
    path = Path(package_path).expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return PolicyPackageVerificationResult(
            ok=False,
            package_path=str(path),
            package_version=None,
            profile_name=None,
            files_checked=0,
            errors=[f"Policy package not found: {path}"],
            warnings=[],
        )

    with tempfile.TemporaryDirectory(prefix="soridormi_policy_verify_") as tmp:
        root = Path(tmp)
        try:
            with tarfile.open(path, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    member_path = root / member.name
                    if not member_path.resolve().is_relative_to(root.resolve()):
                        errors.append(f"Unsafe archive member path: {member.name}")
                if errors:
                    return PolicyPackageVerificationResult(
                        ok=False,
                        package_path=str(path),
                        package_version=None,
                        profile_name=None,
                        files_checked=0,
                        errors=errors,
                        warnings=warnings,
                    )
                tar.extractall(root, filter="data")
        except Exception as exc:
            return PolicyPackageVerificationResult(
                ok=False,
                package_path=str(path),
                package_version=None,
                profile_name=None,
                files_checked=0,
                errors=[f"Failed to read policy package: {exc!r}"],
                warnings=warnings,
            )

        manifest_path = root / "package_manifest.json"
        if not manifest_path.exists():
            return PolicyPackageVerificationResult(
                ok=False,
                package_path=str(path),
                package_version=None,
                profile_name=None,
                files_checked=0,
                errors=["Package is missing package_manifest.json"],
                warnings=warnings,
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return PolicyPackageVerificationResult(
                ok=False,
                package_path=str(path),
                package_version=None,
                profile_name=None,
                files_checked=0,
                errors=[f"Failed to parse package_manifest.json: {exc!r}"],
                warnings=warnings,
            )

        files = manifest.get("files") or []
        checked = 0
        for item in files:
            rel = str(item.get("path", ""))
            if not rel or rel == "package_manifest.json":
                continue
            candidate = root / rel
            if not candidate.exists():
                errors.append(f"Package file missing: {rel}")
                continue
            checked += 1
            size_bytes = candidate.stat().st_size
            if item.get("size_bytes") != size_bytes:
                errors.append(f"Package file size mismatch for {rel}: expected {item.get('size_bytes')} got {size_bytes}")
            digest = _sha256_file(candidate)
            if item.get("sha256") != digest:
                errors.append(f"Package file sha256 mismatch for {rel}")

        if not (root / "profile.yaml").exists():
            errors.append("Package is missing profile.yaml")
        if not (root / "artifacts" / "acceptance.json").exists():
            errors.append("Package is missing artifacts/acceptance.json")
        if manifest.get("errors"):
            warnings.append("Package manifest contains acceptance/package errors")

        return PolicyPackageVerificationResult(
            ok=not errors,
            package_path=str(path),
            package_version=manifest.get("package_version"),
            profile_name=manifest.get("profile_name"),
            files_checked=checked,
            errors=errors,
            warnings=warnings + list(manifest.get("warnings") or []),
        )


def print_package_summary(result: PolicyPackageResult) -> None:
    print("Soridormi policy package")
    print("=========================")
    print(f"Profile: {result.profile_name}")
    print(f"Package: {result.package_path}")
    print(f"Include model: {'yes' if result.include_model else 'no'}")
    print(f"Files: {len(result.files)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")


def print_verification_summary(result: PolicyPackageVerificationResult) -> None:
    print("Soridormi policy package verification")
    print("======================================")
    print(f"Package: {result.package_path}")
    print(f"Profile: {result.profile_name or '<unknown>'}")
    print(f"Files checked: {result.files_checked}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")


def _package_main(args: argparse.Namespace) -> PolicyPackageResult:
    return package_policy_profile(
        args.profile,
        output_dir=args.output_dir,
        robot_config_path=args.robot_config,
        include_model=args.include_model,
        require_model=args.require_model,
        check_model=args.check_model,
        providers=parse_provider_csv(args.providers) or None,
        require_providers=args.require_provider,
        prefer_cuda=not args.cpu,
        hash_model=not args.no_hash,
        profile_only=not args.full_suite,
        force=args.force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package or verify a Soridormi replacement policy handoff.")
    subparsers = parser.add_subparsers(dest="command")

    package = subparsers.add_parser("package", help="Create a replacement-policy tar.gz package")
    package.add_argument("profile", help="Policy profile name or YAML path")
    package.add_argument("--output-dir", default="data/policy_packages", help="Directory for generated .policy.tar.gz files")
    package.add_argument("--robot-config", default=None, help="Robot YAML path used for runtime contracts")
    package.add_argument("--include-model", action="store_true", help="Embed the ONNX model file when present")
    package.add_argument("--require-model", action="store_true", help="Fail if the ONNX model is missing")
    package.add_argument("--check-model", action="store_true", help="Load and validate ONNX metadata/providers")
    package.add_argument("--providers", default=None, help="Comma-separated ONNX Runtime provider order")
    package.add_argument("--require-provider", action="append", default=None, help="Required provider; may be repeated")
    package.add_argument("--cpu", action="store_true", help="Prefer CPU when checking model metadata")
    package.add_argument("--no-hash", action="store_true", help="Do not hash the ONNX model artifact")
    package.add_argument("--full-suite", action="store_true", help="Validate all profiles in the acceptance bundle")
    package.add_argument("--force", action="store_true", help="Allow replacing an existing output path")
    package.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    verify = subparsers.add_parser("verify", help="Verify a generated policy package")
    verify.add_argument("package", help="Path to .policy.tar.gz")
    verify.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    args = parser.parse_args()
    if args.command == "package":
        result = _package_main(args)
        if args.json:
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
        else:
            print_package_summary(result)
        if not result.ok:
            raise SystemExit(1)
    elif args.command == "verify":
        result = verify_policy_package(args.package)
        if args.json:
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
        else:
            print_verification_summary(result)
        if not result.ok:
            raise SystemExit(1)
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
