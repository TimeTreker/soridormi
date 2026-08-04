from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soridormi_runtime.bc_training_contract import DEFAULT_CONTRACT_PATH
from soridormi_runtime.context_bc_dataset_prepare import PREPARED_CONTEXT_DATASET_TYPE
from soridormi_runtime.training_dataset import sha256_file

CONTEXT_BC_TRAINING_READY_SCHEMA_VERSION = "soridormi.context_bc_training_ready.v1"
DEFAULT_OUTPUT_DIR = Path("artifacts/training/context_bc/training_ready")
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class FileHashEntry:
    path: str
    exists: bool
    sha256: str | None = None


@dataclass
class TrainingCommandHints:
    linear_bc: list[str]
    neural_bc: list[str]


@dataclass
class ContextBcTrainingReadyResult:
    ok: bool
    schema_version: str
    generated_at_utc: str
    prepared_manifest_path: str
    contract_path: str
    scenario_gate_report_path: str
    prepared_gate_report_path: str
    output_dir: str | None
    prepared_dataset_type: str | None
    prepared_manifest_ok: bool | None
    scenario_gate_ok: bool | None
    prepared_gate_ok: bool | None
    total_sample_count: int
    split_sample_counts: dict[str, int]
    split_group_counts: dict[str, int]
    scenario_counts: dict[str, int]
    required_scenarios: list[str]
    file_hashes: dict[str, FileHashEntry]
    recommended_train_commands: TrainingCommandHints
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"{label} not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"failed to read {label} {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object: {path}")
        return {}
    return payload


def _file_hash(path: str | Path) -> FileHashEntry:
    item = Path(path)
    if not item.exists() or not item.is_file():
        return FileHashEntry(path=str(item), exists=False)
    return FileHashEntry(path=str(item), exists=True, sha256=sha256_file(item))


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return dict(sorted(result.items()))


def _split_payload(manifest: dict[str, Any], split_name: str) -> dict[str, Any]:
    splits = manifest.get("splits")
    if isinstance(splits, dict) and isinstance(splits.get(split_name), dict):
        return dict(splits[split_name])
    value = manifest.get(split_name)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _split_sample_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split = _split_payload(manifest, split_name)
        try:
            counts[split_name] = int(split.get("sample_count", 0))
        except (TypeError, ValueError):
            counts[split_name] = 0
    return counts


def _split_group_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split = _split_payload(manifest, split_name)
        try:
            counts[split_name] = int(split.get("group_count", 0))
        except (TypeError, ValueError):
            counts[split_name] = 0
    return counts


def _scenario_counts_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    counts = _int_dict(manifest.get("scenario_counts"))
    if counts:
        return counts
    merged: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        split_counts = _int_dict(_split_payload(manifest, split_name).get("scenario_counts"))
        for scenario, count in split_counts.items():
            merged[scenario] = merged.get(scenario, 0) + count
    return dict(sorted(merged.items()))


def _report_ok(payload: dict[str, Any]) -> bool | None:
    return _bool_or_none(payload.get("ok"))


def _required_scenarios(
    scenario_gate: dict[str, Any],
    prepared_gate: dict[str, Any],
) -> list[str]:
    values: list[str] = []
    for payload in (scenario_gate, prepared_gate):
        raw = payload.get("required_scenarios")
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item))
    return sorted(set(values))


def _command_hints(
    prepared_manifest_path: Path,
    *,
    profile_name: str,
    linear_output_dir: str,
    neural_output_dir: str,
    input_mode: str,
) -> TrainingCommandHints:
    manifest = str(prepared_manifest_path)
    return TrainingCommandHints(
        linear_bc=[
            "./scripts/train_behavior_clone.sh",
            manifest,
            "--output-dir",
            linear_output_dir,
            "--input-mode",
            input_mode,
            "--json",
        ],
        neural_bc=[
            "./scripts/train_neural_behavior_clone.sh",
            manifest,
            "--output-dir",
            neural_output_dir,
            "--input-mode",
            input_mode,
            "--profile-name",
            profile_name,
            "--force-profile",
            "--json",
        ],
    )


def build_training_ready_report(
    prepared_manifest_path: str | Path,
    *,
    scenario_gate_report_path: str | Path,
    prepared_gate_report_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    profile_name: str = "context_command_candidate",
    linear_output_dir: str | Path = "/data/training_runs/context_command_candidate_linear_bc",
    neural_output_dir: str | Path = "/data/training_runs/context_command_candidate_neural_bc",
    input_mode: str = "context_command_v1",
) -> ContextBcTrainingReadyResult:
    errors: list[str] = []
    warnings: list[str] = []
    prepared_path = Path(prepared_manifest_path)
    scenario_gate_path = Path(scenario_gate_report_path)
    prepared_gate_path = Path(prepared_gate_report_path)
    contract = Path(contract_path)

    prepared = _load_json(prepared_path, "prepared manifest", errors)
    scenario_gate = _load_json(scenario_gate_path, "scenario coverage gate report", errors)
    prepared_gate = _load_json(prepared_gate_path, "prepared dataset gate report", errors)

    dataset_type = (
        prepared.get("dataset_type") if isinstance(prepared.get("dataset_type"), str) else None
    )
    if dataset_type != PREPARED_CONTEXT_DATASET_TYPE:
        errors.append(
            f"prepared manifest dataset_type must be {PREPARED_CONTEXT_DATASET_TYPE}, "
            f"got {dataset_type!r}"
        )

    prepared_manifest_ok = _bool_or_none(prepared.get("ok"))
    scenario_gate_ok = _report_ok(scenario_gate)
    prepared_gate_ok = _report_ok(prepared_gate)
    if prepared_manifest_ok is not True:
        errors.append("prepared manifest ok is not true")
    if scenario_gate_ok is not True:
        errors.append("scenario coverage gate ok is not true")
    if prepared_gate_ok is not True:
        errors.append("prepared dataset gate ok is not true")

    split_sample_counts = _split_sample_counts(prepared)
    split_group_counts = _split_group_counts(prepared)
    total_sample_count = sum(split_sample_counts.values())
    if total_sample_count <= 0:
        errors.append("prepared dataset has no split samples")
    for split_name in SPLIT_NAMES:
        if split_sample_counts.get(split_name, 0) <= 0:
            errors.append(f"prepared split {split_name!r} has no samples")
        if split_group_counts.get(split_name, 0) <= 0:
            errors.append(f"prepared split {split_name!r} has no rollout groups")

    file_hashes: dict[str, FileHashEntry] = {
        "prepared_manifest": _file_hash(prepared_path),
        "contract": _file_hash(contract),
        "scenario_gate_report": _file_hash(scenario_gate_path),
        "prepared_gate_report": _file_hash(prepared_gate_path),
    }
    for split_name in SPLIT_NAMES:
        split_path = _split_payload(prepared, split_name).get("path")
        if split_path:
            file_hashes[f"split_{split_name}"] = _file_hash(str(split_path))
        else:
            errors.append(f"prepared manifest does not declare split {split_name!r} path")

    for name, entry in file_hashes.items():
        if not entry.exists:
            errors.append(f"required file for {name!r} does not exist: {entry.path}")

    result = ContextBcTrainingReadyResult(
        ok=not errors,
        schema_version=CONTEXT_BC_TRAINING_READY_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        prepared_manifest_path=str(prepared_path),
        contract_path=str(contract),
        scenario_gate_report_path=str(scenario_gate_path),
        prepared_gate_report_path=str(prepared_gate_path),
        output_dir=(
            str(prepared.get("output_dir")) if prepared.get("output_dir") is not None else None
        ),
        prepared_dataset_type=dataset_type,
        prepared_manifest_ok=prepared_manifest_ok,
        scenario_gate_ok=scenario_gate_ok,
        prepared_gate_ok=prepared_gate_ok,
        total_sample_count=total_sample_count,
        split_sample_counts=split_sample_counts,
        split_group_counts=split_group_counts,
        scenario_counts=_scenario_counts_from_manifest(prepared),
        required_scenarios=_required_scenarios(scenario_gate, prepared_gate),
        file_hashes=file_hashes,
        recommended_train_commands=_command_hints(
            prepared_path,
            profile_name=profile_name,
            linear_output_dir=str(linear_output_dir),
            neural_output_dir=str(neural_output_dir),
            input_mode=input_mode,
        ),
        errors=errors,
        warnings=warnings,
    )
    return result


def write_markdown_report(result: ContextBcTrainingReadyResult, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Soridormi Context BC Training-Ready Report",
        "",
        f"Result: **{'PASS' if result.ok else 'FAIL'}**",
        "",
        f"Prepared manifest: `{result.prepared_manifest_path}`",
        f"Contract: `{result.contract_path}`",
        f"Scenario gate: `{result.scenario_gate_report_path}`",
        f"Prepared gate: `{result.prepared_gate_report_path}`",
        f"Total samples: `{result.total_sample_count}`",
        "",
        "## Gate status",
        "",
        f"- Prepared manifest ok: `{result.prepared_manifest_ok}`",
        f"- Scenario coverage gate ok: `{result.scenario_gate_ok}`",
        f"- Prepared dataset gate ok: `{result.prepared_gate_ok}`",
        "",
        "## Splits",
        "",
        "| Split | Samples | Groups |",
        "|---|---:|---:|",
    ]
    for split_name in SPLIT_NAMES:
        lines.append(
            f"| {split_name} | {result.split_sample_counts.get(split_name, 0)} | "
            f"{result.split_group_counts.get(split_name, 0)} |"
        )
    if result.scenario_counts:
        lines.extend(["", "## Scenario counts", ""])
        lines.extend(
            f"- `{scenario}`: {count}" for scenario, count in sorted(result.scenario_counts.items())
        )
    lines.extend(
        ["", "## File hashes", "", "| Name | Exists | SHA256 | Path |", "|---|---|---|---|"]
    )
    for name, entry in sorted(result.file_hashes.items()):
        lines.append(f"| `{name}` | {entry.exists} | `{entry.sha256}` | `{entry.path}` |")
    lines.extend(
        [
            "",
            "## Recommended train commands",
            "",
            "Linear BC:",
            "",
            "```bash",
            " ".join(result.recommended_train_commands.linear_bc),
            "```",
            "",
            "Neural BC:",
            "",
            "```bash",
            " ".join(result.recommended_train_commands.neural_bc),
            "```",
        ]
    )
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in result.errors)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in result.warnings)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(result: ContextBcTrainingReadyResult) -> None:
    print("Soridormi context BC training-ready report")
    print("==========================================")
    print(f"Prepared manifest: {result.prepared_manifest_path}")
    print(f"Samples: {result.total_sample_count}")
    print(f"Scenario gate ok: {result.scenario_gate_ok}")
    print(f"Prepared gate ok: {result.prepared_gate_ok}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")
    if result.errors:
        print("Errors:")
        for item in result.errors[:40]:
            print(f"  - {item}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle context BC gate outputs into a training-ready manifest/report."
    )
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument(
        "--scenario-gate",
        type=Path,
        required=True,
        help="dataset_scenario_gate_summary.json",
    )
    parser.add_argument(
        "--prepared-gate",
        type=Path,
        required=True,
        help="prepared_context_gate_report.json",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--profile-name", default="context_command_candidate")
    parser.add_argument(
        "--linear-output-dir",
        default="/data/training_runs/context_command_candidate_linear_bc",
    )
    parser.add_argument(
        "--neural-output-dir",
        default="/data/training_runs/context_command_candidate_neural_bc",
    )
    parser.add_argument("--input-mode", default="context_command_v1")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write training_ready_manifest.json/md here",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_training_ready_report(
        args.prepared_manifest,
        scenario_gate_report_path=args.scenario_gate,
        prepared_gate_report_path=args.prepared_gate,
        contract_path=args.contract,
        profile_name=args.profile_name,
        linear_output_dir=args.linear_output_dir,
        neural_output_dir=args.neural_output_dir,
        input_mode=args.input_mode,
    )
    output_path = args.output
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "training_ready_manifest.json"
        json_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_path is None:
            output_path = args.output_dir / "training_ready_report.md"
    if output_path is not None:
        write_markdown_report(result, output_path)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
