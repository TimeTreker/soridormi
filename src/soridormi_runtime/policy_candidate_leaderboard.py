"""Offline candidate leaderboard for trained Soridormi policy profiles.

M6.8 sits after training/evaluation/package workflows.  It scans policy
``evaluation.json`` artifacts, ranks candidates by held-out metrics, and writes a
small promotion-oriented report.  It does not modify runtime profiles or launch
simulation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_SEARCH_ROOTS = (Path("data/training_evaluations"), Path("data/training_pipelines"))
DEFAULT_OUTPUT_ROOT = Path("data/policy_candidate_leaderboards")
LEADERBOARD_SCHEMA_VERSION = 1


@dataclass
class CandidateThresholds:
    max_test_mae: float | None = None
    max_test_rmse: float | None = None
    max_test_max_abs_error: float | None = None

    def to_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        if self.max_test_mae is not None:
            payload["max_test_mae"] = float(self.max_test_mae)
        if self.max_test_rmse is not None:
            payload["max_test_rmse"] = float(self.max_test_rmse)
        if self.max_test_max_abs_error is not None:
            payload["max_test_max_abs_error"] = float(self.max_test_max_abs_error)
        return payload


@dataclass
class CandidateSummary:
    rank: int | None
    profile_name: str
    evaluation_path: str
    output_dir: str | None
    ok: bool
    promotable: bool
    model_kind: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    train_mae: float | None = None
    val_mae: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_max_abs_error: float | None = None
    train_sample_count: int | None = None
    val_sample_count: int | None = None
    test_sample_count: int | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PolicyCandidateLeaderboardResult:
    ok: bool
    generated_at_utc: str
    output_dir: str
    leaderboard_path: str
    report_path: str
    search_paths: list[str]
    thresholds: dict[str, float]
    best_profile: str | None
    best_evaluation_path: str | None
    candidate_count: int
    promotable_count: int
    candidates: list[CandidateSummary]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LEADERBOARD_SCHEMA_VERSION
        payload["leaderboard_type"] = "soridormi.policy_candidate_leaderboard.v1"
        return payload


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"Expected JSON object in {path}"
    return payload, None


def _split(payload: dict[str, Any], name: str) -> dict[str, Any]:
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return {}
    split = splits.get(name)
    return split if isinstance(split, dict) else {}


def _candidate_from_evaluation(path: Path, payload: dict[str, Any], thresholds: CandidateThresholds) -> CandidateSummary:
    profile_name = str(payload.get("profile_name") or path.parent.name or "<unknown>")
    train = _split(payload, "train")
    val = _split(payload, "val")
    test = _split(payload, "test")
    errors = [str(item) for item in payload.get("errors", []) if isinstance(item, str)]
    warnings = [str(item) for item in payload.get("warnings", []) if isinstance(item, str)]

    for split_name, split in (("train", train), ("val", val), ("test", test)):
        for item in split.get("errors", []) if isinstance(split.get("errors"), list) else []:
            if isinstance(item, str):
                errors.append(f"{split_name}: {item}")
        for item in split.get("warnings", []) if isinstance(split.get("warnings"), list) else []:
            if isinstance(item, str):
                warnings.append(f"{split_name}: {item}")

    ok = bool(payload.get("ok")) and not errors
    test_mae = _safe_float(test.get("mae"))
    test_rmse = _safe_float(test.get("rmse"))
    test_max_abs = _safe_float(test.get("max_abs_error"))
    threshold_errors: list[str] = []
    if thresholds.max_test_mae is not None and (test_mae is None or test_mae > thresholds.max_test_mae):
        value = "n/a" if test_mae is None else f"{test_mae:.6g}"
        threshold_errors.append(f"test MAE {value} exceeds {thresholds.max_test_mae:.6g}")
    if thresholds.max_test_rmse is not None and (test_rmse is None or test_rmse > thresholds.max_test_rmse):
        value = "n/a" if test_rmse is None else f"{test_rmse:.6g}"
        threshold_errors.append(f"test RMSE {value} exceeds {thresholds.max_test_rmse:.6g}")
    if thresholds.max_test_max_abs_error is not None and (
        test_max_abs is None or test_max_abs > thresholds.max_test_max_abs_error
    ):
        value = "n/a" if test_max_abs is None else f"{test_max_abs:.6g}"
        threshold_errors.append(f"test max abs error {value} exceeds {thresholds.max_test_max_abs_error:.6g}")

    errors.extend(threshold_errors)
    return CandidateSummary(
        rank=None,
        profile_name=profile_name,
        evaluation_path=str(path),
        output_dir=str(payload.get("output_dir")) if payload.get("output_dir") is not None else None,
        ok=ok,
        promotable=ok and not threshold_errors,
        model_kind=str(payload.get("model_kind")) if payload.get("model_kind") is not None else None,
        model_path=str(payload.get("model_path")) if payload.get("model_path") is not None else None,
        model_sha256=str(payload.get("model_sha256")) if payload.get("model_sha256") is not None else None,
        train_mae=_safe_float(train.get("mae")),
        val_mae=_safe_float(val.get("mae")),
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_max_abs_error=test_max_abs,
        train_sample_count=int(train["sample_count"]) if isinstance(train.get("sample_count"), int) else None,
        val_sample_count=int(val["sample_count"]) if isinstance(val.get("sample_count"), int) else None,
        test_sample_count=int(test["sample_count"]) if isinstance(test.get("sample_count"), int) else None,
        errors=errors,
        warnings=warnings,
    )


def _candidate_sort_key(candidate: CandidateSummary) -> tuple[int, float, float, float, str]:
    # Promotable candidates first, then lowest held-out errors. Missing metrics
    # sort after real finite metrics.
    missing = 1e30
    return (
        0 if candidate.promotable else 1,
        candidate.test_mae if candidate.test_mae is not None else missing,
        candidate.test_rmse if candidate.test_rmse is not None else missing,
        candidate.val_mae if candidate.val_mae is not None else missing,
        candidate.profile_name,
    )


def find_evaluation_files(paths: Sequence[str | Path] | None = None) -> tuple[list[Path], list[str]]:
    """Find evaluation.json files from files/directories.

    Direct files are accepted. Directories are searched recursively for files
    named ``evaluation.json``. Missing paths are returned as warnings rather than
    hard errors so a leaderboard can be run before any candidates exist.
    """

    roots = [Path(path) for path in (paths or DEFAULT_SEARCH_ROOTS)]
    found: list[Path] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for root in roots:
        matches: Iterable[Path]
        if root.is_file():
            matches = [root]
        elif root.is_dir():
            matches = root.rglob("evaluation.json")
        else:
            warnings.append(f"Search path not found: {root}")
            continue
        for path in matches:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return sorted(found), warnings


def build_policy_candidate_leaderboard(
    paths: Sequence[str | Path] | None = None,
    *,
    output_dir: str | Path | None = None,
    max_test_mae: float | None = None,
    max_test_rmse: float | None = None,
    max_test_max_abs_error: float | None = None,
    require_promotable: bool = False,
) -> PolicyCandidateLeaderboardResult:
    thresholds = CandidateThresholds(
        max_test_mae=max_test_mae,
        max_test_rmse=max_test_rmse,
        max_test_max_abs_error=max_test_max_abs_error,
    )
    search_paths = [str(path) for path in (paths or DEFAULT_SEARCH_ROOTS)]
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / utc_stamp()
    output.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output / "candidate_leaderboard.json"
    report_path = output / "candidate_leaderboard.md"

    evaluation_files, warnings = find_evaluation_files(paths)
    candidates: list[CandidateSummary] = []
    errors: list[str] = []
    for evaluation_path in evaluation_files:
        payload, error = _load_json(evaluation_path)
        if error is not None or payload is None:
            errors.append(error or f"Could not load {evaluation_path}")
            continue
        candidates.append(_candidate_from_evaluation(evaluation_path, payload, thresholds))

    candidates.sort(key=_candidate_sort_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank

    promotable = [candidate for candidate in candidates if candidate.promotable]
    if not candidates:
        warnings.append("No evaluation.json files found")
    if require_promotable and not promotable:
        errors.append("No promotable candidate found")

    best = promotable[0] if promotable else None
    result = PolicyCandidateLeaderboardResult(
        ok=not errors,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        output_dir=str(output),
        leaderboard_path=str(leaderboard_path),
        report_path=str(report_path),
        search_paths=search_paths,
        thresholds=thresholds.to_dict(),
        best_profile=best.profile_name if best is not None else None,
        best_evaluation_path=best.evaluation_path if best is not None else None,
        candidate_count=len(candidates),
        promotable_count=len(promotable),
        candidates=candidates,
        errors=errors,
        warnings=warnings,
    )
    leaderboard_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _write_report(path: Path, result: PolicyCandidateLeaderboardResult) -> None:
    lines = [
        "# Soridormi policy candidate leaderboard",
        "",
        f"Generated: `{result.generated_at_utc}`",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        f"Candidates: {result.candidate_count}",
        f"Promotable: {result.promotable_count}",
    ]
    if result.best_profile:
        lines.append(f"Best candidate: `{result.best_profile}`")
    if result.thresholds:
        lines.extend(["", "## Thresholds", ""])
        for name, value in sorted(result.thresholds.items()):
            lines.append(f"- {name}: {value:.6g}")
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            "| Rank | Profile | Promotable | Test MAE | Test RMSE | Test max abs | Val MAE | Model kind | Evaluation |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for candidate in result.candidates:
        rank = "n/a" if candidate.rank is None else str(candidate.rank)
        lines.append(
            "| "
            + " | ".join(
                [
                    rank,
                    f"`{candidate.profile_name}`",
                    "yes" if candidate.promotable else "no",
                    _fmt(candidate.test_mae),
                    _fmt(candidate.test_rmse),
                    _fmt(candidate.test_max_abs_error),
                    _fmt(candidate.val_mae),
                    candidate.model_kind or "n/a",
                    f"`{candidate.evaluation_path}`",
                ]
            )
            + " |"
        )
    for section, items in (("Warnings", result.warnings), ("Errors", result.errors)):
        if items:
            lines.extend(["", f"## {section}", ""])
            lines.extend(f"- {item}" for item in items)
    failing = [candidate for candidate in result.candidates if candidate.errors]
    if failing:
        lines.extend(["", "## Candidate issues", ""])
        for candidate in failing:
            lines.append(f"### `{candidate.profile_name}`")
            lines.extend(f"- {error}" for error in candidate.errors)
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"expected finite float, got {value!r}")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rank evaluated Soridormi replacement policy candidates.")
    parser.add_argument("paths", nargs="*", help="Evaluation JSON files or directories to scan. Defaults to data training roots.")
    parser.add_argument("--output-dir", default=None, help="Directory for candidate_leaderboard.json/md.")
    parser.add_argument("--max-test-mae", type=_parse_float, default=None)
    parser.add_argument("--max-test-rmse", type=_parse_float, default=None)
    parser.add_argument("--max-test-max-abs-error", type=_parse_float, default=None)
    parser.add_argument("--require-promotable", action="store_true", help="Fail if no candidate passes all thresholds.")
    parser.add_argument("--json", action="store_true", help="Print JSON result instead of a text summary.")
    args = parser.parse_args(argv)

    result = build_policy_candidate_leaderboard(
        args.paths or None,
        output_dir=args.output_dir,
        max_test_mae=args.max_test_mae,
        max_test_rmse=args.max_test_rmse,
        max_test_max_abs_error=args.max_test_max_abs_error,
        require_promotable=args.require_promotable,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Soridormi policy candidate leaderboard")
        print("=====================================")
        print(f"Candidates: {result.candidate_count}")
        print(f"Promotable: {result.promotable_count}")
        if result.best_profile:
            print(f"Best: {result.best_profile}")
        print(f"Report: {result.report_path}")
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        print(f"Result: {'OK' if result.ok else 'FAILED'}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
