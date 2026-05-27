from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soridormi_runtime.linear_behavior_clone_policy import load_linear_behavior_clone_model
from soridormi_runtime.onnx_providers import resolve_onnx_providers, verify_active_providers
from soridormi_runtime.policy_profiles import PolicyProfile
from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file
from soridormi_runtime.train_behavior_clone import _load_prepared_manifest, _load_split_arrays, _path_from_manifest, predict_linear_behavior_clone

EVALUATION_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("data/training_evaluations")


@dataclass
class EvaluationSplitResult:
    name: str
    path: str
    sample_count: int
    mse: float | None = None
    rmse: float | None = None
    mae: float | None = None
    max_abs_error: float | None = None
    mean_abs_error_by_action: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class PolicyEvaluationResult:
    ok: bool
    profile_name: str
    profile_path: str
    model_kind: str
    model_path: str
    model_sha256: str | None
    prepared_manifest_path: str
    output_dir: str
    evaluation_path: str
    report_path: str
    prediction_paths: dict[str, str]
    observation_size: int
    action_size: int
    splits: dict[str, EvaluationSplitResult]
    thresholds: dict[str, float]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_kind(profile: PolicyProfile) -> str:
    return str(getattr(profile.model, "kind", "onnx") or "onnx").strip().lower().replace("-", "_")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _selected_splits(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return ["train", "val", "test"]
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = [str(item).strip() for item in value]
    out: list[str] = []
    for name in raw:
        if not name:
            continue
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {name!r}; expected train, val, or test")
        if name not in out:
            out.append(name)
    return out or ["train", "val", "test"]


def _metrics_for_predictions(
    name: str,
    path: Path,
    actions: np.ndarray,
    predictions: np.ndarray,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> EvaluationSplitResult:
    errors = list(errors or [])
    warnings = list(warnings or [])
    if errors:
        return EvaluationSplitResult(
            name=name,
            path=str(path),
            sample_count=int(actions.shape[0]),
            errors=errors,
            warnings=warnings,
        )
    if actions.shape[0] == 0:
        return EvaluationSplitResult(
            name=name,
            path=str(path),
            sample_count=0,
            warnings=warnings + ["split has no samples"],
        )
    if predictions.shape != actions.shape:
        return EvaluationSplitResult(
            name=name,
            path=str(path),
            sample_count=int(actions.shape[0]),
            errors=[f"prediction shape {list(predictions.shape)} != action shape {list(actions.shape)}"],
            warnings=warnings,
        )
    if not np.all(np.isfinite(predictions)):
        return EvaluationSplitResult(
            name=name,
            path=str(path),
            sample_count=int(actions.shape[0]),
            errors=["predictions contain non-finite values"],
            warnings=warnings,
        )
    diff = predictions - actions
    mse = float(np.mean(diff * diff))
    return EvaluationSplitResult(
        name=name,
        path=str(path),
        sample_count=int(actions.shape[0]),
        mse=mse,
        rmse=float(math.sqrt(mse)),
        mae=float(np.mean(np.abs(diff))),
        max_abs_error=float(np.max(np.abs(diff))),
        mean_abs_error_by_action=[float(x) for x in np.mean(np.abs(diff), axis=0).reshape(-1)],
        warnings=warnings,
    )


def _write_prediction_jsonl(path: Path, observations: np.ndarray, actions: np.ndarray, predictions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index in range(actions.shape[0]):
            action = actions[index].astype(float)
            prediction = predictions[index].astype(float)
            error = prediction - action
            payload = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "sample_index": index,
                "observation": [float(x) for x in observations[index].reshape(-1)],
                "action": [float(x) for x in action.reshape(-1)],
                "predicted_action": [float(x) for x in prediction.reshape(-1)],
                "error": [float(x) for x in error.reshape(-1)],
                "abs_error": [float(x) for x in np.abs(error).reshape(-1)],
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def _linear_predict(profile: PolicyProfile, observations: np.ndarray) -> tuple[np.ndarray, list[str], list[str], str | None]:
    model = load_linear_behavior_clone_model(profile.model.path)
    errors = list(model.errors)
    warnings = list(model.warnings)
    if errors:
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), errors, warnings, None
    normalization = {
        "observation_mean": model.observation_mean.astype(np.float64),
        "observation_std": model.observation_std.astype(np.float64),
        "action_mean": model.action_mean.astype(np.float64),
        "action_std": model.action_std.astype(np.float64),
    }
    predictions = predict_linear_behavior_clone(
        observations.astype(np.float64),
        weights=model.weights.astype(np.float64),
        bias=model.bias.astype(np.float64),
        normalization=normalization,
    )
    model_path = Path(profile.model.path)
    model_sha = sha256_file(model_path) if model_path.exists() else None
    return predictions.astype(np.float64), errors, warnings, model_sha


def _onnx_predict(
    profile: PolicyProfile,
    observations: np.ndarray,
    *,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> tuple[np.ndarray, list[str], list[str], str | None]:
    errors: list[str] = []
    warnings: list[str] = []
    model_path = Path(profile.model.path)
    if not model_path.exists():
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), [f"Policy model not found: {model_path}"], warnings, None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), [f"onnxruntime is not available: {exc!r}"], warnings, None

    available = list(ort.get_available_providers())
    selection = resolve_onnx_providers(
        available,
        requested=providers,
        required=require_providers,
        prefer_cuda=prefer_cuda,
    )
    if not selection.ok:
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), selection.errors, selection.warnings, sha256_file(model_path)
    warnings.extend(selection.warnings)
    try:
        session = ort.InferenceSession(str(model_path), providers=selection.providers)
    except Exception as exc:
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), [f"Failed to load ONNX policy: {exc!r}"], warnings, sha256_file(model_path)
    active = list(session.get_providers()) if hasattr(session, "get_providers") else list(selection.providers)
    errors.extend(verify_active_providers(active, requested=selection.requested, required=selection.required))
    if errors:
        return np.zeros((observations.shape[0], DEFAULT_ACTION_SIZE), dtype=np.float64), errors, warnings, sha256_file(model_path)

    input_name = str(profile.model.input_name or "obs")
    predictions: list[np.ndarray] = []
    for start in range(0, observations.shape[0], 1024):
        batch = observations[start : start + 1024].astype(np.float32)
        output = session.run([profile.model.output_name], {input_name: batch})[0]
        predictions.append(np.asarray(output, dtype=np.float64).reshape((batch.shape[0], -1)))
    if predictions:
        return np.concatenate(predictions, axis=0), errors, warnings, sha256_file(model_path)
    return np.zeros((0, DEFAULT_ACTION_SIZE), dtype=np.float64), errors, warnings, sha256_file(model_path)


def _predict_profile(
    profile: PolicyProfile,
    observations: np.ndarray,
    *,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
) -> tuple[np.ndarray, list[str], list[str], str | None]:
    kind = _model_kind(profile)
    if kind in {"linear", "linear_npz", "linear_behavior_clone", "behavior_clone_linear"}:
        return _linear_predict(profile, observations)
    return _onnx_predict(
        profile,
        observations,
        providers=providers,
        require_providers=require_providers,
        prefer_cuda=prefer_cuda,
    )


def _threshold_errors(result: EvaluationSplitResult, thresholds: dict[str, float]) -> list[str]:
    errors: list[str] = []
    for metric in ("mae", "rmse", "max_abs_error"):
        value = getattr(result, metric)
        limit = thresholds.get(f"{result.name}_{metric}")
        if value is not None and limit is not None and value > limit:
            errors.append(f"{result.name}.{metric} {value:.6g} exceeds threshold {limit:.6g}")
    return errors


def _write_report(path: Path, result: PolicyEvaluationResult) -> None:
    lines = [
        "# Soridormi offline policy evaluation",
        "",
        f"Profile: `{result.profile_name}`",
        f"Model kind: `{result.model_kind}`",
        f"Prepared manifest: `{result.prepared_manifest_path}`",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        "",
        "## Metrics",
        "",
        "| Split | Samples | MAE | RMSE | Max abs error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in result.splits.values():
        mae = "n/a" if split.mae is None else f"{split.mae:.6g}"
        rmse = "n/a" if split.rmse is None else f"{split.rmse:.6g}"
        max_abs = "n/a" if split.max_abs_error is None else f"{split.max_abs_error:.6g}"
        lines.append(f"| {split.name} | {split.sample_count} | {mae} | {rmse} | {max_abs} |")
    if result.prediction_paths:
        lines.extend(["", "## Prediction files", ""])
        for name, value in sorted(result.prediction_paths.items()):
            lines.append(f"- {name}: `{value}`")
    if result.thresholds:
        lines.extend(["", "## Thresholds", ""])
        for key, value in sorted(result.thresholds.items()):
            lines.append(f"- {key}: {value:.6g}")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_policy_profile(
    profile: str | Path | PolicyProfile,
    prepared: str | Path,
    *,
    output_dir: str | Path | None = None,
    splits: str | Iterable[str] | None = None,
    write_predictions: bool = False,
    max_samples_per_split: int | None = None,
    providers: list[str] | str | None = None,
    require_providers: list[str] | str | None = None,
    prefer_cuda: bool = True,
    max_train_mae: float | None = None,
    max_val_mae: float | None = None,
    max_test_mae: float | None = None,
    max_test_rmse: float | None = None,
    max_test_max_abs_error: float | None = None,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
) -> PolicyEvaluationResult:
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
    manifest_path, manifest, manifest_errors = _load_prepared_manifest(prepared)
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / f"{policy_profile.name}_{utc_stamp()}"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_path = output / "evaluation.json"
    report_path = output / "evaluation_report.md"

    selected = _selected_splits(splits)
    thresholds: dict[str, float] = {}
    for key, value in {
        "train_mae": max_train_mae,
        "val_mae": max_val_mae,
        "test_mae": max_test_mae,
        "test_rmse": max_test_rmse,
        "test_max_abs_error": max_test_max_abs_error,
    }.items():
        number = _safe_float(value)
        if number is not None:
            thresholds[key] = number

    errors: list[str] = list(manifest_errors)
    warnings: list[str] = []
    split_results: dict[str, EvaluationSplitResult] = {}
    prediction_paths: dict[str, str] = {}
    model_sha: str | None = None

    if manifest_errors:
        result = PolicyEvaluationResult(
            ok=False,
            profile_name=policy_profile.name,
            profile_path=str(policy_profile.path),
            model_kind=_model_kind(policy_profile),
            model_path=str(policy_profile.model.path),
            model_sha256=model_sha,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            evaluation_path=str(evaluation_path),
            report_path=str(report_path),
            prediction_paths=prediction_paths,
            observation_size=observation_size,
            action_size=action_size,
            splits=split_results,
            thresholds=thresholds,
            errors=errors,
            warnings=warnings,
        )
        evaluation_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_report(report_path, result)
        return result

    split_payloads = manifest.get("splits")
    if not isinstance(split_payloads, dict):
        errors.append("Prepared manifest is missing splits")
        split_payloads = {}

    for name in selected:
        payload = split_payloads.get(name)
        if not isinstance(payload, dict) or not payload.get("path"):
            split_results[name] = EvaluationSplitResult(name=name, path="", sample_count=0, errors=[f"Prepared manifest missing {name} split path"])
            continue
        path = _path_from_manifest(manifest_path, str(payload["path"]))
        observations, actions, split_errors, split_warnings = _load_split_arrays(
            path,
            observation_size=observation_size,
            action_size=action_size,
        )
        if max_samples_per_split is not None and max_samples_per_split > 0:
            observations = observations[:max_samples_per_split]
            actions = actions[:max_samples_per_split]
        if split_errors:
            split_results[name] = EvaluationSplitResult(
                name=name,
                path=str(path),
                sample_count=int(actions.shape[0]),
                errors=split_errors,
                warnings=split_warnings,
            )
            continue
        predictions, predict_errors, predict_warnings, model_sha_candidate = _predict_profile(
            policy_profile,
            observations,
            providers=providers,
            require_providers=require_providers,
            prefer_cuda=prefer_cuda,
        )
        if model_sha is None and model_sha_candidate is not None:
            model_sha = model_sha_candidate
        split = _metrics_for_predictions(
            name,
            path,
            actions,
            predictions,
            errors=predict_errors,
            warnings=split_warnings + predict_warnings,
        )
        threshold_errors = _threshold_errors(split, thresholds)
        if threshold_errors:
            split.errors.extend(threshold_errors)
        split_results[name] = split
        if write_predictions and split.ok:
            pred_path = output / f"predictions_{name}.jsonl"
            _write_prediction_jsonl(pred_path, observations, actions, predictions)
            prediction_paths[name] = str(pred_path)

    for split in split_results.values():
        errors.extend(f"{split.name}: {error}" for error in split.errors)
        warnings.extend(f"{split.name}: {warning}" for warning in split.warnings)

    result = PolicyEvaluationResult(
        ok=not errors,
        profile_name=policy_profile.name,
        profile_path=str(policy_profile.path),
        model_kind=_model_kind(policy_profile),
        model_path=str(policy_profile.model.path),
        model_sha256=model_sha,
        prepared_manifest_path=str(manifest_path),
        output_dir=str(output),
        evaluation_path=str(evaluation_path),
        report_path=str(report_path),
        prediction_paths=prediction_paths,
        observation_size=observation_size,
        action_size=action_size,
        splits=split_results,
        thresholds=thresholds,
        errors=errors,
        warnings=warnings,
    )
    payload = asdict(result)
    payload["schema_version"] = EVALUATION_SCHEMA_VERSION
    payload["evaluation_type"] = "soridormi.offline_policy_evaluation.v1"
    evaluation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def print_evaluation_summary(result: PolicyEvaluationResult) -> None:
    print("Soridormi offline policy evaluation")
    print("====================================")
    print(f"Profile: {result.profile_name}")
    print(f"Model kind: {result.model_kind}")
    print(f"Model: {result.model_path}")
    print(f"Prepared manifest: {result.prepared_manifest_path}")
    print(f"Output dir: {result.output_dir}")
    print(f"Evaluation: {result.evaluation_path}")
    print(f"Report: {result.report_path}")
    print("Metrics:")
    for split in result.splits.values():
        mae = "n/a" if split.mae is None else f"{split.mae:.6g}"
        rmse = "n/a" if split.rmse is None else f"{split.rmse:.6g}"
        max_abs = "n/a" if split.max_abs_error is None else f"{split.max_abs_error:.6g}"
        print(f"  {split.name}: samples={split.sample_count} mae={mae} rmse={rmse} max_abs={max_abs}")
    if result.prediction_paths:
        print("Prediction files:")
        for name, path in sorted(result.prediction_paths.items()):
            print(f"  {name}: {path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:30]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Soridormi policy profile offline on a prepared supervised dataset.")
    parser.add_argument("profile", help="Policy profile name or YAML path")
    parser.add_argument("prepared", type=Path, help="Prepared dataset directory or prepared_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evaluation artifacts")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to evaluate")
    parser.add_argument("--write-predictions", action="store_true", help="Write predictions_SPLIT.jsonl files")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Limit samples per split for quick smoke checks")
    parser.add_argument("--providers", default=None, help="Comma-separated ONNX Runtime providers for ONNX profiles")
    parser.add_argument("--require-provider", action="append", default=None, help="Required ONNX Runtime provider; repeatable")
    parser.add_argument("--no-prefer-cuda", action="store_true", help="Do not prefer CUDAExecutionProvider by default for ONNX profiles")
    parser.add_argument("--max-train-mae", type=float, default=None, help="Fail if train MAE exceeds this threshold")
    parser.add_argument("--max-val-mae", type=float, default=None, help="Fail if val MAE exceeds this threshold")
    parser.add_argument("--max-test-mae", type=float, default=None, help="Fail if test MAE exceeds this threshold")
    parser.add_argument("--max-test-rmse", type=float, default=None, help="Fail if test RMSE exceeds this threshold")
    parser.add_argument("--max-test-max-abs-error", type=float, default=None, help="Fail if test max absolute error exceeds this threshold")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    result = evaluate_policy_profile(
        args.profile,
        args.prepared,
        output_dir=args.output_dir,
        splits=args.splits,
        write_predictions=args.write_predictions,
        max_samples_per_split=args.max_samples_per_split,
        providers=args.providers,
        require_providers=args.require_provider,
        prefer_cuda=not args.no_prefer_cuda,
        max_train_mae=args.max_train_mae,
        max_val_mae=args.max_val_mae,
        max_test_mae=args.max_test_mae,
        max_test_rmse=args.max_test_rmse,
        max_test_max_abs_error=args.max_test_max_abs_error,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_evaluation_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
