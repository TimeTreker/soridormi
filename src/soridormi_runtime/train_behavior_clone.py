from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soridormi_runtime.policy_input_features import (
    CONTEXT_COMMAND_V1_FIELDS,
    INPUT_MODE_CONTEXT_COMMAND_V1,
    INPUT_MODE_OBSERVATION,
    INPUT_MODES,
    input_size_for,
)
from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file

BEHAVIOR_CLONE_SCHEMA_VERSION = 1
DEFAULT_RIDGE_LAMBDA = 1e-4
PREPARED_DATASET_TYPES = {
    "soridormi.policy_supervision.prepared.v1",
    "soridormi.policy_supervision.context_prepared.v1",
}


@dataclass
class SplitMetrics:
    name: str
    sample_count: int
    mse: float | None = None
    rmse: float | None = None
    mae: float | None = None
    max_abs_error: float | None = None
    path: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BehaviorCloneTrainResult:
    ok: bool
    prepared_manifest_path: str
    output_dir: str
    model_path: str
    metrics_path: str
    report_path: str
    normalization_path: str | None
    ridge_lambda: float
    observation_size: int
    action_size: int
    metrics: dict[str, SplitMetrics]
    train_sample_count: int = 0
    input_mode: str = INPUT_MODE_OBSERVATION
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_manifest_path(prepared: str | Path) -> Path:
    path = Path(prepared)
    if path.is_dir():
        return path / "prepared_manifest.json"
    return path


def _path_from_manifest(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"File not found: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"Invalid JSON in {path}: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"Expected JSON object in {path}"]
    return payload, []


def _load_prepared_manifest(prepared: str | Path) -> tuple[Path, dict[str, Any], list[str]]:
    manifest_path = _resolve_manifest_path(prepared)
    manifest, errors = _load_json(manifest_path)
    if errors:
        return manifest_path, manifest, errors
    dataset_type = manifest.get("dataset_type")
    if dataset_type not in PREPARED_DATASET_TYPES:
        allowed = ", ".join(sorted(PREPARED_DATASET_TYPES))
        errors.append(f"Prepared manifest dataset_type must be one of: {allowed}")
    if not isinstance(manifest.get("splits"), dict):
        errors.append("Prepared manifest is missing splits")
    return manifest_path, manifest, errors


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _vector(value: Any, *, size: int, field_name: str) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{field_name} must be a list"
    if len(value) != size:
        return None, f"{field_name} size {len(value)} != expected {size}"
    bad = [index for index, item in enumerate(value) if not _is_finite_number(item)]
    if bad:
        preview = ", ".join(str(index) for index in bad[:8])
        return None, f"{field_name} contains non-finite/non-numeric values at indices {preview}"
    return [float(item) for item in value], None


def _input_size_for(input_mode: str, *, robot_observation_size: int) -> int:
    return input_size_for(input_mode, robot_observation_size=robot_observation_size)


def _observation_value_from_sample(sample: dict[str, Any]) -> Any:
    observation_value = sample.get("observation")
    if observation_value is None:
        robot_state = sample.get("robot_state")
        if isinstance(robot_state, dict):
            observation_value = robot_state.get("observation")
    return observation_value


def _action_value_from_sample(sample: dict[str, Any]) -> Any:
    action_value = sample.get("action")
    if action_value is None:
        action_value = sample.get("teacher_action")
    return action_value


def _policy_input_from_sample(
    sample: dict[str, Any],
    *,
    input_mode: str,
    robot_observation_size: int,
) -> tuple[list[float] | None, str | None]:
    observation, observation_error = _vector(
        _observation_value_from_sample(sample),
        size=robot_observation_size,
        field_name="observation",
    )
    if observation_error is not None:
        return None, observation_error
    assert observation is not None

    if input_mode == INPUT_MODE_OBSERVATION:
        return observation, None

    if input_mode == INPUT_MODE_CONTEXT_COMMAND_V1:
        desired_command = sample.get("desired_command")
        if not isinstance(desired_command, dict):
            return None, "desired_command must be an object for context_command_v1 input mode"
        command_values: list[float] = []
        for field_name in CONTEXT_COMMAND_V1_FIELDS:
            value = desired_command.get(field_name)
            if not _is_finite_number(value):
                return None, f"desired_command.{field_name} must be a finite number"
            command_values.append(float(value))
        return observation + command_values, None

    return None, f"Unsupported input mode {input_mode!r}; use one of: {', '.join(sorted(INPUT_MODES))}"


def _load_split_arrays(
    path: Path,
    *,
    observation_size: int,
    action_size: int,
    input_mode: str = INPUT_MODE_OBSERVATION,
    max_reported_issues: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    input_size = _input_size_for(input_mode, robot_observation_size=observation_size)
    observations: list[list[float]] = []
    actions: list[list[float]] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return (
            np.zeros((0, input_size), dtype=np.float64),
            np.zeros((0, action_size), dtype=np.float64),
            [f"Split file not found: {path}"],
            warnings,
        )

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                sample = json.loads(text)
            except json.JSONDecodeError as exc:
                if len(errors) < max_reported_issues:
                    errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(sample, dict):
                if len(errors) < max_reported_issues:
                    errors.append(f"line {line_number}: expected JSON object")
                continue
            obs, obs_error = _policy_input_from_sample(
                sample,
                input_mode=input_mode,
                robot_observation_size=observation_size,
            )
            action, action_error = _vector(
                _action_value_from_sample(sample),
                size=action_size,
                field_name="action",
            )
            sample_errors = [error for error in (obs_error, action_error) if error is not None]
            if sample_errors:
                if len(errors) < max_reported_issues:
                    errors.extend(f"line {line_number}: {error}" for error in sample_errors[: max_reported_issues - len(errors)])
                continue
            assert obs is not None and action is not None
            observations.append(obs)
            actions.append(action)

    if not observations and not errors:
        warnings.append(f"Split file has no samples: {path}")
    return (
        np.asarray(observations, dtype=np.float64).reshape((-1, input_size)),
        np.asarray(actions, dtype=np.float64).reshape((-1, action_size)),
        errors,
        warnings,
    )


def _default_normalization_path(prepared_manifest_path: Path, input_mode: str = INPUT_MODE_OBSERVATION) -> Path:
    if input_mode != INPUT_MODE_OBSERVATION:
        return prepared_manifest_path.parent / f"normalization.{input_mode}.json"
    return prepared_manifest_path.parent / "normalization.json"


def _normalization_from_payload(
    payload: dict[str, Any],
    *,
    observation_size: int,
    action_size: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    errors: list[str] = []
    fields = {
        "observation_mean": observation_size,
        "observation_std": observation_size,
        "action_mean": action_size,
        "action_std": action_size,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, size in fields.items():
        values, error = _vector(payload.get(name), size=size, field_name=name)
        if error is not None:
            errors.append(error)
            arrays[name] = np.zeros(size, dtype=np.float64)
        else:
            assert values is not None
            arrays[name] = np.asarray(values, dtype=np.float64)
    for name in ("observation_std", "action_std"):
        if np.any(arrays[name] <= 0.0):
            errors.append(f"{name} must contain positive values")
    return arrays, errors


def _load_or_make_normalization(
    normalization_path: Path | None,
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    *,
    observation_size: int,
    action_size: int,
    input_mode: str = INPUT_MODE_OBSERVATION,
) -> tuple[dict[str, np.ndarray], Path | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if normalization_path is not None and normalization_path.exists():
        payload, load_errors = _load_json(normalization_path)
        if load_errors:
            return {}, normalization_path, load_errors, warnings
        arrays, norm_errors = _normalization_from_payload(
            payload,
            observation_size=observation_size,
            action_size=action_size,
        )
        errors.extend(norm_errors)
        return arrays, normalization_path, errors, warnings

    if normalization_path is not None:
        warnings.append(f"Normalization file not found; computed train-only normalization inline: {normalization_path}")
    else:
        warnings.append("No normalization file supplied; computed train-only normalization inline")
    if train_observations.shape[0] == 0:
        return {}, normalization_path, ["Cannot compute normalization without train samples"], warnings
    obs_std = np.std(train_observations, axis=0)
    action_std = np.std(train_actions, axis=0)
    arrays = {
        "observation_mean": np.mean(train_observations, axis=0),
        "observation_std": np.where(obs_std > 1e-6, obs_std, 1e-6),
        "action_mean": np.mean(train_actions, axis=0),
        "action_std": np.where(action_std > 1e-6, action_std, 1e-6),
    }
    if normalization_path is not None:
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "normalization_type": "soridormi.policy_supervision.normalization.v1",
            "input_mode": input_mode,
            "source": "computed_from_train_split",
            "observation_mean": [float(x) for x in arrays["observation_mean"].tolist()],
            "observation_std": [float(x) for x in arrays["observation_std"].tolist()],
            "action_mean": [float(x) for x in arrays["action_mean"].tolist()],
            "action_std": [float(x) for x in arrays["action_std"].tolist()],
        }
        normalization_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        warnings[-1] = f"Normalization file not found; wrote train-only normalization: {normalization_path}"
    return arrays, normalization_path, errors, warnings


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean.reshape((1, -1))) / std.reshape((1, -1))


def _denormalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std.reshape((1, -1)) + mean.reshape((1, -1))


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(features.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[-1, -1] = 0.0
    lhs = features.T @ features + penalty
    rhs = features.T @ y
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(lhs) @ rhs
    return solution[:-1, :], solution[-1, :]


def predict_linear_behavior_clone(
    observations: np.ndarray,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> np.ndarray:
    x = _normalize(observations, normalization["observation_mean"], normalization["observation_std"])
    y_norm = x @ weights + bias.reshape((1, -1))
    return _denormalize(y_norm, normalization["action_mean"], normalization["action_std"])


def _metrics_for(
    name: str,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    normalization: dict[str, np.ndarray],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> SplitMetrics:
    errors = list(errors or [])
    warnings = list(warnings or [])
    if errors:
        return SplitMetrics(name=name, sample_count=int(actions.shape[0]), path=str(path), errors=errors, warnings=warnings)
    if actions.shape[0] == 0:
        return SplitMetrics(name=name, sample_count=0, path=str(path), warnings=warnings + ["split has no samples"])
    prediction = predict_linear_behavior_clone(
        observations,
        weights=weights,
        bias=bias,
        normalization=normalization,
    )
    diff = prediction - actions
    mse = float(np.mean(diff * diff))
    return SplitMetrics(
        name=name,
        sample_count=int(actions.shape[0]),
        mse=mse,
        rmse=float(math.sqrt(mse)),
        mae=float(np.mean(np.abs(diff))),
        max_abs_error=float(np.max(np.abs(diff))),
        path=str(path),
        warnings=warnings,
    )


def _write_report(path: Path, result: BehaviorCloneTrainResult) -> None:
    lines = [
        "# Soridormi behavior cloning baseline",
        "",
        f"Prepared manifest: `{result.prepared_manifest_path}`",
        f"Model: `{result.model_path}`",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        "",
        "## Metrics",
        "",
        "| Split | Samples | MAE | RMSE | Max abs error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in result.metrics.values():
        mae = "n/a" if split.mae is None else f"{split.mae:.6g}"
        rmse = "n/a" if split.rmse is None else f"{split.rmse:.6g}"
        max_abs = "n/a" if split.max_abs_error is None else f"{split.max_abs_error:.6g}"
        lines.append(f"| {split.name} | {split.sample_count} | {mae} | {rmse} | {max_abs} |")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_behavior_clone_baseline(
    prepared: str | Path,
    *,
    output_dir: str | Path | None = None,
    normalization_path: str | Path | None = None,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
    input_mode: str = INPUT_MODE_OBSERVATION,
) -> BehaviorCloneTrainResult:
    manifest_path, manifest, manifest_errors = _load_prepared_manifest(prepared)
    try:
        model_observation_size = _input_size_for(input_mode, robot_observation_size=observation_size)
    except ValueError as exc:
        model_observation_size = observation_size
        manifest_errors.append(str(exc))
    output = Path(output_dir) if output_dir is not None else manifest_path.parent / "behavior_clone_baseline"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "linear_behavior_clone.npz"
    metrics_path = output / "train_metrics.json"
    report_path = output / "train_report.md"

    empty_metrics = {
        name: SplitMetrics(name=name, sample_count=0)
        for name in ("train", "val", "test")
    }
    if manifest_errors:
        result = BehaviorCloneTrainResult(
            ok=False,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            report_path=str(report_path),
            normalization_path=str(normalization_path) if normalization_path is not None else None,
            ridge_lambda=ridge_lambda,
            observation_size=model_observation_size,
            action_size=action_size,
            metrics=empty_metrics,
            input_mode=input_mode,
            errors=manifest_errors,
        )
        metrics_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_report(report_path, result)
        return result

    split_payloads = manifest["splits"]
    split_paths = {
        name: _path_from_manifest(manifest_path, str(split_payloads[name]["path"]))
        for name in ("train", "val", "test")
        if isinstance(split_payloads.get(name), dict) and split_payloads[name].get("path")
    }
    errors: list[str] = []
    warnings: list[str] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]] = {}
    for name in ("train", "val", "test"):
        path = split_paths.get(name)
        if path is None:
            arrays[name] = (
                np.zeros((0, model_observation_size), dtype=np.float64),
                np.zeros((0, action_size), dtype=np.float64),
                [f"Prepared manifest missing {name} split path"],
                [],
            )
        else:
            arrays[name] = _load_split_arrays(
                path,
                observation_size=observation_size,
                action_size=action_size,
                input_mode=input_mode,
            )
        errors.extend(f"{name}: {error}" for error in arrays[name][2])
        warnings.extend(f"{name}: {warning}" for warning in arrays[name][3])
    train_observations, train_actions, _train_errors, _train_warnings = arrays["train"]
    if train_observations.shape[0] == 0:
        errors.append("train split has no valid samples")

    norm_path = Path(normalization_path) if normalization_path is not None else _default_normalization_path(manifest_path, input_mode)
    normalization, used_norm_path, norm_errors, norm_warnings = _load_or_make_normalization(
        norm_path,
        train_observations,
        train_actions,
        observation_size=model_observation_size,
        action_size=action_size,
        input_mode=input_mode,
    )
    errors.extend(norm_errors)
    warnings.extend(norm_warnings)

    if not math.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        errors.append("ridge_lambda must be a finite non-negative number")

    if errors:
        result = BehaviorCloneTrainResult(
            ok=False,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            report_path=str(report_path),
            normalization_path=str(used_norm_path) if used_norm_path is not None else None,
            ridge_lambda=ridge_lambda,
            observation_size=model_observation_size,
            action_size=action_size,
            metrics=empty_metrics,
            train_sample_count=int(train_observations.shape[0]),
            input_mode=input_mode,
            errors=errors,
            warnings=warnings,
        )
        metrics_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_report(report_path, result)
        return result

    x_train = _normalize(train_observations, normalization["observation_mean"], normalization["observation_std"])
    y_train = _normalize(train_actions, normalization["action_mean"], normalization["action_std"])
    weights, bias = _fit_ridge(x_train, y_train, ridge_lambda=ridge_lambda)

    metrics: dict[str, SplitMetrics] = {}
    for name in ("train", "val", "test"):
        observations, actions, split_errors, split_warnings = arrays[name]
        metrics[name] = _metrics_for(
            name,
            observations,
            actions,
            path=split_paths.get(name, Path("")),
            weights=weights,
            bias=bias,
            normalization=normalization,
            errors=split_errors,
            warnings=split_warnings,
        )

    np.savez(
        model_path,
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
        observation_mean=normalization["observation_mean"].astype(np.float32),
        observation_std=normalization["observation_std"].astype(np.float32),
        action_mean=normalization["action_mean"].astype(np.float32),
        action_std=normalization["action_std"].astype(np.float32),
        observation_size=np.asarray([model_observation_size], dtype=np.int64),
        action_size=np.asarray([action_size], dtype=np.int64),
        ridge_lambda=np.asarray([ridge_lambda], dtype=np.float64),
        input_mode=np.asarray([input_mode]),
    )

    result = BehaviorCloneTrainResult(
        ok=True,
        prepared_manifest_path=str(manifest_path),
        output_dir=str(output),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        normalization_path=str(used_norm_path) if used_norm_path is not None else None,
        ridge_lambda=ridge_lambda,
        observation_size=model_observation_size,
        action_size=action_size,
        metrics=metrics,
        train_sample_count=int(train_observations.shape[0]),
        input_mode=input_mode,
        warnings=warnings,
    )
    payload = asdict(result)
    payload["schema_version"] = BEHAVIOR_CLONE_SCHEMA_VERSION
    payload["training_run_type"] = "soridormi.policy_supervision.linear_behavior_clone.v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["model_sha256"] = sha256_file(model_path)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def print_train_summary(result: BehaviorCloneTrainResult) -> None:
    print("Soridormi behavior cloning baseline training")
    print("=============================================")
    print(f"Prepared manifest: {result.prepared_manifest_path}")
    print(f"Output dir: {result.output_dir}")
    print(f"Model: {result.model_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Report: {result.report_path}")
    print(f"Normalization: {result.normalization_path or 'inline'}")
    print(f"Input mode: {result.input_mode}")
    print(f"Train samples: {result.train_sample_count}")
    print("Metrics:")
    for split in result.metrics.values():
        mae = "n/a" if split.mae is None else f"{split.mae:.6g}"
        rmse = "n/a" if split.rmse is None else f"{split.rmse:.6g}"
        print(f"  {split.name}: samples={split.sample_count} mae={mae} rmse={rmse}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:20]:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deterministic linear behavior-cloning baseline from a prepared Soridormi dataset.")
    parser.add_argument("prepared", type=Path, help="Prepared dataset directory or prepared_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for training artifacts")
    parser.add_argument("--normalization", type=Path, default=None, help="normalization.json path; defaults to PREPARED/normalization.json")
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODES), default=INPUT_MODE_OBSERVATION)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    args = parser.parse_args()

    result = train_behavior_clone_baseline(
        args.prepared,
        output_dir=args.output_dir,
        normalization_path=args.normalization,
        ridge_lambda=args.ridge_lambda,
        observation_size=args.observation_size,
        action_size=args.action_size,
        input_mode=args.input_mode,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_train_summary(result)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
