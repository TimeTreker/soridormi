from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soridormi_runtime.create_policy_profile import create_replacement_profile
from soridormi_runtime.train_behavior_clone import (
    SplitMetrics,
    _default_normalization_path,
    _denormalize,
    _load_or_make_normalization,
    _load_prepared_manifest,
    _load_split_arrays,
    _normalize,
    _path_from_manifest,
)
from soridormi_runtime.training_dataset import DEFAULT_ACTION_SIZE, DEFAULT_OBSERVATION_SIZE, sha256_file

NEURAL_BC_SCHEMA_VERSION = 1
DEFAULT_HIDDEN_SIZES = (256, 256)
DEFAULT_ACTIVATION = "silu"


@dataclass
class NeuralBehaviorCloneTrainResult:
    ok: bool
    prepared_manifest_path: str
    output_dir: str
    checkpoint_path: str
    onnx_path: str | None
    metrics_path: str
    report_path: str
    profile_path: str | None
    normalization_path: str | None
    observation_size: int
    action_size: int
    hidden_sizes: list[int]
    activation: str
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    best_val_loss: float | None
    best_epoch: int | None
    metrics: dict[str, SplitMetrics]
    train_sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required for M6.12 neural behavior cloning.\n"
            "Build the runtime image with training dependencies, then rerun training:\n"
            "  ./scripts/build_runtime_training.sh\n"
            "This installs torch from the CUDA wheel index selected by PYTORCH_INDEX_URL "
            "and installs the project training extra."
        ) from exc
    return torch


def _parse_hidden_sizes(text: str | Iterable[int] | None) -> list[int]:
    if text is None:
        return list(DEFAULT_HIDDEN_SIZES)
    if isinstance(text, str):
        items = [item.strip() for item in text.replace("x", ",").split(",") if item.strip()]
        if not items:
            raise ValueError("hidden sizes must not be empty")
        out = [int(item) for item in items]
    else:
        out = [int(item) for item in text]
    if any(size <= 0 for size in out):
        raise ValueError("hidden sizes must be positive integers")
    return out


def _activation_module(torch: Any, name: str) -> Any:
    key = str(name).strip().lower()
    if key == "relu":
        return torch.nn.ReLU()
    if key == "gelu":
        return torch.nn.GELU()
    if key in {"silu", "swish"}:
        return torch.nn.SiLU()
    if key == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation {name!r}; use relu, gelu, silu, or tanh")


def _build_mlp(
    torch: Any,
    *,
    observation_size: int,
    action_size: int,
    hidden_sizes: list[int],
    activation: str,
    dropout: float,
) -> Any:
    layers: list[Any] = []
    in_size = observation_size
    for hidden in hidden_sizes:
        layers.append(torch.nn.Linear(in_size, hidden))
        layers.append(_activation_module(torch, activation))
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        in_size = hidden
    layers.append(torch.nn.Linear(in_size, action_size))
    return torch.nn.Sequential(*layers)


class _NormalizedPolicyModule:  # replaced below after torch import
    pass


def _make_normalized_policy_module(torch: Any, network: Any, normalization: dict[str, np.ndarray]) -> Any:
    class NormalizedPolicyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network
            self.register_buffer(
                "observation_mean",
                torch.as_tensor(normalization["observation_mean"], dtype=torch.float32).reshape(1, -1),
            )
            self.register_buffer(
                "observation_std",
                torch.as_tensor(normalization["observation_std"], dtype=torch.float32).reshape(1, -1),
            )
            self.register_buffer(
                "action_mean",
                torch.as_tensor(normalization["action_mean"], dtype=torch.float32).reshape(1, -1),
            )
            self.register_buffer(
                "action_std",
                torch.as_tensor(normalization["action_std"], dtype=torch.float32).reshape(1, -1),
            )

        def forward(self, obs: Any) -> Any:  # pragma: no cover - exercised through torch
            x = (obs - self.observation_mean) / self.observation_std
            y_norm = self.network(x)
            return y_norm * self.action_std + self.action_mean

    return NormalizedPolicyModule()


def _resolve_device(torch: Any, requested: str) -> str:
    text = str(requested or "auto").strip().lower()
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if text not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return text


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch variants
        pass


def _split_metrics_from_prediction(
    name: str,
    actions: np.ndarray,
    predictions: np.ndarray,
    *,
    path: Path,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> SplitMetrics:
    errors = list(errors or [])
    warnings = list(warnings or [])
    if errors:
        return SplitMetrics(name=name, sample_count=int(actions.shape[0]), path=str(path), errors=errors, warnings=warnings)
    if actions.shape[0] == 0:
        return SplitMetrics(name=name, sample_count=0, path=str(path), warnings=warnings + ["split has no samples"])
    diff = predictions - actions
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


def _predict(
    torch: Any,
    network: Any,
    observations: np.ndarray,
    normalization: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if observations.shape[0] == 0:
        return np.zeros((0, normalization["action_mean"].shape[0]), dtype=np.float64)
    network.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            chunk = observations[start : start + batch_size]
            x = _normalize(chunk, normalization["observation_mean"], normalization["observation_std"])
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            y_norm = network(tensor).detach().cpu().numpy().astype(np.float64)
            y = _denormalize(y_norm, normalization["action_mean"], normalization["action_std"])
            predictions.append(y)
    return np.concatenate(predictions, axis=0)


def _write_report(path: Path, result: NeuralBehaviorCloneTrainResult) -> None:
    lines = [
        "# Soridormi neural behavior-cloning policy",
        "",
        f"Prepared manifest: `{result.prepared_manifest_path}`",
        f"Checkpoint: `{result.checkpoint_path}`",
        f"ONNX: `{result.onnx_path or 'n/a'}`",
        f"Profile: `{result.profile_path or 'n/a'}`",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        "",
        "## Training",
        "",
        f"- Device: `{result.device}`",
        f"- Epochs: `{result.epochs}`",
        f"- Batch size: `{result.batch_size}`",
        f"- Hidden sizes: `{result.hidden_sizes}`",
        f"- Activation: `{result.activation}`",
        f"- Best val loss: `{result.best_val_loss}` at epoch `{result.best_epoch}`",
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


def _runtime_model_path(path: Path) -> str:
    text = str(path)
    if text.startswith("/data/"):
        return text
    if text == "data":
        return "/data"
    if text.startswith("data/"):
        return "/data/" + text[len("data/") :]
    return text


def train_neural_behavior_clone(
    prepared: str | Path,
    *,
    output_dir: str | Path | None = None,
    normalization_path: str | Path | None = None,
    hidden_sizes: str | Iterable[int] | None = None,
    activation: str = DEFAULT_ACTIVATION,
    dropout: float = 0.0,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 123,
    device: str = "auto",
    patience: int = 20,
    export_onnx: bool = True,
    create_profile: bool = True,
    profile_name: str | None = None,
    profile_template: str = "open_duck_forward",
    profile_output_dir: str | Path = "configs/policies",
    profile_force: bool = False,
    observation_size: int = DEFAULT_OBSERVATION_SIZE,
    action_size: int = DEFAULT_ACTION_SIZE,
) -> NeuralBehaviorCloneTrainResult:
    torch = _import_torch()
    hidden = _parse_hidden_sizes(hidden_sizes)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    selected_device = _resolve_device(torch, device)
    _seed_everything(torch, seed)

    manifest_path, manifest, manifest_errors = _load_prepared_manifest(prepared)
    output = Path(output_dir) if output_dir is not None else manifest_path.parent / "neural_behavior_clone"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "neural_behavior_clone.pt"
    onnx_path = output / "neural_behavior_clone.onnx"
    metrics_path = output / "train_metrics.json"
    report_path = output / "train_report.md"
    profile_path: Path | None = None
    empty_metrics = {name: SplitMetrics(name=name, sample_count=0) for name in ("train", "val", "test")}

    def finish_failed(errors: list[str], warnings: list[str] | None = None) -> NeuralBehaviorCloneTrainResult:
        result = NeuralBehaviorCloneTrainResult(
            ok=False,
            prepared_manifest_path=str(manifest_path),
            output_dir=str(output),
            checkpoint_path=str(checkpoint_path),
            onnx_path=str(onnx_path) if export_onnx else None,
            metrics_path=str(metrics_path),
            report_path=str(report_path),
            profile_path=str(profile_path) if profile_path is not None else None,
            normalization_path=str(normalization_path) if normalization_path is not None else None,
            observation_size=observation_size,
            action_size=action_size,
            hidden_sizes=hidden,
            activation=activation,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=selected_device,
            best_val_loss=None,
            best_epoch=None,
            metrics=empty_metrics,
            errors=errors,
            warnings=list(warnings or []),
        )
        metrics_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_report(report_path, result)
        return result

    if manifest_errors:
        return finish_failed(manifest_errors)

    split_payloads = manifest["splits"]
    split_paths = {
        name: _path_from_manifest(manifest_path, str(split_payloads[name]["path"]))
        for name in ("train", "val", "test")
        if isinstance(split_payloads.get(name), dict) and split_payloads[name].get("path")
    }
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str], list[str]]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for name in ("train", "val", "test"):
        path = split_paths.get(name)
        if path is None:
            arrays[name] = (
                np.zeros((0, observation_size), dtype=np.float64),
                np.zeros((0, action_size), dtype=np.float64),
                [f"Prepared manifest missing {name} split path"],
                [],
            )
        else:
            arrays[name] = _load_split_arrays(path, observation_size=observation_size, action_size=action_size)
        errors.extend(f"{name}: {error}" for error in arrays[name][2])
        warnings.extend(f"{name}: {warning}" for warning in arrays[name][3])

    train_obs, train_actions, _train_errors, _train_warnings = arrays["train"]
    if train_obs.shape[0] == 0:
        errors.append("train split has no valid samples")

    norm_path = Path(normalization_path) if normalization_path is not None else _default_normalization_path(manifest_path)
    normalization, used_norm_path, norm_errors, norm_warnings = _load_or_make_normalization(
        norm_path,
        train_obs,
        train_actions,
        observation_size=observation_size,
        action_size=action_size,
    )
    errors.extend(norm_errors)
    warnings.extend(norm_warnings)
    if errors:
        return finish_failed(errors, warnings)

    x_train = _normalize(train_obs, normalization["observation_mean"], normalization["observation_std"]).astype(np.float32)
    y_train = _normalize(train_actions, normalization["action_mean"], normalization["action_std"]).astype(np.float32)
    val_obs, val_actions, _val_errors, _val_warnings = arrays["val"]
    x_val = _normalize(val_obs, normalization["observation_mean"], normalization["observation_std"]).astype(np.float32) if val_obs.shape[0] else None
    y_val = _normalize(val_actions, normalization["action_mean"], normalization["action_std"]).astype(np.float32) if val_actions.shape[0] else None

    network = _build_mlp(
        torch,
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=hidden,
        activation=activation,
        dropout=dropout,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=selected_device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=selected_device)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=selected_device) if x_val is not None else None
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=selected_device) if y_val is not None else None

    n = int(x_tensor.shape[0])
    best_state: dict[str, Any] | None = None
    best_val_loss: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        network.train()
        permutation = torch.randperm(n, device=selected_device)
        train_loss_sum = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_tensor[idx]
            yb = y_tensor[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = network(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_n
            seen += batch_n
        train_loss = train_loss_sum / max(1, seen)

        network.eval()
        if x_val_tensor is not None and y_val_tensor is not None and int(x_val_tensor.shape[0]) > 0:
            with torch.no_grad():
                val_loss = float(loss_fn(network(x_val_tensor), y_val_tensor).detach().cpu())
        else:
            val_loss = train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if best_val_loss is None or val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        network.load_state_dict(best_state)

    metrics: dict[str, SplitMetrics] = {}
    for name in ("train", "val", "test"):
        observations, actions, split_errors, split_warnings = arrays[name]
        prediction = _predict(
            torch,
            network,
            observations,
            normalization,
            device=selected_device,
            batch_size=batch_size,
        )
        metrics[name] = _split_metrics_from_prediction(
            name,
            actions,
            prediction,
            path=split_paths.get(name, Path("")),
            errors=split_errors,
            warnings=split_warnings,
        )

    checkpoint_payload = {
        "training_run_type": "soridormi.policy_supervision.neural_behavior_clone.v1",
        "schema_version": NEURAL_BC_SCHEMA_VERSION,
        "network_state_dict": network.state_dict(),
        "hidden_sizes": hidden,
        "activation": activation,
        "dropout": dropout,
        "observation_size": observation_size,
        "action_size": action_size,
        "normalization": {key: value.astype(np.float32) for key, value in normalization.items()},
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    onnx_errors: list[str] = []
    if export_onnx:
        try:
            export_module = _make_normalized_policy_module(torch, network.cpu(), normalization)
            export_module.eval()
            dummy = torch.zeros((1, observation_size), dtype=torch.float32)
            torch.onnx.export(
                export_module,
                dummy,
                str(onnx_path),
                input_names=["obs"],
                output_names=["continuous_actions"],
                dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        except Exception as exc:
            onnx_errors.append(
                "Failed to export ONNX policy. Install ONNX export dependencies in the training environment "
                f"and retry, e.g. `uv pip install onnx`. Details: {exc!r}"
            )
    else:
        warnings.append("ONNX export skipped by request; checkpoint is not directly runnable by Soridormi ONNX runtime")

    if export_onnx and not onnx_errors and create_profile:
        try:
            profile = create_replacement_profile(
                name=profile_name or output.name,
                model_path=_runtime_model_path(onnx_path),
                template=profile_template,
                description="Neural behavior-cloning policy trained by Soridormi M6.12.",
                output_dir=profile_output_dir,
                force=profile_force,
                input_name="obs",
                output_name="continuous_actions",
                input_shape=[1, observation_size],
                output_shape=[1, action_size],
                input_type="tensor(float)",
                output_type="tensor(float)",
            )
            profile_path = profile.path
        except Exception as exc:
            onnx_errors.append(f"Failed to create runtime profile: {exc!r}")

    errors.extend(onnx_errors)
    ok = not errors
    result = NeuralBehaviorCloneTrainResult(
        ok=ok,
        prepared_manifest_path=str(manifest_path),
        output_dir=str(output),
        checkpoint_path=str(checkpoint_path),
        onnx_path=str(onnx_path) if export_onnx else None,
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        profile_path=str(profile_path) if profile_path is not None else None,
        normalization_path=str(used_norm_path) if used_norm_path is not None else None,
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=hidden,
        activation=activation,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=selected_device,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        metrics=metrics,
        train_sample_count=int(train_obs.shape[0]),
        errors=errors,
        warnings=warnings,
    )
    payload = asdict(result)
    payload["schema_version"] = NEURAL_BC_SCHEMA_VERSION
    payload["training_run_type"] = "soridormi.policy_supervision.neural_behavior_clone.v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["checkpoint_sha256"] = sha256_file(checkpoint_path)
    if export_onnx and onnx_path.exists():
        payload["onnx_sha256"] = sha256_file(onnx_path)
    payload["history"] = history
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural behavior-cloning policy and optionally export runtime ONNX/profile artifacts.")
    parser.add_argument("prepared", help="Prepared dataset directory or prepared_manifest.json")
    parser.add_argument("--output-dir", default=None, help="Training run output directory")
    parser.add_argument("--normalization", default=None, help="normalization.json path; defaults to prepared dataset normalization.json")
    parser.add_argument("--hidden-sizes", default="256,256", help="Comma-separated MLP hidden sizes, e.g. 256,256")
    parser.add_argument("--activation", default=DEFAULT_ACTIVATION, choices=["relu", "gelu", "silu", "tanh"], help="MLP activation")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience; <=0 disables early stopping")
    parser.add_argument("--skip-onnx", action="store_true", help="Train checkpoint only; do not export ONNX")
    parser.add_argument("--no-profile", action="store_true", help="Do not create configs/policies profile")
    parser.add_argument("--profile-name", default=None, help="Generated profile name; defaults to output directory name")
    parser.add_argument("--profile-template", default="open_duck_forward", help="Template profile for generated runtime profile")
    parser.add_argument("--profile-output-dir", default="configs/policies")
    parser.add_argument("--force-profile", action="store_true", help="Overwrite generated profile YAML")
    parser.add_argument("--json", action="store_true", help="Print result JSON")
    args = parser.parse_args()

    try:
        result = train_neural_behavior_clone(
            args.prepared,
            output_dir=args.output_dir,
            normalization_path=args.normalization,
            hidden_sizes=args.hidden_sizes,
            activation=args.activation,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            patience=args.patience,
            export_onnx=not args.skip_onnx,
            create_profile=not args.no_profile,
            profile_name=args.profile_name,
            profile_template=args.profile_template,
            profile_output_dir=args.profile_output_dir,
            profile_force=args.force_profile,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}")
        raise SystemExit(2)

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print("Soridormi neural behavior-clone trainer")
        print("========================================")
        print(f"Prepared: {result.prepared_manifest_path}")
        print(f"Output: {result.output_dir}")
        print(f"Checkpoint: {result.checkpoint_path}")
        print(f"ONNX: {result.onnx_path or 'n/a'}")
        print(f"Profile: {result.profile_path or 'n/a'}")
        print(f"Device: {result.device}")
        print(f"Best val loss: {result.best_val_loss}")
        for name, metrics in result.metrics.items():
            mae = "n/a" if metrics.mae is None else f"{metrics.mae:.6g}"
            print(f"{name}: samples={metrics.sample_count} mae={mae}")
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


if __name__ == "__main__":
    main()
