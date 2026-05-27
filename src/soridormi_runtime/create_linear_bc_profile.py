from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.create_policy_profile import _profile_yaml_text, _validate_profile_name
from soridormi_runtime.linear_behavior_clone_policy import load_linear_behavior_clone_model
from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_profiles import DEFAULT_PROFILE_NAME, PolicyProfile


@dataclass(frozen=True)
class CreatedLinearBehaviorCloneProfile:
    name: str
    path: Path | None
    model_path: Path
    yaml_text: str


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping, got {type(value).__name__}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_linear_model_path(training_run: str | Path | None, model: str | Path | None) -> Path:
    if model is not None:
        return Path(model)
    if training_run is None:
        raise ValueError("Either --training-run or --model is required")
    run_path = Path(training_run)
    if run_path.is_file():
        return run_path
    metrics_path = run_path / "train_metrics.json"
    if not metrics_path.exists():
        fallback = run_path / "linear_behavior_clone.npz"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Could not find train_metrics.json or linear_behavior_clone.npz in {run_path}")
    metrics = _load_json(metrics_path)
    model_path = Path(str(metrics.get("model_path") or run_path / "linear_behavior_clone.npz"))
    if model_path.is_absolute():
        return model_path
    return metrics_path.parent / model_path


def build_linear_bc_profile_payload(
    *,
    name: str,
    model_path: str | Path,
    template: str | Path | PolicyProfile = DEFAULT_PROFILE_NAME,
    description: str | None = None,
    robot_config_path: str | Path | None = None,
) -> dict[str, Any]:
    profile_name = _validate_profile_name(name)
    template_profile = template if isinstance(template, PolicyProfile) else PolicyProfile.load(template)
    contract = build_policy_contract(template_profile, robot_config_path=robot_config_path)
    if not contract.ok:
        raise ValueError(f"template profile contract is invalid: {'; '.join(contract.errors)}")

    payload = copy.deepcopy(template_profile.payload)
    payload["name"] = profile_name
    payload["description"] = description or "Linear behavior-clone baseline profile generated from an M6 training run."

    metadata = _mapping(payload.get("metadata"))
    metadata.setdefault("format_version", 1)
    metadata.setdefault("policy_family", "open_duck_mini_v2")
    metadata["derived_from_profile"] = template_profile.name
    metadata["generated_by"] = "soridormi_m65_linear_bc_profile"
    payload["metadata"] = metadata

    contract_payload = _mapping(payload.get("contract"))
    contract_payload["observation_size"] = int(contract.observation["size"])
    contract_payload["action_size"] = int(contract.action["size"])
    contract_payload["joint_names"] = list(contract.action["joint_order"])
    payload["contract"] = contract_payload

    model = _mapping(payload.get("model"))
    model["kind"] = "linear_behavior_clone"
    model["path"] = str(model_path)
    model["input_name"] = "obs"
    model["output_name"] = "continuous_actions"
    model["input_shape"] = [1, int(contract.observation["size"])]
    model["output_shape"] = [1, int(contract.action["size"])]
    model["input_type"] = "tensor(float)"
    model["output_type"] = "tensor(float)"
    payload["model"] = model

    logging = _mapping(payload.get("logging"))
    logging["prefix"] = f"policy_{profile_name}"
    payload["logging"] = logging
    return payload


def create_linear_bc_profile(
    *,
    name: str,
    training_run: str | Path | None = None,
    model: str | Path | None = None,
    template: str | Path | PolicyProfile = DEFAULT_PROFILE_NAME,
    description: str | None = None,
    output_dir: str | Path = "configs/policies",
    output_path: str | Path | None = None,
    robot_config_path: str | Path | None = None,
    force: bool = False,
    stdout: bool = False,
    check_model: bool = True,
) -> CreatedLinearBehaviorCloneProfile:
    profile_name = _validate_profile_name(name)
    model_path = resolve_linear_model_path(training_run, model)
    if check_model:
        loaded = load_linear_behavior_clone_model(model_path)
        if not loaded.ok:
            raise ValueError("linear behavior-clone model is invalid: " + "; ".join(loaded.errors))
    payload = build_linear_bc_profile_payload(
        name=profile_name,
        model_path=model_path,
        template=template,
        description=description,
        robot_config_path=robot_config_path,
    )
    yaml_text = _profile_yaml_text(payload)
    if stdout:
        return CreatedLinearBehaviorCloneProfile(profile_name, None, model_path, yaml_text)

    path = Path(output_path) if output_path is not None else Path(output_dir) / f"{profile_name}.yaml"
    if path.exists() and not force:
        raise FileExistsError(f"Policy profile already exists: {path}. Pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")
    return CreatedLinearBehaviorCloneProfile(profile_name, path, model_path, yaml_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runtime policy profile for an M6 linear behavior-clone baseline.")
    parser.add_argument("name", help="New profile name, used for configs/policies/NAME.yaml by default")
    parser.add_argument("--training-run", default=None, help="Training run directory containing train_metrics.json / linear_behavior_clone.npz")
    parser.add_argument("--model", default=None, help="Explicit linear_behavior_clone.npz path")
    parser.add_argument("--template", default=DEFAULT_PROFILE_NAME, help="Template profile name/path to clone")
    parser.add_argument("--description", default=None, help="Profile description")
    parser.add_argument("--output-dir", default="configs/policies", help="Directory for generated YAML")
    parser.add_argument("--output", default=None, help="Explicit output YAML path")
    parser.add_argument("--robot-config", default=None, help="Robot YAML used to stamp contract metadata")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output profile")
    parser.add_argument("--stdout", action="store_true", help="Print YAML instead of writing a file")
    parser.add_argument("--no-check-model", action="store_true", help="Do not load/validate the NPZ model before writing")
    args = parser.parse_args()

    result = create_linear_bc_profile(
        name=args.name,
        training_run=args.training_run,
        model=args.model,
        template=args.template,
        description=args.description,
        output_dir=args.output_dir,
        output_path=args.output,
        robot_config_path=args.robot_config,
        force=args.force,
        stdout=args.stdout,
        check_model=not args.no_check_model,
    )
    if args.stdout:
        print(result.yaml_text, end="")
    else:
        assert result.path is not None
        print(f"Created linear behavior-clone profile: {result.path}")
        print(f"Model: {result.model_path}")
        print("Next checks:")
        print(f"  ./scripts/export_policy_contract.sh {result.path}")
        print(f"  ./scripts/check_policy_model.sh --profile {result.path}")
        print(f"  ./scripts/run_policy_experiment.sh {result.name}")


if __name__ == "__main__":
    main()
