from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from soridormi_runtime.policy_contract import build_policy_contract
from soridormi_runtime.policy_profiles import DEFAULT_PROFILE_NAME, PolicyProfile

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class CreatedPolicyProfile:
    name: str
    path: Path | None
    yaml_text: str


def _validate_profile_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("profile name must not be empty")
    if not _PROFILE_NAME_RE.fullmatch(text):
        raise ValueError(
            "profile name must contain only letters, numbers, underscores, and hyphens, "
            "and must start with a letter or number"
        )
    return text


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping, got {type(value).__name__}")
    return value


def _parse_shape(value: str | list[Any] | tuple[Any, ...] | None, *, name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    loaded = yaml.safe_load(str(value))
    if isinstance(loaded, int):
        return [loaded]
    if isinstance(loaded, str):
        return [item.strip() for item in loaded.split(",") if item.strip()]
    if not isinstance(loaded, list):
        raise ValueError(f"{name} must be a YAML list or comma-separated shape, got {value!r}")
    return list(loaded)


def _profile_yaml_text(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def build_replacement_profile_payload(
    *,
    name: str,
    model_path: str,
    template: str | Path | PolicyProfile = DEFAULT_PROFILE_NAME,
    description: str | None = None,
    robot_config_path: str | Path | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
    input_shape: list[Any] | tuple[Any, ...] | str | None = None,
    output_shape: list[Any] | tuple[Any, ...] | str | None = None,
    input_type: str | None = None,
    output_type: str | None = None,
) -> dict[str, Any]:
    """Clone a known-good policy profile and stamp the runtime contract into it.

    The returned YAML payload is intentionally static. It does not load the ONNX
    file; use check_policy_model.sh after writing the profile to validate the
    actual model metadata and provider selection.
    """
    profile_name = _validate_profile_name(name)
    template_profile = template if isinstance(template, PolicyProfile) else PolicyProfile.load(template)
    contract = build_policy_contract(template_profile, robot_config_path=robot_config_path)
    if not contract.ok:
        joined = "; ".join(contract.errors)
        raise ValueError(f"template profile contract is invalid: {joined}")

    payload = copy.deepcopy(template_profile.payload)
    payload["name"] = profile_name
    payload["description"] = description or f"Replacement policy profile cloned from {template_profile.name}."

    metadata = _mapping(payload.get("metadata"))
    metadata.setdefault("format_version", 1)
    metadata.setdefault("policy_family", "open_duck_mini_v2")
    metadata["derived_from_profile"] = template_profile.name
    metadata["generated_by"] = "soridormi_m5_profile_scaffold"
    payload["metadata"] = metadata

    contract_payload = _mapping(payload.get("contract"))
    contract_payload["observation_size"] = int(contract.observation["size"])
    contract_payload["action_size"] = int(contract.action["size"])
    contract_payload["joint_names"] = list(contract.action["joint_order"])
    payload["contract"] = contract_payload

    model = _mapping(payload.get("model"))
    model["path"] = str(model_path)
    if input_name is not None:
        model["input_name"] = str(input_name)
    if output_name is not None:
        model["output_name"] = str(output_name)
    parsed_input_shape = _parse_shape(input_shape, name="input_shape")
    parsed_output_shape = _parse_shape(output_shape, name="output_shape")
    if parsed_input_shape is not None:
        model["input_shape"] = parsed_input_shape
    if parsed_output_shape is not None:
        model["output_shape"] = parsed_output_shape
    if input_type is not None:
        model["input_type"] = str(input_type)
    if output_type is not None:
        model["output_type"] = str(output_type)
    payload["model"] = model

    logging = _mapping(payload.get("logging"))
    logging["prefix"] = f"policy_{profile_name}"
    payload["logging"] = logging

    return payload


def create_replacement_profile(
    *,
    name: str,
    model_path: str,
    template: str | Path | PolicyProfile = DEFAULT_PROFILE_NAME,
    description: str | None = None,
    output_dir: str | Path = "configs/policies",
    output_path: str | Path | None = None,
    robot_config_path: str | Path | None = None,
    force: bool = False,
    stdout: bool = False,
    input_name: str | None = None,
    output_name: str | None = None,
    input_shape: list[Any] | tuple[Any, ...] | str | None = None,
    output_shape: list[Any] | tuple[Any, ...] | str | None = None,
    input_type: str | None = None,
    output_type: str | None = None,
) -> CreatedPolicyProfile:
    profile_name = _validate_profile_name(name)
    payload = build_replacement_profile_payload(
        name=profile_name,
        model_path=model_path,
        template=template,
        description=description,
        robot_config_path=robot_config_path,
        input_name=input_name,
        output_name=output_name,
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=input_type,
        output_type=output_type,
    )
    yaml_text = _profile_yaml_text(payload)
    if stdout:
        return CreatedPolicyProfile(profile_name, None, yaml_text)

    path = Path(output_path) if output_path is not None else Path(output_dir) / f"{profile_name}.yaml"
    if path.exists() and not force:
        raise FileExistsError(f"Policy profile already exists: {path}. Pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")
    return CreatedPolicyProfile(profile_name, path, yaml_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Soridormi replacement policy profile from a known-good template.")
    parser.add_argument("name", help="New profile name, used for configs/policies/NAME.yaml by default")
    parser.add_argument("--model", required=True, help="Path to the replacement ONNX model")
    parser.add_argument("--template", default=DEFAULT_PROFILE_NAME, help="Template profile name/path to clone")
    parser.add_argument("--description", default=None, help="Profile description")
    parser.add_argument("--output-dir", default="configs/policies", help="Directory for the generated YAML")
    parser.add_argument("--output", default=None, help="Explicit output YAML path")
    parser.add_argument("--robot-config", default=None, help="Robot YAML used to stamp contract metadata")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output profile")
    parser.add_argument("--stdout", action="store_true", help="Print YAML instead of writing a file")
    parser.add_argument("--input-name", default=None, help="Override model.input_name")
    parser.add_argument("--output-name", default=None, help="Override model.output_name")
    parser.add_argument("--input-shape", default=None, help="Override model.input_shape, e.g. '[1, 101]' or '1,101'")
    parser.add_argument("--output-shape", default=None, help="Override model.output_shape, e.g. '[1, 14]' or '1,14'")
    parser.add_argument("--input-type", default=None, help="Override model.input_type")
    parser.add_argument("--output-type", default=None, help="Override model.output_type")
    args = parser.parse_args()

    result = create_replacement_profile(
        name=args.name,
        model_path=args.model,
        template=args.template,
        description=args.description,
        output_dir=args.output_dir,
        output_path=args.output,
        robot_config_path=args.robot_config,
        force=args.force,
        stdout=args.stdout,
        input_name=args.input_name,
        output_name=args.output_name,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
        input_type=args.input_type,
        output_type=args.output_type,
    )
    if args.stdout:
        print(result.yaml_text, end="")
    else:
        assert result.path is not None
        print(f"Created policy profile: {result.path}")
        print("Next checks:")
        print(f"  ./scripts/export_policy_contract.sh {result.path}")
        print(f"  ./scripts/check_policy_model.sh --profile {result.path}")


if __name__ == "__main__":
    main()
