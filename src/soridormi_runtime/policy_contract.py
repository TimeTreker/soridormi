from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from soridormi_runtime.action_mapper import PolicyActionMapper
from soridormi_runtime.observation_builder import ObservationBuilder, resolve_robot_config_path
from soridormi_runtime.policy_profiles import PolicyProfile


@dataclass(frozen=True)
class ObservationSegmentSpec:
    name: str
    size: int
    start: int
    end: int
    description: str = ""


@dataclass(frozen=True)
class PolicyContractResult:
    ok: bool
    profile_name: str
    profile_path: str
    robot_config_path: str
    model: dict[str, Any]
    observation: dict[str, Any]
    action: dict[str, Any]
    command: dict[str, Any]
    phase: dict[str, Any]
    metadata: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_OBSERVATION_SEGMENT_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("gyro_xyz", 3, "IMU angular velocity xyz"),
    ("accelerometer_xyz", 3, "IMU linear acceleration xyz after configured policy bias"),
    ("command", 7, "Policy command vector: x, y, yaw, neck_pitch, head_pitch, head_yaw, head_roll"),
    ("joint_offsets", 14, "Joint positions minus policy default actuator pose in action joint order"),
    ("joint_velocities_scaled", 14, "Joint velocities multiplied by the policy dof_vel_scale"),
    ("last_action", 14, "Previous policy action"),
    ("last_last_action", 14, "Policy action from two inference steps ago"),
    ("last_last_last_action", 14, "Policy action from three inference steps ago"),
    ("motor_targets", 14, "Previous speed-limited motor targets in action joint order"),
    ("feet_contacts", 2, "Left/right foot contact bits or probabilities"),
    ("imitation_phase", 2, "Policy imitation phase sine/cosine-like reference"),
)


def observation_segments() -> list[ObservationSegmentSpec]:
    """Return the canonical Open Duck Mini v2 101D observation layout."""
    out: list[ObservationSegmentSpec] = []
    cursor = 0
    for name, size, description in _OBSERVATION_SEGMENT_LAYOUT:
        out.append(
            ObservationSegmentSpec(
                name=name,
                size=size,
                start=cursor,
                end=cursor + size,
                description=description,
            )
        )
        cursor += size
    return out


def _shape_last_dim(shape: list[Any] | tuple[Any, ...] | None) -> int | None:
    if not shape:
        return None
    value = shape[-1]
    if value in {None, "", "?", -1}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_strings(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _profile_action_value(profile: PolicyProfile, key: str, fallback: Any) -> Any:
    action_mapping = _mapping(profile.payload.get("action_mapping"))
    return action_mapping.get(key, fallback)


def _profile_contract_value(profile: PolicyProfile, key: str, fallback: Any) -> Any:
    contract = _mapping(profile.payload.get("contract"))
    return contract.get(key, fallback)


def build_policy_contract(
    profile: str | os.PathLike[str] | PolicyProfile | None = None,
    *,
    robot_config_path: str | os.PathLike[str] | None = None,
) -> PolicyContractResult:
    """Build and validate the model-replacement contract for one policy profile.

    The contract is intentionally static: it does not load the ONNX model or start
    MuJoCo. Use check_policy_model for ONNX file validation, and use this contract
    export to verify the profile/robot observation-action interface before a new
    model is plugged into the runtime.
    """
    policy_profile = profile if isinstance(profile, PolicyProfile) else PolicyProfile.load(profile)
    robot_path = resolve_robot_config_path(robot_config_path)
    builder = ObservationBuilder.from_robot_config(robot_path)
    mapper = PolicyActionMapper.from_robot_config(robot_path)

    joint_names = list(builder.config.joint_names)
    action_joint_names = list(mapper.config.joint_names)
    segments = observation_segments()
    obs_size = int(sum(item.size for item in segments))
    action_size = int(mapper.ACTION_SIZE)

    errors: list[str] = []
    warnings: list[str] = []

    if obs_size != ObservationBuilder.OBS_SIZE:
        errors.append(f"Observation segment sizes sum to {obs_size}, expected {ObservationBuilder.OBS_SIZE}")
    if action_size != ObservationBuilder.ACTION_SIZE:
        errors.append(f"Action size {action_size} does not match observation action-history size {ObservationBuilder.ACTION_SIZE}")
    if joint_names != action_joint_names:
        errors.append("Observation joint order and action-mapper joint order differ")

    model = policy_profile.model
    model_input_size = _shape_last_dim(model.input_shape)
    model_output_size = _shape_last_dim(model.output_shape)
    expected_obs_size = int(_profile_contract_value(policy_profile, "observation_size", obs_size))
    expected_action_size = int(_profile_contract_value(policy_profile, "action_size", action_size))
    declared_joint_names = _list_of_strings(_profile_contract_value(policy_profile, "joint_names", None))

    if expected_obs_size != obs_size:
        errors.append(f"Profile contract.observation_size={expected_obs_size}, runtime observation size={obs_size}")
    if expected_action_size != action_size:
        errors.append(f"Profile contract.action_size={expected_action_size}, runtime action size={action_size}")
    if model_input_size is not None and model_input_size != obs_size:
        errors.append(f"Model input last dimension is {model_input_size}, runtime observation size is {obs_size}")
    if model_output_size is not None and model_output_size != action_size:
        errors.append(f"Model output last dimension is {model_output_size}, runtime action size is {action_size}")
    if declared_joint_names is not None and declared_joint_names != joint_names:
        errors.append("Profile contract.joint_names does not match robot actuator/order contract")
    if model.input_type != "tensor(float)":
        warnings.append(f"Model input type is {model.input_type!r}; runtime currently builds float32 observations")
    if model.output_type != "tensor(float)":
        warnings.append(f"Model output type is {model.output_type!r}; runtime currently expects float32-like actions")

    action_scale = float(_profile_action_value(policy_profile, "action_scale", mapper.config.action_scale))
    max_motor_velocity = float(_profile_action_value(policy_profile, "max_motor_velocity", mapper.config.max_motor_velocity))
    speed_limit_enabled = bool(_profile_action_value(policy_profile, "speed_limit_enabled", mapper.config.speed_limit_enabled))

    observation_payload: dict[str, Any] = {
        "size": obs_size,
        "dtype": "float32",
        "segments": [asdict(item) for item in segments],
        "joint_order": joint_names,
        "dof_vel_scale": float(builder.config.dof_vel_scale),
        "accelerometer_bias_xyz": [float(x) for x in builder.config.accelerometer_bias_xyz],
        "use_state_feet_contacts": bool(builder.config.use_state_feet_contacts),
        "default_positions_by_name": {
            str(name): float(builder.config.default_positions_by_name.get(name, 0.0))
            for name in joint_names
        },
    }
    action_payload: dict[str, Any] = {
        "size": action_size,
        "dtype": "float32",
        "joint_order": action_joint_names,
        "mapping": "target = default_position + action_scale * action, then optional speed/limit clipping",
        "action_scale": action_scale,
        "max_motor_velocity": max_motor_velocity,
        "speed_limit_enabled": speed_limit_enabled,
        "clip_to_limits": bool(mapper.config.clip_to_limits),
        "kp_default": float(mapper.config.kp_default),
        "kd_default": float(mapper.config.kd_default),
        "torque_default": float(mapper.config.torque_default),
        "limits_by_name": {
            str(name): [float(limit[0]), float(limit[1])]
            for name, limit in sorted(mapper.config.limits_by_name.items())
        },
    }
    command_payload = {
        "size": 7,
        "fields": ["x", "y", "yaw", "neck_pitch", "head_pitch", "head_yaw", "head_roll"],
    }
    phase_payload = {
        "size": 2,
        "mode": str(_mapping(policy_profile.payload.get("phase")).get("mode", "step")),
        "period_steps": _mapping(policy_profile.payload.get("phase")).get("period_steps", "auto"),
        "reference_data": _mapping(policy_profile.payload.get("phase")).get("reference_data"),
    }
    model_payload = {
        "path": model.path,
        "input_name": model.input_name,
        "output_name": model.output_name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "input_type": model.input_type,
        "output_type": model.output_type,
    }
    metadata_payload = {
        "profile_description": policy_profile.description,
        "profile_metadata": _mapping(policy_profile.payload.get("metadata")),
        "profile_contract": _mapping(policy_profile.payload.get("contract")),
        "contract_version": 1,
    }

    return PolicyContractResult(
        ok=not errors,
        profile_name=policy_profile.name,
        profile_path=str(policy_profile.path),
        robot_config_path=str(robot_path),
        model=model_payload,
        observation=observation_payload,
        action=action_payload,
        command=command_payload,
        phase=phase_payload,
        metadata=metadata_payload,
        errors=errors,
        warnings=warnings,
    )


def print_contract_summary(result: PolicyContractResult) -> None:
    print("Soridormi policy replacement contract")
    print("====================================")
    print(f"Profile: {result.profile_name}")
    print(f"Profile file: {result.profile_path}")
    print(f"Robot config: {result.robot_config_path}")
    print(f"Model: {result.model['path']}")
    print(
        "Model IO: "
        f"{result.model['input_name']} {result.model['input_shape']} {result.model['input_type']} -> "
        f"{result.model['output_name']} {result.model['output_shape']} {result.model['output_type']}"
    )
    print(f"Observation: size={result.observation['size']} dtype={result.observation['dtype']}")
    print(f"Action: size={result.action['size']} dtype={result.action['dtype']}")
    print("Observation segments:")
    for item in result.observation["segments"]:
        print(f"  {item['start']:3d}:{item['end']:<3d} {item['name']}[{item['size']}]")
    print("Action joint order:")
    print("  " + ", ".join(result.action["joint_order"]))
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    print("Result:", "OK" if result.ok else "FAILED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate a Soridormi policy replacement contract.")
    parser.add_argument("profile", nargs="?", help="Policy profile name or YAML path")
    parser.add_argument("--robot-config", default=None, help="Robot YAML path used for joint/action contract")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--validate-only", action="store_true", help="Only print OK/FAILED and diagnostics")
    args = parser.parse_args()

    result = build_policy_contract(args.profile, robot_config_path=args.robot_config)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.validate_only:
        if result.warnings:
            for warning in result.warnings:
                print(f"WARNING: {warning}")
        if result.errors:
            for error in result.errors:
                print(f"ERROR: {error}")
        print("OK" if result.ok else "FAILED")
    else:
        print_contract_summary(result)

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
