from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE_NAME = "open_duck_forward"
DEFAULT_PROFILE_DIRS = (Path("/app/configs/policies"), Path("configs/policies"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    if value is None:
        return ""
    return str(value)


def _shape_to_env(shape: list[Any] | tuple[Any, ...] | None) -> str:
    if shape is None:
        return ""
    return ",".join(str(item) for item in shape)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping, got {type(value).__name__}")
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Policy profile must be a YAML mapping: {path}")
    return payload


def resolve_policy_profile_path(profile: str | os.PathLike[str] | None = None) -> Path:
    explicit_file = os.environ.get("SORIDORMI_POLICY_PROFILE_FILE")
    if explicit_file and (profile is None or str(profile).strip() == ""):
        return Path(explicit_file)

    text = str(profile or os.environ.get("SORIDORMI_POLICY_PROFILE") or DEFAULT_PROFILE_NAME).strip()
    if not text:
        text = DEFAULT_PROFILE_NAME

    candidate = Path(text)
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        return candidate

    name = text[:-5] if text.endswith(".yaml") else text
    for directory in DEFAULT_PROFILE_DIRS:
        path = directory / f"{name}.yaml"
        if path.exists():
            return path

    searched = ", ".join(str(d) for d in DEFAULT_PROFILE_DIRS)
    raise FileNotFoundError(f"Policy profile {text!r} not found. Searched: {searched}")


@dataclass(frozen=True)
class PolicyModelSpec:
    path: str
    kind: str = "onnx"
    input_name: str = "obs"
    output_name: str = "continuous_actions"
    input_shape: list[Any] = field(default_factory=lambda: [1, 101])
    output_shape: list[Any] = field(default_factory=lambda: [1, 14])
    input_type: str = "tensor(float)"
    output_type: str = "tensor(float)"


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    description: str
    path: Path
    payload: dict[str, Any]
    model: PolicyModelSpec

    @classmethod
    def load(cls, profile: str | os.PathLike[str] | None = None) -> "PolicyProfile":
        path = resolve_policy_profile_path(profile)
        payload = _load_yaml_mapping(path)
        model_payload = _mapping(payload.get("model"))
        model_path = model_payload.get("path")
        if not model_path:
            raise ValueError("policy profile model.path is required")
        model = PolicyModelSpec(
            path=str(model_path),
            kind=str(model_payload.get("kind", "onnx")),
            input_name=str(model_payload.get("input_name", "obs")),
            output_name=str(model_payload.get("output_name", "continuous_actions")),
            input_shape=list(model_payload.get("input_shape", [1, 101])),
            output_shape=list(model_payload.get("output_shape", [1, 14])),
            input_type=str(model_payload.get("input_type", "tensor(float)")),
            output_type=str(model_payload.get("output_type", "tensor(float)")),
        )
        return cls(
            name=str(payload.get("name") or path.stem),
            description=str(payload.get("description") or ""),
            path=path,
            payload=payload,
            model=model,
        )

    def env(self) -> dict[str, str]:
        runtime = _mapping(self.payload.get("runtime"))
        command = _mapping(self.payload.get("command"))
        phase = _mapping(self.payload.get("phase"))
        action = _mapping(self.payload.get("action_mapping"))
        observation = _mapping(self.payload.get("observation"))
        simulator = _mapping(self.payload.get("simulator"))
        postprocess = _mapping(self.payload.get("action_postprocess"))
        residual = _mapping(self.payload.get("residual_policy"))
        logging = _mapping(self.payload.get("logging"))
        accel_bias = observation.get("accel_bias_xyz", [1.3, 0.0, 0.0])
        if not isinstance(accel_bias, (list, tuple)) or len(accel_bias) != 3:
            raise ValueError("observation.accel_bias_xyz must be a 3-item list")

        return {
            "SORIDORMI_POLICY_PROFILE": self.name,
            "SORIDORMI_POLICY_PROFILE_FILE": str(self.path),
            "SORIDORMI_POLICY_PATH": self.model.path,
            "SORIDORMI_POLICY_BACKEND": self.model.kind,
            "SORIDORMI_POLICY_KIND": self.model.kind,
            "SORIDORMI_POLICY_INPUT_NAME": self.model.input_name,
            "SORIDORMI_POLICY_OUTPUT_NAME": self.model.output_name,
            "SORIDORMI_POLICY_EXPECTED_INPUT_SHAPE": _shape_to_env(self.model.input_shape),
            "SORIDORMI_POLICY_EXPECTED_OUTPUT_SHAPE": _shape_to_env(self.model.output_shape),
            "SORIDORMI_POLICY_EXPECTED_INPUT_TYPE": self.model.input_type,
            "SORIDORMI_POLICY_EXPECTED_OUTPUT_TYPE": self.model.output_type,
            "SORIDORMI_RUNTIME_MODE": str(runtime.get("mode", "onnx_policy")),
            "SORIDORMI_BACKEND": str(runtime.get("backend", "sim")),
            "CONTROL_HZ": _fmt(runtime.get("control_hz", 50)),
            "SORIDORMI_RESET_AT_START": _fmt(_as_bool(runtime.get("reset_at_start", False), False)),
            "SORIDORMI_SIM_SYNC_STEP": _fmt(_as_bool(runtime.get("sync_step", False), False)),
            "SORIDORMI_SIM_PREROLL_STEPS": _fmt(_as_int(runtime.get("sim_preroll_steps", runtime.get("preroll_steps", 0)), 0)),
            "SORIDORMI_COMMAND_X": _fmt(command.get("x", 0.0)),
            "SORIDORMI_COMMAND_Y": _fmt(command.get("y", 0.0)),
            "SORIDORMI_COMMAND_YAW": _fmt(command.get("yaw", 0.0)),
            "SORIDORMI_NECK_PITCH": _fmt(command.get("neck_pitch", 0.0)),
            "SORIDORMI_HEAD_PITCH": _fmt(command.get("head_pitch", 0.0)),
            "SORIDORMI_HEAD_YAW": _fmt(command.get("head_yaw", 0.0)),
            "SORIDORMI_HEAD_ROLL": _fmt(command.get("head_roll", 0.0)),
            "SORIDORMI_COMMAND_RAMP_SECONDS": _fmt(command.get("ramp_seconds", 0.0)),
            "SORIDORMI_PHASE_MODE": str(phase.get("mode", "step")),
            "SORIDORMI_PHASE_FREQUENCY": _fmt(phase.get("frequency", 1.0)),
            "SORIDORMI_PHASE_PERIOD_STEPS": _fmt(phase.get("period_steps", "auto")),
            "SORIDORMI_PHASE_REFERENCE_DATA": str(phase.get(
                "reference_data",
                "/workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
            )),
            "SORIDORMI_PHASE_REQUIRE_REFERENCE_DATA": _fmt(_as_bool(phase.get("require_reference_data", False), False)),
            "SORIDORMI_PHASE_STEP_INCREMENT": _fmt(phase.get("step_increment", 1.0)),
            "SORIDORMI_PHASE_ENABLED": _fmt(_as_bool(phase.get("enabled", True), True)),
            "SORIDORMI_PHASE_OFFSET": _fmt(phase.get("offset", 0.0)),
            "SORIDORMI_ACTION_SCALE": _fmt(action.get("action_scale", 0.25)),
            "SORIDORMI_MAX_MOTOR_VELOCITY": _fmt(action.get("max_motor_velocity", 5.24)),
            "SORIDORMI_ACTION_POSTPROCESS": _fmt(_as_bool(postprocess.get("enabled", False), False)),
            "SORIDORMI_ACTION_POSTPROCESS_MODE": str(postprocess.get("mode", "locomotion_boost")),
            "SORIDORMI_LEG_ACTION_GAIN": _fmt(postprocess.get("leg_gain", 1.0)),
            "SORIDORMI_HEAD_ACTION_GAIN": _fmt(postprocess.get("head_gain", 1.0)),
            "SORIDORMI_HIP_YAW_ACTION_GAIN": _fmt(postprocess.get("hip_yaw_gain", 1.0)),
            "SORIDORMI_HIP_ROLL_ACTION_GAIN": _fmt(postprocess.get("hip_roll_gain", 1.0)),
            "SORIDORMI_HIP_PITCH_ACTION_GAIN": _fmt(postprocess.get("hip_pitch_gain", 1.0)),
            "SORIDORMI_KNEE_ACTION_GAIN": _fmt(postprocess.get("knee_gain", 1.0)),
            "SORIDORMI_ANKLE_ACTION_GAIN": _fmt(postprocess.get("ankle_gain", 1.0)),
            "SORIDORMI_ACTION_CLIP_ABS": _fmt(postprocess.get("clip_abs", 0.0)),
            "SORIDORMI_RESIDUAL_TEACHER_PROFILE": str(residual.get("teacher_profile", "open_duck_forward")),
            "SORIDORMI_RESIDUAL_SCALE": _fmt(residual.get("residual_scale", 0.05)),
            "SORIDORMI_RESIDUAL_CLIP_ABS": _fmt(residual.get("residual_clip_abs", 1.0)),
            "SORIDORMI_RESIDUAL_FINAL_ACTION_CLIP_ABS": _fmt(residual.get("final_action_clip_abs", 0.0)),
            "SORIDORMI_POLICY_ACCEL_BIAS_X": _fmt(accel_bias[0]),
            "SORIDORMI_POLICY_ACCEL_BIAS_Y": _fmt(accel_bias[1]),
            "SORIDORMI_POLICY_ACCEL_BIAS_Z": _fmt(accel_bias[2]),
            "SORIDORMI_USE_STATE_FEET_CONTACTS": _fmt(_as_bool(observation.get("use_state_feet_contacts", True), True)),
            "SORIDORMI_BOOTSTRAP_POLICY_DEFAULTS_FROM_STATE": _fmt(_as_bool(observation.get("bootstrap_policy_defaults_from_state", True), True)),
            "SORIDORMI_MUJOCO_USE_HOME_KEYFRAME": _fmt(_as_bool(simulator.get("use_home_keyframe", True), True)),
            "SORIDORMI_MUJOCO_HOME_KEYFRAME_OVERRIDES_RESET_POSE": _fmt(_as_bool(simulator.get("home_keyframe_overrides_reset_pose", True), True)),
            "SORIDORMI_MUJOCO_OFFICIAL_RESET_SEQUENCE": _fmt(_as_bool(simulator.get("official_reset_sequence", True), True)),
            "SORIDORMI_MUJOCO_OFFICIAL_SENSOR_MODE": _fmt(_as_bool(simulator.get("official_sensor_mode", True), True)),
            "SORIDORMI_MUJOCO_OFFICIAL_CONTACT_MODE": _fmt(_as_bool(simulator.get("official_contact_mode", True), True)),
            "SORIDORMI_AUTO_RESET": _fmt(_as_bool(simulator.get("auto_reset", True), True)),
            "SORIDORMI_MUJOCO_VIEWER": _fmt(_as_bool(simulator.get("viewer", True), True)),
            "SORIDORMI_SIM_BACKEND": str(simulator.get("backend", "mujoco")),
            "SORIDORMI_RUNTIME_LOG": _fmt(_as_bool(logging.get("enabled", True), True)),
            "SORIDORMI_RUNTIME_LOG_FORMAT": str(logging.get("format", "mcap")),
            "SORIDORMI_RUNTIME_LOG_EVERY_N": _fmt(_as_int(logging.get("every_n", 1), 1)),
            "SORIDORMI_RUNTIME_LOG_PREFIX": str(logging.get("prefix", f"policy_{self.name}")),
        }

    def shell_exports(self) -> str:
        return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(self.env().items()))

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "description": self.description, "path": str(self.path), "model": self.model.__dict__, "env": self.env()}, indent=2, sort_keys=True)


def list_policy_profiles() -> list[Path]:
    paths: list[Path] = []
    for directory in DEFAULT_PROFILE_DIRS:
        if directory.exists():
            paths.extend(sorted(directory.glob("*.yaml")))
            paths.extend(sorted(directory.glob("*.yml")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve Soridormi ONNX policy profile settings.")
    parser.add_argument("profile", nargs="?", help="Profile name or YAML path")
    parser.add_argument("--shell", action="store_true", help="Print shell exports")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    parser.add_argument("--list", action="store_true", help="List available profile files")
    args = parser.parse_args()

    if args.list:
        for path in list_policy_profiles():
            try:
                profile = PolicyProfile.load(path)
                print(f"{profile.name}\t{path}\t{profile.description}")
            except Exception:
                print(path)
        return

    profile = PolicyProfile.load(args.profile)
    if args.shell:
        print(profile.shell_exports())
    elif args.json:
        print(profile.to_json())
    else:
        print(f"{profile.name}: {profile.description}")
        print(f"  file: {profile.path}")
        print(f"  model: {profile.model.path}")
        print(f"  model kind: {profile.model.kind}")
        print(f"  input: {profile.model.input_name} {profile.model.input_shape} {profile.model.input_type}")
        print(f"  output: {profile.model.output_name} {profile.model.output_shape} {profile.model.output_type}")


if __name__ == "__main__":
    main()
