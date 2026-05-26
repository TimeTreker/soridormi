from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PolicyTuningProfile:
    """A repeatable ONNX-policy runtime tuning profile.

    Profiles are intentionally small and conservative. They are not meant to
    solve walking by themselves; they make log collection repeatable so M3.7+
    analysis can compare like-for-like experiments.
    """

    name: str
    description: str
    command_x: float = 0.0
    command_y: float = 0.0
    command_yaw: float = 0.0
    phase_frequency: float = 1.0
    action_scale: float = 0.10
    max_motor_velocity: float = 3.0
    runtime_log_every_n: int = 1

    def env(self) -> dict[str, str]:
        return {
            "SORIDORMI_POLICY_PROFILE": self.name,
            "SORIDORMI_RUNTIME_MODE": "onnx_policy",
            "SORIDORMI_RUNTIME_LOG": "1",
            "SORIDORMI_RUNTIME_LOG_FORMAT": "mcap",
            "SORIDORMI_RUNTIME_LOG_EVERY_N": str(int(self.runtime_log_every_n)),
            "SORIDORMI_RUNTIME_LOG_PREFIX": f"runtime_{self.name}",
            "SORIDORMI_COMMAND_X": _fmt(self.command_x),
            "SORIDORMI_COMMAND_Y": _fmt(self.command_y),
            "SORIDORMI_COMMAND_YAW": _fmt(self.command_yaw),
            "SORIDORMI_PHASE_FREQUENCY": _fmt(self.phase_frequency),
            "SORIDORMI_ACTION_SCALE": _fmt(self.action_scale),
            "SORIDORMI_MAX_MOTOR_VELOCITY": _fmt(self.max_motor_velocity),
        }

    def shell_exports(self) -> str:
        return "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in sorted(self.env().items())
        )


BUILTIN_PROFILES: tuple[PolicyTuningProfile, ...] = (
    PolicyTuningProfile(
        name="idle_debug",
        description="Zero command and zero phase; verifies static policy/logging baseline.",
        command_x=0.0,
        phase_frequency=0.0,
        action_scale=0.05,
        max_motor_velocity=2.0,
    ),
    PolicyTuningProfile(
        name="crawl_very_safe",
        description="Smallest forward command with very small action scale.",
        command_x=0.005,
        phase_frequency=0.8,
        action_scale=0.05,
        max_motor_velocity=2.0,
    ),
    PolicyTuningProfile(
        name="crawl_safe",
        description="Recommended first dynamic walking test after M3.7 logging.",
        command_x=0.01,
        phase_frequency=1.0,
        action_scale=0.10,
        max_motor_velocity=3.0,
    ),
    PolicyTuningProfile(
        name="walk_cautious",
        description="Slightly stronger forward command/action for comparison.",
        command_x=0.02,
        phase_frequency=1.0,
        action_scale=0.12,
        max_motor_velocity=4.0,
    ),
    PolicyTuningProfile(
        name="walk_default_soft",
        description="Near Open-Duck-like timing, but softer than the raw default action scale.",
        command_x=0.03,
        phase_frequency=1.0,
        action_scale=0.20,
        max_motor_velocity=5.24,
    ),
    PolicyTuningProfile(
        name="turn_cautious",
        description="Low yaw command; useful for checking yaw sign/order after forward tests.",
        command_x=0.0,
        command_yaw=0.05,
        phase_frequency=1.0,
        action_scale=0.10,
        max_motor_velocity=3.0,
    ),
)


def _fmt(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def list_profiles() -> list[PolicyTuningProfile]:
    return list(BUILTIN_PROFILES)


def get_profile(name: str) -> PolicyTuningProfile:
    normalized = name.strip().lower()
    for profile in BUILTIN_PROFILES:
        if profile.name == normalized:
            return profile
    available = ", ".join(profile.name for profile in BUILTIN_PROFILES)
    raise KeyError(f"Unknown policy tuning profile {name!r}. Available: {available}")


def profiles_as_json(profiles: Iterable[PolicyTuningProfile] | None = None) -> str:
    payload = [asdict(profile) | {"env": profile.env()} for profile in (profiles or BUILTIN_PROFILES)]
    return json.dumps(payload, indent=2, sort_keys=True)


def print_profile_table(profiles: Iterable[PolicyTuningProfile] | None = None) -> None:
    rows = list(profiles or BUILTIN_PROFILES)
    print("Available Soridormi ONNX policy tuning profiles")
    print("================================================")
    for profile in rows:
        print(f"{profile.name}")
        print(f"  {profile.description}")
        print(
            "  "
            f"x={profile.command_x:g} y={profile.command_y:g} yaw={profile.command_yaw:g} "
            f"phase={profile.phase_frequency:g}Hz action_scale={profile.action_scale:g} "
            f"max_motor_velocity={profile.max_motor_velocity:g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List Soridormi ONNX policy tuning profiles.")
    parser.add_argument("profile", nargs="?", help="Profile name to print as shell exports")
    parser.add_argument("--json", action="store_true", help="Print all profiles as JSON")
    parser.add_argument("--shell", action="store_true", help="Print shell exports for the selected profile")
    args = parser.parse_args()

    if args.json:
        print(profiles_as_json())
        return

    if args.profile:
        profile = get_profile(args.profile)
        if args.shell:
            print(profile.shell_exports())
        else:
            print(json.dumps(asdict(profile) | {"env": profile.env()}, indent=2, sort_keys=True))
        return

    print_profile_table()


if __name__ == "__main__":
    main()
