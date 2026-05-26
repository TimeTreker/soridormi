from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PLAYGROUND_ROOT = Path("/workspaces/Open_Duck_Playground")
DEFAULT_POLICY_PATH = Path("/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")
DEFAULT_MODEL_PATH = DEFAULT_PLAYGROUND_ROOT / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
DEFAULT_REFERENCE_PATH = DEFAULT_PLAYGROUND_ROOT / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
DEFAULT_OUTPUT_DIR = Path("/data/official_baseline")


@dataclass(frozen=True)
class OfficialBaselineCommand:
    x: float = 0.15
    y: float = 0.0
    yaw: float = 0.0
    neck_pitch: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

    def as_list(self) -> list[float]:
        return [
            float(self.x),
            float(self.y),
            float(self.yaw),
            float(self.neck_pitch),
            float(self.head_pitch),
            float(self.head_yaw),
            float(self.head_roll),
        ]


@dataclass(frozen=True)
class OfficialBaselineConfig:
    playground_root: Path = DEFAULT_PLAYGROUND_ROOT
    model_path: Path = DEFAULT_MODEL_PATH
    reference_data: Path = DEFAULT_REFERENCE_PATH
    onnx_model_path: Path = DEFAULT_POLICY_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    command: OfficialBaselineCommand = OfficialBaselineCommand()
    max_seconds: float = 20.0
    viewer: bool = True
    realtime: bool = True
    standing: bool = False
    summary_prefix: str = "official_open_duck_forward"


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            "Make sure submodules are initialized and paths are mounted inside the container.\n"
            "Typical host command: git submodule update --init --recursive "
            "workspace/Open_Duck_Playground workspace/Open_Duck_Mini"
        )


def _install_lightweight_official_utils_stub() -> None:
    """Avoid importing JAX just because official mujoco_infer imports LowPassActionFilter.

    The official Open Duck file imports playground.common.utils, whose only symbol
    needed by mujoco_infer.py is LowPassActionFilter. The filter is currently
    commented out in the official run loop, and this baseline intentionally mirrors
    that unfiltered path. Installing this tiny module before importing official
    mujoco_infer.py lets the official baseline run in a lightweight Docker image
    without pulling JAX into the simulator image.
    """

    module_name = "playground.common.utils"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class LowPassActionFilter:  # pragma: no cover - used only if official code enables filtering
        def __init__(self, control_freq: float, cutoff_frequency: float = 30.0) -> None:
            self.last_action: Any = 0
            self.current_action: Any = 0
            self.control_freq = float(control_freq)
            self.cutoff_frequency = float(cutoff_frequency)
            self.alpha = (1.0 / self.cutoff_frequency) / (
                1.0 / self.control_freq + 1.0 / self.cutoff_frequency
            )

        def push(self, action: Any) -> None:
            self.current_action = action

        def get_filtered_action(self) -> Any:
            self.last_action = self.alpha * self.last_action + (1.0 - self.alpha) * self.current_action
            return self.last_action

    module.LowPassActionFilter = LowPassActionFilter
    sys.modules[module_name] = module


def _install_lightweight_open_duck_base_stub(playground_root: Path) -> None:
    """Avoid importing official base.py, which pulls JAX/MJX into the baseline runner.

    Official mujoco_infer_base.py only needs base.get_assets() while constructing
    the MuJoCo model. The official base.py also imports jax, ml_collections and
    mujoco_playground for training environments, which are unnecessary for this
    pure MuJoCo + ONNX baseline. This stub provides get_assets() with the same
    practical purpose: load XML/mesh/texture assets from Open_Duck_Playground so
    MjModel.from_xml_string(..., assets=...) can resolve included files.
    """

    module_name = "playground.open_duck_mini_v2.base"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    def get_assets() -> dict[str, bytes]:
        root = Path(playground_root) / "playground" / "open_duck_mini_v2"
        assets: dict[str, bytes] = {}

        search_roots = [
            root,
            root / "xmls",
            root / "xmls" / "assets",
            root / "assets",
        ]
        files: list[Path] = []
        for search_root in search_roots:
            if not search_root.exists():
                continue
            files.extend([p for p in search_root.rglob("*") if p.is_file()])

        for file_path in files:
            try:
                data = file_path.read_bytes()
            except OSError:
                continue

            # Important: MuJoCo checks asset names by basename when loading from
            # an in-memory assets dict. If the dict contains both "head.stl"
            # and "assets/head.stl", or two relative paths ending in the same
            # basename, MuJoCo raises:
            #   ValueError: Repeated file name in assets dict: head.stl
            # Therefore the lightweight JAX-free stub intentionally stores only
            # one entry per basename. This mirrors the practical behavior needed
            # by the official XMLs while avoiding duplicate mesh names across
            # asset directories.
            assets.setdefault(file_path.name, data)

        return assets

    module.get_assets = get_assets  # type: ignore[attr-defined]
    sys.modules[module_name] = module


def _prepare_official_imports(playground_root: Path) -> None:
    root = str(playground_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    _install_lightweight_official_utils_stub()
    _install_lightweight_open_duck_base_stub(playground_root)


def _load_official_mjinfer(config: OfficialBaselineConfig):
    _prepare_official_imports(config.playground_root)
    from playground.open_duck_mini_v2.mujoco_infer import (  # type: ignore[import-not-found]
        USE_MOTOR_SPEED_LIMITS,
        MjInfer,
    )

    return MjInfer, bool(USE_MOTOR_SPEED_LIMITS)


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "abs_max": float(np.abs(arr).max()),
    }


def _vector_stats(vectors: list[list[float]]) -> dict[str, Any] | None:
    if not vectors:
        return None
    arr = np.asarray(vectors, dtype=float)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "abs_max": float(np.abs(arr).max()),
    }


class OfficialForwardRunner:
    """Non-interactive official Open Duck MuJoCo baseline runner.

    This class reuses the official playground.open_duck_mini_v2.mujoco_infer.MjInfer
    implementation instead of rewriting the observation/action pipeline in
    Soridormi. The run loop is intentionally close to the official script, but
    adds:
      - fixed command values from CLI/env instead of keyboard-only input
      - a finite max_seconds duration
      - JSON summary output for comparison with Soridormi logs
    """

    def __init__(self, config: OfficialBaselineConfig) -> None:
        self.config = config
        _check_file(config.model_path, "Open Duck MuJoCo XML")
        _check_file(config.reference_data, "Open Duck polynomial reference data")
        _check_file(config.onnx_model_path, "ONNX policy")
        self._mjinfer_cls, self.use_motor_speed_limits = _load_official_mjinfer(config)
        self.mjinfer = self._mjinfer_cls(
            str(config.model_path),
            str(config.reference_data),
            str(config.onnx_model_path),
            config.standing,
        )
        self.mjinfer.commands = config.command.as_list()
        self.records: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        import mujoco

        if self.config.viewer:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(
                self.mjinfer.model,
                self.mjinfer.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.mjinfer.key_callback,
            ) as viewer:
                return self._run_loop(viewer=viewer)

        return self._run_loop(viewer=None)

    def _run_loop(self, viewer: Any | None) -> dict[str, Any]:
        import mujoco

        start_wall = time.monotonic()
        counter = 0
        policy_steps = 0
        base_start = np.array(self.mjinfer.get_floating_base_qpos(self.mjinfer.data.qpos)[:3], dtype=float)
        base_z_values: list[float] = []
        action_values: list[float] = []
        motor_target_values: list[float] = []
        contact_values: list[float] = []
        base_positions: list[list[float]] = []

        while True:
            step_start = time.monotonic()
            elapsed = step_start - start_wall
            if elapsed >= self.config.max_seconds:
                break

            mujoco.mj_step(self.mjinfer.model, self.mjinfer.data)
            counter += 1

            if counter % self.mjinfer.decimation == 0:
                if not self.mjinfer.standing:
                    self.mjinfer.imitation_i += 1.0 * self.mjinfer.phase_frequency_factor
                    self.mjinfer.imitation_i = self.mjinfer.imitation_i % self.mjinfer.PRM.nb_steps_in_period
                    self.mjinfer.imitation_phase = np.array(
                        [
                            np.cos(
                                self.mjinfer.imitation_i
                                / self.mjinfer.PRM.nb_steps_in_period
                                * 2.0
                                * np.pi
                            ),
                            np.sin(
                                self.mjinfer.imitation_i
                                / self.mjinfer.PRM.nb_steps_in_period
                                * 2.0
                                * np.pi
                            ),
                        ]
                    )

                obs = self.mjinfer.get_obs(self.mjinfer.data, self.mjinfer.commands)
                action = np.asarray(self.mjinfer.policy.infer(obs), dtype=float)

                self.mjinfer.last_last_last_action = self.mjinfer.last_last_action.copy()
                self.mjinfer.last_last_action = self.mjinfer.last_action.copy()
                self.mjinfer.last_action = action.copy()

                self.mjinfer.motor_targets = self.mjinfer.default_actuator + action * self.mjinfer.action_scale
                if self.use_motor_speed_limits:
                    self.mjinfer.motor_targets = np.clip(
                        self.mjinfer.motor_targets,
                        self.mjinfer.prev_motor_targets
                        - self.mjinfer.max_motor_velocity
                        * (self.mjinfer.sim_dt * self.mjinfer.decimation),
                        self.mjinfer.prev_motor_targets
                        + self.mjinfer.max_motor_velocity
                        * (self.mjinfer.sim_dt * self.mjinfer.decimation),
                    )
                self.mjinfer.prev_motor_targets = self.mjinfer.motor_targets.copy()
                self.mjinfer.data.ctrl = self.mjinfer.motor_targets.copy()

                contacts = [float(x) for x in self.mjinfer.get_feet_contacts(self.mjinfer.data)]
                base = [float(x) for x in self.mjinfer.get_floating_base_qpos(self.mjinfer.data.qpos)[:3]]
                policy_steps += 1
                base_z_values.append(base[2])
                action_values.extend([float(x) for x in action])
                motor_target_values.extend([float(x) for x in self.mjinfer.motor_targets])
                contact_values.extend(contacts)
                base_positions.append(base)
                self.records.append(
                    {
                        "policy_step": policy_steps,
                        "sim_time": float(self.mjinfer.data.time),
                        "base_position_xyz": base,
                        "contacts": contacts,
                        "imitation_i": float(self.mjinfer.imitation_i),
                        "imitation_phase": [float(x) for x in self.mjinfer.imitation_phase],
                        "action_min": float(action.min()),
                        "action_max": float(action.max()),
                        "motor_target_min": float(np.min(self.mjinfer.motor_targets)),
                        "motor_target_max": float(np.max(self.mjinfer.motor_targets)),
                    }
                )

            if viewer is not None:
                viewer.sync()

            if self.config.realtime:
                sleep_time = self.mjinfer.model.opt.timestep - (time.monotonic() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        base_final = np.array(
            self.mjinfer.get_floating_base_qpos(self.mjinfer.data.qpos)[:3],
            dtype=float,
        )
        displacement = base_final - base_start
        summary = {
            "kind": "official_open_duck_baseline",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "playground_root": str(self.config.playground_root),
            "model_path": str(self.config.model_path),
            "reference_data": str(self.config.reference_data),
            "onnx_model_path": str(self.config.onnx_model_path),
            "command": asdict(self.config.command),
            "standing": self.config.standing,
            "viewer": self.config.viewer,
            "max_seconds": self.config.max_seconds,
            "wall_duration_seconds": float(time.monotonic() - start_wall),
            "sim_time_seconds": float(self.mjinfer.data.time),
            "mujoco_steps": int(counter),
            "policy_steps": int(policy_steps),
            "sim_dt": float(self.mjinfer.sim_dt),
            "decimation": int(self.mjinfer.decimation),
            "policy_dt": float(self.mjinfer.sim_dt * self.mjinfer.decimation),
            "action_scale": float(self.mjinfer.action_scale),
            "max_motor_velocity": float(self.mjinfer.max_motor_velocity),
            "use_motor_speed_limits": bool(self.use_motor_speed_limits),
            "phase_period_steps": int(getattr(self.mjinfer.PRM, "nb_steps_in_period", 0)),
            "base_start_xyz": [float(x) for x in base_start],
            "base_final_xyz": [float(x) for x in base_final],
            "base_displacement_xyz": [float(x) for x in displacement],
            "base_z_stats": _stats(base_z_values),
            "action_stats": _stats(action_values),
            "motor_target_stats": _stats(motor_target_values),
            "contact_stats": _stats(contact_values),
            "base_position_stats": _vector_stats(base_positions),
            "first_records": self.records[:5],
            "last_records": self.records[-5:],
        }
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official Open Duck Mini v2 MuJoCo inference path with a fixed command.",
    )
    parser.add_argument("--playground-root", type=Path, default=DEFAULT_PLAYGROUND_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-data", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--onnx-model-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-prefix", default="official_open_duck_forward")
    parser.add_argument("--max-seconds", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_MAX_SECONDS", "20")))
    parser.add_argument("--command-x", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_COMMAND_X", "0.15")))
    parser.add_argument("--command-y", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_COMMAND_Y", "0.0")))
    parser.add_argument("--command-yaw", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_COMMAND_YAW", "0.0")))
    parser.add_argument("--neck-pitch", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_NECK_PITCH", "0.0")))
    parser.add_argument("--head-pitch", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_HEAD_PITCH", "0.0")))
    parser.add_argument("--head-yaw", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_HEAD_YAW", "0.0")))
    parser.add_argument("--head-roll", type=float, default=float(os.environ.get("SORIDORMI_OFFICIAL_HEAD_ROLL", "0.0")))
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--no-viewer", action="store_true", default=False)
    parser.add_argument("--no-realtime", action="store_true", default=False)
    parser.add_argument(
        "--normal-exit",
        action="store_true",
        default=False,
        help=(
            "Use normal Python interpreter shutdown. By default this diagnostic runner "
            "exits immediately after writing the summary to avoid MuJoCo/viewer teardown "
            "segfaults observed in some Docker/X11 environments."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OfficialBaselineConfig:
    playground_root = _path(args.playground_root)
    return OfficialBaselineConfig(
        playground_root=playground_root,
        model_path=_path(args.model_path),
        reference_data=_path(args.reference_data),
        onnx_model_path=_path(args.onnx_model_path),
        output_dir=Path(args.output_dir),
        summary_prefix=str(args.summary_prefix),
        command=OfficialBaselineCommand(
            x=float(args.command_x),
            y=float(args.command_y),
            yaw=float(args.command_yaw),
            neck_pitch=float(args.neck_pitch),
            head_pitch=float(args.head_pitch),
            head_yaw=float(args.head_yaw),
            head_roll=float(args.head_roll),
        ),
        max_seconds=float(args.max_seconds),
        viewer=not bool(args.no_viewer),
        realtime=not bool(args.no_realtime),
        standing=bool(args.standing),
    )


def write_summary(summary: dict[str, Any], output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    latest = output_dir / "latest_official_baseline.json"
    latest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_summary(summary: dict[str, Any], path: Path) -> None:
    dx, dy, dz = summary["base_displacement_xyz"]
    print("Official Open Duck baseline finished")
    print("====================================")
    print(f"Summary: {path}")
    print(f"Command: {summary['command']}")
    print(f"Policy steps: {summary['policy_steps']}")
    print(f"Sim time: {summary['sim_time_seconds']:.3f} s")
    print(f"Base displacement xyz: [{dx:.4f}, {dy:.4f}, {dz:.4f}]")
    print(f"Action stats: {summary['action_stats']}")
    print(f"Motor target stats: {summary['motor_target_stats']}")
    print(f"Contact stats: {summary['contact_stats']}")


def _should_fast_exit(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "normal_exit", False)):
        return False

    value = os.environ.get("SORIDORMI_OFFICIAL_FAST_EXIT", "1").strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    runner = OfficialForwardRunner(config)
    summary = runner.run()
    path = write_summary(summary, config.output_dir, config.summary_prefix)
    print_summary(summary, path)

    # In Docker + X11, MuJoCo/passive-viewer cleanup can segfault after a
    # successful run, even though the summary has already been written. This
    # baseline is a diagnostic runner, so make successful completion reliable by
    # bypassing Python/C-extension destructors unless explicitly disabled.
    if _should_fast_exit(args):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
