from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from soridormi_api import MotorCommand, RobotState
from soridormi_runtime.policy_command import PolicyCommand


_EPS = 1e-9


@dataclass(frozen=True)
class WalkingRewardConfig:
    """Walking-quality reward for residual policy fine-tuning.

    The reward is intentionally decomposed and inspectable. M6.18 is the reward
    backbone; M6.19 can tune these weights once the residual trainer exists.
    """

    alive_bonus: float = 0.05
    forward_tracking_weight: float = 1.0
    lateral_tracking_weight: float = 0.25
    yaw_tracking_weight: float = 0.15
    upright_weight: float = 0.35
    height_weight: float = 0.10
    action_l2_weight: float = 0.01
    residual_l2_weight: float = 0.05
    action_rate_weight: float = 0.02
    lateral_drift_weight: float = 0.05
    vertical_motion_weight: float = 0.02
    fall_penalty: float = 5.0
    forward_velocity_sigma: float = 0.20
    lateral_velocity_sigma: float = 0.12
    yaw_velocity_sigma: float = 0.50
    target_height: float = 0.30
    height_sigma: float = 0.08
    min_upright: float = 0.65
    fall_height: float = 0.14
    terminate_on_fall: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkingRewardResult:
    reward: float
    terminated: bool
    terms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float | bool | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "terms": {str(k): float(v) for k, v in self.terms.items()},
            "diagnostics": dict(self.diagnostics),
        }


def compute_walking_reward(
    before: RobotState,
    after: RobotState,
    *,
    command: PolicyCommand,
    motor_command: MotorCommand | None = None,
    final_action: np.ndarray | list[float] | None = None,
    residual_action: np.ndarray | list[float] | None = None,
    previous_final_action: np.ndarray | list[float] | None = None,
    config: WalkingRewardConfig | None = None,
) -> WalkingRewardResult:
    """Compute one-step walking reward from a simulator transition.

    The reward follows the high-level objective we want for fine-tuning:
    track commanded velocity, stay upright, avoid excessive action/residual,
    and terminate/penalize falls. It only uses fields already exposed through
    RobotState/MotorCommand so it remains compatible with the sim-to-real API.
    """

    cfg = config or WalkingRewardConfig()
    dt = max(float(after.time) - float(before.time), _EPS)
    before_pos = _position(before)
    after_pos = _position(after, fallback=before_pos)
    delta = after_pos - before_pos
    velocity = delta / dt

    yaw_before = _yaw_from_state(before)
    yaw_after = _yaw_from_state(after)
    yaw_rate = _angle_diff(yaw_after, yaw_before) / dt if yaw_before is not None and yaw_after is not None else 0.0

    upright = _upright_score(after)
    height = float(after_pos[2]) if after_pos.shape[0] >= 3 else None
    fallen = _is_fallen(height=height, upright=upright, cfg=cfg)

    vx = float(velocity[0]) if velocity.shape[0] > 0 else 0.0
    vy = float(velocity[1]) if velocity.shape[0] > 1 else 0.0
    vz = float(velocity[2]) if velocity.shape[0] > 2 else 0.0

    target_vx = _command_component(command, 0, "x_velocity")
    target_vy = _command_component(command, 1, "y_velocity")
    target_yaw = _command_component(command, 2, "yaw_velocity")

    forward_tracking = _gaussian_tracking(vx, target_vx, cfg.forward_velocity_sigma)
    lateral_tracking = _gaussian_tracking(vy, target_vy, cfg.lateral_velocity_sigma)
    yaw_tracking = _gaussian_tracking(yaw_rate, target_yaw, cfg.yaw_velocity_sigma)
    upright_reward = max(0.0, upright)
    height_reward = 0.0 if height is None else _gaussian_tracking(height, cfg.target_height, cfg.height_sigma)

    final = _optional_array(final_action)
    residual = _optional_array(residual_action)
    previous = _optional_array(previous_final_action)
    action_l2 = _mean_square(final)
    residual_l2 = _mean_square(residual)
    action_rate_l2 = _mean_square(final - previous) if final is not None and previous is not None else 0.0
    lateral_drift = abs(float(delta[1])) if delta.shape[0] > 1 else 0.0
    vertical_motion = abs(vz)

    terms = {
        "alive_bonus": cfg.alive_bonus,
        "forward_tracking": cfg.forward_tracking_weight * forward_tracking,
        "lateral_tracking": cfg.lateral_tracking_weight * lateral_tracking,
        "yaw_tracking": cfg.yaw_tracking_weight * yaw_tracking,
        "upright": cfg.upright_weight * upright_reward,
        "height": cfg.height_weight * height_reward,
        "action_l2_penalty": -cfg.action_l2_weight * action_l2,
        "residual_l2_penalty": -cfg.residual_l2_weight * residual_l2,
        "action_rate_penalty": -cfg.action_rate_weight * action_rate_l2,
        "lateral_drift_penalty": -cfg.lateral_drift_weight * lateral_drift,
        "vertical_motion_penalty": -cfg.vertical_motion_weight * vertical_motion,
        "fall_penalty": -cfg.fall_penalty if fallen else 0.0,
    }
    reward = float(sum(terms.values()))
    diagnostics: dict[str, float | bool | None] = {
        "dt": float(dt),
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "target_vx": float(target_vx),
        "target_vy": float(target_vy),
        "yaw_rate": float(yaw_rate),
        "target_yaw_rate": float(target_yaw),
        "upright": float(upright),
        "height": height,
        "fallen": bool(fallen),
        "action_l2": float(action_l2),
        "residual_l2": float(residual_l2),
        "action_rate_l2": float(action_rate_l2),
        "lateral_drift": float(lateral_drift),
    }
    if motor_command is not None:
        diagnostics["motor_target_abs_max"] = max((abs(float(x)) for x in motor_command.positions), default=0.0)
    return WalkingRewardResult(reward=reward, terminated=bool(cfg.terminate_on_fall and fallen), terms=terms, diagnostics=diagnostics)



def _command_component(command: PolicyCommand, index: int, attr: str) -> float:
    value = getattr(command, attr, None)
    if value is not None:
        return float(value)
    as_list = getattr(command, "as_list", None)
    if callable(as_list):
        values = list(as_list())
        if len(values) > index:
            return float(values[index])
    return 0.0

def _position(state: RobotState, *, fallback: np.ndarray | None = None) -> np.ndarray:
    if state.base_position_xyz is None:
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(state.base_position_xyz, dtype=np.float64)
    if arr.shape != (3,):
        return np.zeros(3, dtype=np.float64) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return arr


def _yaw_from_state(state: RobotState) -> float | None:
    quat = state.base_quat_wxyz or state.imu.quat_wxyz
    if quat is None or len(quat) != 4:
        return None
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= _EPS:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _upright_score(state: RobotState) -> float:
    quat = state.base_quat_wxyz or state.imu.quat_wxyz
    if quat is None or len(quat) != 4:
        return 1.0
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= _EPS:
        return 0.0
    x, y = x / norm, y / norm
    # z component of the body z-axis in world coordinates.
    return float(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def _is_fallen(*, height: float | None, upright: float, cfg: WalkingRewardConfig) -> bool:
    if height is not None and height < cfg.fall_height:
        return True
    return upright < cfg.min_upright


def _gaussian_tracking(value: float, target: float, sigma: float) -> float:
    sigma = max(float(sigma), _EPS)
    error = float(value) - float(target)
    return float(math.exp(-(error * error) / (2.0 * sigma * sigma)))


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _optional_array(values: np.ndarray | list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _mean_square(values: np.ndarray | None) -> float:
    if values is None:
        return 0.0
    if values.size == 0:
        return 0.0
    return float(np.mean(np.square(values)))
