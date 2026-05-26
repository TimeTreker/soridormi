from __future__ import annotations

from dataclasses import dataclass

from soridormi_sim.mujoco_backend import MujocoBackend


@dataclass
class OldFootContactConfig:
    left_geoms: list[str]
    right_geoms: list[str]


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def _read_site_or_geom_position(self, site_name: str, fallback_geoms: list[str]):
        self.calls.append((site_name, fallback_geoms))
        return [0.0, 0.0, 0.0]


def test_read_feet_positions_accepts_old_contact_config_without_sites() -> None:
    backend = DummyBackend()

    class PolicyObservation:
        foot_contact = OldFootContactConfig(
            left_geoms=["left_foot_bottom_tpu"],
            right_geoms=["right_foot_bottom_tpu"],
        )

    class Config:
        policy_observation = PolicyObservation()

    backend.config = Config()

    result = MujocoBackend._read_feet_positions(backend)  # type: ignore[arg-type]

    assert result == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert backend.calls == [
        ("", ["left_foot_bottom_tpu"]),
        ("", ["right_foot_bottom_tpu"]),
    ]
