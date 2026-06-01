from __future__ import annotations

from contextlib import contextmanager

import pytest

from soridormi_sim.mujoco_viewer import MujocoViewerHandle, env_float


class FakeCam:
    def __init__(self) -> None:
        self.lookat = [0.0, 0.0, 0.0]
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0


class FakeViewer:
    def __init__(self) -> None:
        self.cam = FakeCam()
        self.sync_count = 0
        self.lock_count = 0

    def is_running(self) -> bool:
        return True

    @contextmanager
    def lock(self):
        self.lock_count += 1
        yield

    def sync(self) -> None:
        self.sync_count += 1


class FakeData:
    qpos = [1.25, -0.5, 0.18, 1.0]


def test_follow_camera_tracks_base_qpos_before_viewer_sync() -> None:
    viewer = FakeViewer()
    handle = MujocoViewerHandle(
        model=object(),
        data=FakeData(),
        enabled=False,
        follow_camera=True,
        camera_distance=2.0,
        camera_azimuth=90.0,
        camera_elevation=-12.5,
    )
    handle._viewer = viewer

    handle.sync()

    assert viewer.cam.lookat == [1.25, -0.5, 0.18]
    assert viewer.cam.distance == 2.0
    assert viewer.cam.azimuth == 90.0
    assert viewer.cam.elevation == -12.5
    assert viewer.lock_count == 1
    assert viewer.sync_count == 1


def test_follow_camera_is_noop_when_disabled() -> None:
    viewer = FakeViewer()
    handle = MujocoViewerHandle(
        model=object(),
        data=FakeData(),
        enabled=False,
        follow_camera=False,
        camera_distance=2.0,
    )
    handle._viewer = viewer

    handle.sync()

    assert viewer.cam.lookat == [0.0, 0.0, 0.0]
    assert viewer.cam.distance == 0.0
    assert viewer.lock_count == 0
    assert viewer.sync_count == 1


def test_env_float_parses_values_and_reports_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_float("SORIDORMI_TEST_FLOAT", 1.25) == 1.25

    monkeypatch.setenv("SORIDORMI_TEST_FLOAT", "2.5")
    assert env_float("SORIDORMI_TEST_FLOAT", 1.25) == 2.5

    monkeypatch.setenv("SORIDORMI_TEST_FLOAT", "not-a-float")
    with pytest.raises(ValueError, match="SORIDORMI_TEST_FLOAT must be a float"):
        env_float("SORIDORMI_TEST_FLOAT", 1.25)
