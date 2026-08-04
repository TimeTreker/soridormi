from __future__ import annotations

import sys
from pathlib import Path

from soridormi_sim.official_open_duck_baseline import (
    OfficialBaselineCommand,
    _install_lightweight_official_utils_stub,
    _install_lightweight_open_duck_base_stub,
    build_arg_parser,
    config_from_args,
    write_summary,
)


def test_official_baseline_command_as_list() -> None:
    command = OfficialBaselineCommand(x=0.15, y=-0.02, yaw=0.3, head_yaw=0.1)

    assert command.as_list() == [0.15, -0.02, 0.3, 0.0, 0.0, 0.1, 0.0]


def test_official_baseline_parser_builds_config(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--playground-root",
            str(tmp_path / "playground"),
            "--model-path",
            str(tmp_path / "model.xml"),
            "--reference-data",
            str(tmp_path / "reference.pkl"),
            "--onnx-model-path",
            str(tmp_path / "policy.onnx"),
            "--output-dir",
            str(tmp_path / "out"),
            "--command-x",
            "0.15",
            "--command-yaw",
            "-0.2",
            "--max-seconds",
            "3",
            "--no-viewer",
            "--no-realtime",
        ]
    )

    config = config_from_args(args)

    assert config.command.x == 0.15
    assert config.command.yaw == -0.2
    assert config.max_seconds == 3.0
    assert config.viewer is False
    assert config.realtime is False


def test_lightweight_official_utils_stub_installs_filter() -> None:
    sys.modules.pop("playground.common.utils", None)

    _install_lightweight_official_utils_stub()

    module = sys.modules["playground.common.utils"]
    filt = module.LowPassActionFilter(50, cutoff_frequency=25)
    filt.push(1.0)

    assert hasattr(module, "LowPassActionFilter")
    assert filt.get_filtered_action() > 0.0


def test_write_summary_creates_latest_file(tmp_path: Path) -> None:
    summary = {
        "kind": "official_open_duck_baseline",
        "base_displacement_xyz": [0.1, 0.0, 0.0],
    }

    path = write_summary(summary, tmp_path, "official_test")

    assert path.exists()
    assert (tmp_path / "latest_official_baseline.json").exists()
    assert "official_open_duck_baseline" in path.read_text()



def test_lightweight_open_duck_base_stub_installs_asset_loader(tmp_path: Path) -> None:
    sys.modules.pop("playground.open_duck_mini_v2.base", None)

    root = tmp_path / "Open_Duck_Playground"
    asset_dir = root / "playground" / "open_duck_mini_v2" / "xmls" / "assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "dummy_mesh.stl").write_bytes(b"solid dummy\nendsolid dummy\n")

    _install_lightweight_open_duck_base_stub(root)

    module = sys.modules["playground.open_duck_mini_v2.base"]
    assets = module.get_assets()

    assert "dummy_mesh.stl" in assets
    assert "assets/dummy_mesh.stl" not in assets


def test_lightweight_open_duck_base_stub_deduplicates_asset_basenames(tmp_path: Path) -> None:
    sys.modules.pop("playground.open_duck_mini_v2.base", None)

    root = tmp_path / "Open_Duck_Playground"
    first = root / "playground" / "open_duck_mini_v2" / "xmls" / "assets" / "head.stl"
    second = root / "playground" / "open_duck_mini_v2" / "assets" / "head.stl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"solid first\nendsolid first\n")
    second.write_bytes(b"solid second\nendsolid second\n")

    _install_lightweight_open_duck_base_stub(root)

    module = sys.modules["playground.open_duck_mini_v2.base"]
    assets = module.get_assets()

    assert list(k for k in assets if k.endswith("head.stl")) == ["head.stl"]


def test_should_fast_exit_default(monkeypatch) -> None:
    from soridormi_sim.official_open_duck_baseline import _should_fast_exit, build_arg_parser

    monkeypatch.delenv("SORIDORMI_OFFICIAL_FAST_EXIT", raising=False)
    args = build_arg_parser().parse_args([])

    assert _should_fast_exit(args) is True


def test_should_fast_exit_can_be_disabled_by_env(monkeypatch) -> None:
    from soridormi_sim.official_open_duck_baseline import _should_fast_exit, build_arg_parser

    monkeypatch.setenv("SORIDORMI_OFFICIAL_FAST_EXIT", "0")
    args = build_arg_parser().parse_args([])

    assert _should_fast_exit(args) is False


def test_should_fast_exit_can_be_disabled_by_flag(monkeypatch) -> None:
    from soridormi_sim.official_open_duck_baseline import _should_fast_exit, build_arg_parser

    monkeypatch.setenv("SORIDORMI_OFFICIAL_FAST_EXIT", "1")
    args = build_arg_parser().parse_args(["--normal-exit"])

    assert _should_fast_exit(args) is False
