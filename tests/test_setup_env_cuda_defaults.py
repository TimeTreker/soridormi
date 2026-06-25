from __future__ import annotations

from pathlib import Path


def test_setup_env_cuda_image_tags_match_default_bases() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "setup_env.sh").read_text(encoding="utf-8")

    assert "SORIDORMI_RUNTIME_IMAGE:=soridormi-runtime:cuda13.1-cudnn-dev" in source
    assert "SORIDORMI_RUNTIME_MCP_IMAGE:=soridormi-runtime-mcp:cuda13.1-cudnn-dev" in source
    assert "SORIDORMI_SIM_IMAGE:=soridormi-sim:cuda13.1-cudnn" in source
    assert "RUNTIME_DEV_BASE:=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04" in source
    assert "SIM_BASE:=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04" in source
    assert "RUNTIME_DEV_BASE:=nvidia/cuda:12.8.1" not in source
    assert "SIM_BASE:=nvidia/cuda:12.8.1" not in source
