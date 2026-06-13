from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import importlib.util

import pytest

_MCP_HTTP_DEPS = ("httpx", "mcp", "uvicorn", "starlette")
_MISSING_MCP_HTTP_DEPS = [
    name for name in _MCP_HTTP_DEPS if importlib.util.find_spec(name) is None
]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MCP_HTTP_DEPS),
    reason=(
        "install the mcp extra to run Soridormi MCP HTTP server tests; "
        f"missing: {', '.join(_MISSING_MCP_HTTP_DEPS)}"
    ),
)

if not _MISSING_MCP_HTTP_DEPS:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from soridormi_runtime.mcp.http_server import build_mcp_tools, create_mcp_server
    from soridormi_runtime.mcp.runtime_tools import SoridormiRuntimeToolService
    from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"Soridormi MCP server did not listen on port {port}")


def test_mcp_tool_schemas_come_from_authoritative_manifest() -> None:
    bundle = build_soridormi_capability_bundle(mode="sim")
    expected = {
        tool.name: (tool.input_schema, tool.output_schema)
        for agent in bundle.agents
        for tool in agent.tools
    }

    actual = {
        tool.name: (tool.inputSchema, tool.outputSchema)
        for tool in build_mcp_tools(bundle)
    }

    assert actual == expected


def test_mcp_server_rejects_hardware_mode_until_runtime_adapter_exists() -> None:
    with pytest.raises(ValueError, match="hardware_dry_run"):
        create_mcp_server(mode="hardware")


def test_mcp_server_accepts_injected_runtime_adapter() -> None:
    service = object.__new__(SoridormiRuntimeToolService)
    server = create_mcp_server(mode="sim", adapter="runtime", service=service)

    assert server.name == "soridormi"


def test_streamable_http_preserves_plan_state_across_requests() -> None:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "soridormi_runtime.mcp.http_server",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port)

        async def exercise_server() -> None:
            async with httpx.AsyncClient(trust_env=False) as http_client:
                async with streamable_http_client(
                    f"http://127.0.0.1:{port}/mcp",
                    http_client=http_client,
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        listed = await session.list_tools()
                        assert len(listed.tools) == 14
                        plan = await session.call_tool(
                            "soridormi.motion.create_plan",
                            {
                                "commands": [
                                    {
                                        "vx": 0.0,
                                        "vy": 0.0,
                                        "yaw": 0.0,
                                        "duration_s": 0.05,
                                    }
                                ]
                            },
                        )
                        assert plan.structuredContent is not None
                        plan_id = plan.structuredContent["plan_id"]
                        executed = await session.call_tool(
                            "soridormi.motion.execute_plan",
                            {"plan_id": plan_id},
                        )
                        assert executed.structuredContent is not None
                        assert executed.structuredContent["completed"] is True
                        assert executed.structuredContent["dry_run_only"] is True

        asyncio.run(exercise_server())
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
