from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from .local_tools import SoridormiLocalToolService
from .manifest import CapabilityBundle, build_soridormi_capability_bundle
from .runtime_tools import SoridormiRuntimeToolService

logger = logging.getLogger(__name__)
_SERVER_MODES = {"sim", "hardware_shadow", "hardware_dry_run"}


class _StreamableHttpApp:
    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await self.session_manager.handle_request(scope, receive, send)


def build_mcp_tools(bundle: CapabilityBundle) -> list[types.Tool]:
    return [
        types.Tool(
            name=tool.name,
            title=tool.display_name,
            description=tool.description,
            inputSchema=tool.input_schema,
            outputSchema=tool.output_schema,
        )
        for agent in bundle.agents
        for tool in agent.tools
    ]


def create_mcp_server(
    *,
    mode: str = "sim",
    adapter: str = "dry_run",
    service: SoridormiLocalToolService | SoridormiRuntimeToolService | None = None,
) -> Server:
    if mode not in _SERVER_MODES:
        raise ValueError(
            "the current MCP server supports only sim, hardware_shadow, and "
            "hardware_dry_run modes"
        )
    if adapter not in {"dry_run", "runtime"}:
        raise ValueError("adapter must be 'dry_run' or 'runtime'")
    tool_service = service
    if tool_service is None:
        tool_service = (
            SoridormiLocalToolService(mode=mode)
            if adapter == "dry_run"
            else SoridormiRuntimeToolService.from_env(mode=mode)
        )
    tools = build_mcp_tools(build_soridormi_capability_bundle(mode=mode))
    tool_names = {tool.name for tool in tools}
    server = Server("soridormi")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in tool_names:
            raise ValueError(f"unknown Soridormi MCP tool: {name}")
        result = await asyncio.to_thread(tool_service.call_tool, name, arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    return server


def create_asgi_app(
    *,
    mode: str = "sim",
    adapter: str = "dry_run",
    service: SoridormiLocalToolService | SoridormiRuntimeToolService | None = None,
    path: str = "/mcp",
) -> Starlette:
    server = create_mcp_server(mode=mode, adapter=adapter, service=service)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info("Soridormi MCP server ready at %s", path)
            yield

    return Starlette(
        routes=[Route(path, endpoint=_StreamableHttpApp(session_manager))],
        lifespan=lifespan,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Soridormi's safe dry-run tools over MCP Streamable HTTP."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("SORIDORMI_MCP_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SORIDORMI_MCP_PORT", "8000")),
    )
    parser.add_argument(
        "--path",
        default=os.getenv("SORIDORMI_MCP_PATH", "/mcp"),
    )
    parser.add_argument(
        "--mode",
        default=os.getenv("SORIDORMI_MCP_MODE", "sim"),
        choices=sorted(_SERVER_MODES),
    )
    parser.add_argument(
        "--adapter",
        default=os.getenv("SORIDORMI_MCP_ADAPTER", "dry_run"),
        choices=["dry_run", "runtime"],
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SORIDORMI_MCP_LOG_LEVEL", "info"),
    )
    args = parser.parse_args()

    uvicorn.run(
        create_asgi_app(mode=args.mode, adapter=args.adapter, path=args.path),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
