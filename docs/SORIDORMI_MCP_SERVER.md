# Soridormi MCP server

Soridormi publishes its robot-facing capabilities through an MCP Streamable
HTTP service. Chromie remains a separate deployment and connects as an MCP
client.

```text
chromie-agent container
        |
        | MCP Streamable HTTP
        v
soridormi-mcp container
        |
        v
Soridormi safe tool/runtime boundary
```

## Current safety boundary

The server exposes the nine tools in
`soridormi_runtime.mcp.manifest`. The current implementation wraps
`SoridormiLocalToolService`, so motion execution is dry-run only and never
sends motor commands. Plan and emergency-stop state are shared across HTTP
requests within one server process.

Do not treat this server as proof of hardware motion acceptance. A future
runtime adapter must replace the local dry-run service before hardware mode is
enabled.

## Run in the Soridormi container

```bash
./scripts/run_mcp_server.sh
```

This builds and starts the dedicated `soridormi-mcp` container from
`compose.mcp.yaml`. It publishes:

```text
http://127.0.0.1:8000/mcp
```

Override `SORIDORMI_MCP_PORT` in `.env` when port 8000 is unavailable.

Chromie should use a reachable host address, for example:

```env
SORIDORMI_MCP_URL=http://host.docker.internal:8000/mcp
```

Keep `host.docker.internal` in Chromie's `NO_PROXY` list so robot-control MCP
traffic does not pass through a general HTTP proxy.

## Run directly for development

```bash
python -m soridormi_runtime.mcp.http_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode sim
```

The server uses stateless MCP transport with JSON responses. Application state
such as created plans remains process-local and protected against concurrent
tool calls.
