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

## Safety boundary

The server exposes the nine tools in
`soridormi_runtime.mcp.manifest`. Plan and emergency-stop state are shared
across HTTP requests within one server process.

The default adapter wraps `SoridormiLocalToolService`, so motion execution is
dry-run only and never sends motor commands:

```bash
./scripts/run_mcp_server.sh
```

The runtime adapter drives the existing Soridormi robot/controller interfaces
and is deliberately limited to `sim` until `HardwareRobot` is implemented:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/run_runtime_mcp_server.sh
```

Do not run the standalone runtime loop and the runtime MCP adapter against the
same robot backend at the same time. The adapter owns the control loop while a
plan is active.

`motion.stop`, `motion.cancel`, and `safety.emergency_stop` can preempt between
control ticks. Cancelling an in-flight MCP request also transitions the robot
to safe hold. Emergency-stop state remains active until the MCP process is
restarted; inspect robot state before resuming.

## Run the dry-run container

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

Run the runtime adapter directly only from an environment with Soridormi
runtime dependencies, policy assets, and a reachable simulator:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy \
python -m soridormi_runtime.mcp.http_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode sim \
  --adapter runtime
```

The server uses stateless MCP transport with JSON responses. Application state
such as created plans remains process-local and protected against concurrent
tool calls.
