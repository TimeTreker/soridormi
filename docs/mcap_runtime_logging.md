# M2.9: MCAP runtime logging

Soridormi runtime can record robot state and motor commands to MCAP.

MCAP is the preferred robotics logging format for Soridormi. The first implementation stores JSON payloads inside MCAP channels so the robot API can keep evolving without requiring ROS 2 or Protobuf schemas.

## Topics

The runtime logger writes:

```text
/soridormi/robot_state
/soridormi/motor_command
/soridormi/runtime_status
```

## Enable logging

Start the simulator, for example:

```bash
./scripts/run_auto_reset_stand_server.sh
```

Run the logged runtime:

```bash
./scripts/run_logged_stand_runtime.sh
```

This defaults to:

```bash
SORIDORMI_RUNTIME_LOG=1
SORIDORMI_RUNTIME_LOG_FORMAT=mcap
SORIDORMI_RUNTIME_LOG_DIR=/data/logs
SORIDORMI_RUNTIME_MODE=stand
```

The output appears on the host under:

```text
data/logs/runtime_YYYYMMDD_HHMMSS.mcap
```

## JSONL fallback

For simple debugging, use JSONL:

```bash
SORIDORMI_RUNTIME_LOG=1 \
SORIDORMI_RUNTIME_LOG_FORMAT=jsonl \
SORIDORMI_RUNTIME_MODE=stand \
./scripts/run_runtime_loop.sh
```

## Inspect a log

Inspect the newest log:

```bash
./scripts/inspect_latest_log.sh
```

Or inspect a specific file inside the runtime container:

```bash
./scripts/enter_runtime_dev.sh
python -m soridormi_runtime.inspect_log /data/logs/runtime_YYYYMMDD_HHMMSS.mcap
```

## Rebuild note

This milestone adds the Python package `mcap`, so rebuild once after applying it:

```bash
./scripts/build_sim.sh
```

After that, normal source-code changes do not require rebuilding.
