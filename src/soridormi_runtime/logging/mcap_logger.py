from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from soridormi_api import MotorCommand, RobotState

from .base import default_log_dir, model_to_json_dict, now_ns


_STATE_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "type": {"const": "robot_state"},
        "step_index": {"type": "integer"},
        "time_wall_ns": {"type": "integer"},
        "robot_time": {"type": "number"},
        "mode": {"type": "string"},
        "backend": {"type": "string"},
        "state": {"type": "object"},
    },
}

_COMMAND_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "type": {"const": "motor_command"},
        "step_index": {"type": "integer"},
        "time_wall_ns": {"type": "integer"},
        "robot_time": {"type": "number"},
        "mode": {"type": "string"},
        "backend": {"type": "string"},
        "command": {"type": "object"},
    },
}

_STATUS_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "type": {"const": "runtime_status"},
        "time_wall_ns": {"type": "integer"},
        "mode": {"type": "string"},
        "backend": {"type": "string"},
        "message": {"type": "string"},
    },
}


class McapRuntimeLogger:
    """MCAP runtime logger using JSON payloads.

    We intentionally store JSON in MCAP first. This preserves robotics-friendly
    timestamped topics while keeping Soridormi independent of ROS 2 or Protobuf
    schemas during early API iteration.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        every_n: int = 1,
        prefix: str = "runtime",
        mode: str = "unknown",
        backend: str = "unknown",
    ) -> None:
        try:
            from mcap.writer import Writer
        except ImportError as exc:
            raise RuntimeError(
                "MCAP logging requested, but the 'mcap' Python package is not installed. "
                "Run './scripts/build_sim.sh' after applying the M2.9 update, or set "
                "SORIDORMI_RUNTIME_LOG_FORMAT=jsonl."
            ) from exc

        self.log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{prefix}_{stamp}.mcap"
        self.every_n = max(1, int(every_n))
        self._stream = self.path.open("wb")
        self._writer = Writer(self._stream)
        self._writer.start(profile="x-jsonschema", library="soridormi-runtime")

        state_schema_id = self._writer.register_schema(
            name="soridormi.RobotStateLog",
            encoding="jsonschema",
            data=json.dumps(_STATE_SCHEMA).encode("utf-8"),
        )
        command_schema_id = self._writer.register_schema(
            name="soridormi.MotorCommandLog",
            encoding="jsonschema",
            data=json.dumps(_COMMAND_SCHEMA).encode("utf-8"),
        )
        status_schema_id = self._writer.register_schema(
            name="soridormi.RuntimeStatus",
            encoding="jsonschema",
            data=json.dumps(_STATUS_SCHEMA).encode("utf-8"),
        )

        self._state_channel = self._writer.register_channel(
            topic="/soridormi/robot_state",
            message_encoding="json",
            schema_id=state_schema_id,
        )
        self._command_channel = self._writer.register_channel(
            topic="/soridormi/motor_command",
            message_encoding="json",
            schema_id=command_schema_id,
        )
        self._status_channel = self._writer.register_channel(
            topic="/soridormi/runtime_status",
            message_encoding="json",
            schema_id=status_schema_id,
        )
        self._closed = False
        self.write_status(mode=mode, backend=backend, message="runtime logger started")

    def write_status(self, *, mode: str, backend: str, message: str) -> None:
        timestamp_ns = now_ns()
        payload = {
            "type": "runtime_status",
            "time_wall_ns": timestamp_ns,
            "time_wall": timestamp_ns / 1_000_000_000.0,
            "mode": mode,
            "backend": backend,
            "message": message,
        }
        self._writer.add_message(
            channel_id=self._status_channel,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )

    def log_step(
        self,
        *,
        step_index: int,
        state: RobotState,
        command: MotorCommand,
        mode: str,
        backend: str,
    ) -> None:
        if step_index % self.every_n != 0:
            return

        timestamp_ns = now_ns()
        robot_time = float(state.time)
        state_payload = {
            "type": "robot_state",
            "step_index": step_index,
            "time_wall_ns": timestamp_ns,
            "time_wall": timestamp_ns / 1_000_000_000.0,
            "robot_time": robot_time,
            "mode": mode,
            "backend": backend,
            "state": model_to_json_dict(state),
        }
        command_payload = {
            "type": "motor_command",
            "step_index": step_index,
            "time_wall_ns": timestamp_ns,
            "time_wall": timestamp_ns / 1_000_000_000.0,
            "robot_time": robot_time,
            "mode": mode,
            "backend": backend,
            "command": model_to_json_dict(command),
        }

        self._writer.add_message(
            channel_id=self._state_channel,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            data=json.dumps(state_payload, separators=(",", ":")).encode("utf-8"),
        )
        self._writer.add_message(
            channel_id=self._command_channel,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            data=json.dumps(command_payload, separators=(",", ":")).encode("utf-8"),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.finish()
        finally:
            self._stream.close()
