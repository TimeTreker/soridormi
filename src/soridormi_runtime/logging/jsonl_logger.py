from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from soridormi_api import MotorCommand, RobotState

from .base import default_log_dir, model_to_json_dict, now_ns


class JsonlRuntimeLogger:
    """Simple JSONL runtime logger.

    This is useful for quick debugging and tests. MCAP is preferred for robotics
    logging, but JSONL remains convenient for grep/diff/manual inspection.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        every_n: int = 1,
        prefix: str = "runtime",
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{prefix}_{stamp}.jsonl"
        self.every_n = max(1, int(every_n))
        self._stream = self.path.open("w", encoding="utf-8")

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
        payload = {
            "type": "runtime_step",
            "step_index": step_index,
            "time_wall_ns": timestamp_ns,
            "time_wall": timestamp_ns / 1_000_000_000.0,
            "robot_time": float(state.time),
            "mode": mode,
            "backend": backend,
            "state": model_to_json_dict(state),
            "command": model_to_json_dict(command),
        }
        self._stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
