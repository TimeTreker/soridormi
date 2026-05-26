from __future__ import annotations

import os

from .base import NullRuntimeLogger, RuntimeLogger, default_log_dir, env_bool, env_int
from .jsonl_logger import JsonlRuntimeLogger


def make_runtime_logger_from_env(*, mode: str, backend: str) -> RuntimeLogger:
    if not env_bool("SORIDORMI_RUNTIME_LOG", default=False):
        return NullRuntimeLogger()

    log_format = os.environ.get("SORIDORMI_RUNTIME_LOG_FORMAT", "mcap").strip().lower()
    log_dir = default_log_dir()
    every_n = env_int("SORIDORMI_RUNTIME_LOG_EVERY_N", 1)
    prefix = os.environ.get("SORIDORMI_RUNTIME_LOG_PREFIX", "runtime").strip() or "runtime"

    if log_format == "jsonl":
        logger = JsonlRuntimeLogger(log_dir=log_dir, every_n=every_n, prefix=prefix)
        print(f"Soridormi JSONL runtime logger: {logger.path}")
        return logger

    if log_format == "mcap":
        from .mcap_logger import McapRuntimeLogger

        logger = McapRuntimeLogger(log_dir=log_dir, every_n=every_n, prefix=prefix, mode=mode, backend=backend)
        print(f"Soridormi MCAP runtime logger: {logger.path}")
        return logger

    raise ValueError(
        f"Unknown SORIDORMI_RUNTIME_LOG_FORMAT={log_format!r}. Use 'mcap' or 'jsonl'."
    )
