from __future__ import annotations

import os

SOURCE_REVISION_ENV = "SORIDORMI_SOURCE_REVISION"


def current_source_revision() -> str | None:
    """Return launcher-provided source identity without inspecting Git at runtime."""

    value = os.environ.get(SOURCE_REVISION_ENV, "").strip()
    return value or None


__all__ = ["SOURCE_REVISION_ENV", "current_source_revision"]
