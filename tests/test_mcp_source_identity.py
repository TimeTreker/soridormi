from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from soridormi_runtime.mcp.local_tools import SoridormiLocalToolService
from soridormi_runtime.mcp.manifest import build_soridormi_capability_bundle
from soridormi_runtime.mcp.source_identity import current_source_revision


class McpSourceIdentityTests(unittest.TestCase):
    def test_source_revision_is_read_from_launcher_environment(self) -> None:
        revision = "a" * 40
        with patch.dict(
            os.environ,
            {"SORIDORMI_SOURCE_REVISION": revision},
            clear=False,
        ):
            self.assertEqual(current_source_revision(), revision)
            status = SoridormiLocalToolService().get_status()

        self.assertEqual(status["source_revision"], revision)

    def test_status_omits_revision_when_launcher_did_not_supply_one(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SORIDORMI_SOURCE_REVISION", None)
            status = SoridormiLocalToolService().get_status()

        self.assertNotIn("source_revision", status)

    def test_capability_manifest_declares_source_revision(self) -> None:
        bundle = build_soridormi_capability_bundle(mode="sim")
        status_tool = next(
            tool
            for agent in bundle.agents
            for tool in agent.tools
            if tool.name == "soridormi.robot.get_status"
        )
        properties = status_tool.output_schema["properties"]

        self.assertIn("source_revision", properties)
        self.assertEqual(properties["source_revision"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
