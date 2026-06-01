"""MCP-ready Soridormi capability manifest exports."""

from .dag_contract import build_soridormi_dag_contract
from .manifest import CapabilityBundle, build_soridormi_capability_bundle

__all__ = ["CapabilityBundle", "build_soridormi_capability_bundle", "build_soridormi_dag_contract"]
