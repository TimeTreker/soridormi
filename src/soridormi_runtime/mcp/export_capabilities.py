from __future__ import annotations

import argparse
import json

from .dag_contract import build_soridormi_dag_contract
from .manifest import build_soridormi_capability_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Soridormi MCP-ready capability manifest JSON.")
    parser.add_argument("--mode", default="sim", choices=["sim", "hardware_shadow", "hardware_dry_run", "hardware"], help="Runtime mode to include in manifest status details.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    parser.add_argument("--dag-contract-only", action="store_true", help="Emit only Soridormi task-graph integration hints.")
    args = parser.parse_args()

    payload = build_soridormi_dag_contract(mode=args.mode) if args.dag_contract_only else build_soridormi_capability_bundle(mode=args.mode).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
