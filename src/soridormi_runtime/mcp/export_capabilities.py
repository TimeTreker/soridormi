from __future__ import annotations

import argparse
import json

from .manifest import build_soridormi_capability_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Soridormi MCP-ready capability manifest JSON.")
    parser.add_argument("--mode", default="sim", choices=["sim", "hardware_dry_run", "hardware"], help="Runtime mode to include in manifest status details.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    bundle = build_soridormi_capability_bundle(mode=args.mode)
    print(json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
