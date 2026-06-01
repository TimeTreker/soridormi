from __future__ import annotations

import argparse
import json
import sys

from .local_tools import SoridormiLocalToolService


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a Soridormi local MCP-style tool in dry-run mode.")
    parser.add_argument("tool", help="Tool name, for example soridormi.robot.get_status")
    parser.add_argument("--args-json", default="{}", help="JSON object passed as tool arguments.")
    parser.add_argument("--mode", default="sim", choices=["sim", "hardware_dry_run", "hardware"], help="Runtime mode reported by status tools.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    try:
        tool_args = json.loads(args.args_json)
        if not isinstance(tool_args, dict):
            raise ValueError("--args-json must decode to an object")
        service = SoridormiLocalToolService(mode=args.mode)
        payload = {"status": "success", "output": service.call_tool(args.tool, tool_args)}
    except Exception as exc:
        payload = {"status": "failed_fatal", "error": str(exc), "output": {}}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2), file=sys.stdout)
        raise SystemExit(1)

    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2))


if __name__ == "__main__":
    main()
