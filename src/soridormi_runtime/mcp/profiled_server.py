from __future__ import annotations

import os
import sys

from soridormi_runtime.policy_profiles import PolicyProfile


def main() -> None:
    profile_name = os.environ.get("SORIDORMI_POLICY_PROFILE", "").strip()
    if not profile_name:
        raise RuntimeError("SORIDORMI_POLICY_PROFILE must be set for mcp-runtime")

    profile = PolicyProfile.load(profile_name)
    resolved_env = os.environ.copy()
    resolved_env.update(profile.env())

    print(
        "Soridormi MCP runtime profile resolved: "
        f"profile={resolved_env['SORIDORMI_POLICY_PROFILE']} "
        f"file={resolved_env['SORIDORMI_POLICY_PROFILE_FILE']} "
        f"action_scale={resolved_env['SORIDORMI_ACTION_SCALE']}",
        flush=True,
    )

    argv = [
        sys.executable,
        "-m",
        "soridormi_runtime.mcp.http_server",
        "--adapter",
        "runtime",
    ]
    os.execvpe(sys.executable, argv, resolved_env)


if __name__ == "__main__":
    main()
