from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = (ROOT / "scripts/start_soridormi_mujoco.sh").read_text(encoding="utf-8")
RUN = (ROOT / "scripts/run_sim_server.sh").read_text(encoding="utf-8")
HELPER = (ROOT / "scripts/x11_access.sh").read_text(encoding="utf-8")


class X11LauncherContractTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        subprocess.run(
            [
                "bash",
                "-n",
                str(ROOT / "scripts/x11_access.sh"),
                str(ROOT / "scripts/run_sim_server.sh"),
                str(ROOT / "scripts/start_soridormi_mujoco.sh"),
            ],
            check=True,
        )

    def test_broad_unchecked_docker_xhost_rule_is_removed(self) -> None:
        combined = START + RUN + HELPER
        self.assertNotIn("xhost +local:docker", combined)
        self.assertNotIn("xhost -local:docker", combined)
        self.assertIn("SI:localuser:", HELPER)

    def test_start_preflights_before_stopping_existing_simulator(self) -> None:
        self.assertIn("source ./scripts/x11_access.sh", START)
        self.assertLess(
            START.index('soridormi_x11_preflight "$VIEWER"'),
            START.index("\n    stop_existing_sim_containers\n"),
        )

    def test_run_script_owns_authorization_lifecycle(self) -> None:
        self.assertIn('soridormi_x11_acquire "$VIEWER_ENABLED"', RUN)
        self.assertIn("trap cleanup_x11 EXIT", RUN)
        self.assertIn('"${SORIDORMI_X11_DOCKER_ARGS[@]}"', RUN)
        self.assertIn("soridormi_x11_cleanup", RUN)

    def test_cleanup_only_removes_state_added_by_launcher(self) -> None:
        self.assertIn('SORIDORMI_X11_XHOST_RULE_ADDED=1', HELPER)
        self.assertIn('[ "$SORIDORMI_X11_XHOST_RULE_ADDED" = "1" ]', HELPER)
        self.assertIn('xhost "-${rule}"', HELPER)
        self.assertIn('rm -rf "$SORIDORMI_X11_TEMP_DIR"', HELPER)

    def test_cookie_authorization_is_temporary_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fake_bin = temp / "bin"
            runtime_dir = temp / "runtime"
            fake_bin.mkdir()
            runtime_dir.mkdir()
            xauth = fake_bin / "xauth"
            xauth.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "nlist" ]; then
  printf '%s\n' '0100 0000 4d49542d4d414749432d434f4f4b49452d31 00112233445566778899aabbccddeeff'
  exit 0
fi
if [ "${1:-}" = "-f" ] && [ "${3:-}" = "nmerge" ]; then
  cat > "$2"
  exit 0
fi
if [ "${1:-}" = "-f" ] && [ "${3:-}" = "nlist" ]; then
  cat "$2"
  exit 0
fi
exit 1
""",
                encoding="utf-8",
            )
            xauth.chmod(0o755)
            script = ROOT / "scripts/x11_access.sh"
            command = "\n".join(
                [
                    "set -euo pipefail",
                    f"source {script}",
                    "soridormi_x11_acquire 1",
                    '[ "$SORIDORMI_X11_ACQUIRED" = 1 ]',
                    '[ "${#SORIDORMI_X11_DOCKER_ARGS[@]}" -eq 6 ]',
                    'auth_file="$SORIDORMI_X11_AUTH_FILE"',
                    '[ -s "$auth_file" ]',
                    "soridormi_x11_cleanup 0",
                    '[ ! -e "$auth_file" ]',
                    '[ "$SORIDORMI_X11_ACQUIRED" = 0 ]',
                ]
            )
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "DISPLAY": ":99",
                "XDG_RUNTIME_DIR": str(runtime_dir),
                "XAUTHORITY": "",
            }
            subprocess.run(["bash", "-c", command], check=True, env=env)

    def test_xhost_fallback_removes_only_the_rule_it_adds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fake_bin = temp / "bin"
            fake_bin.mkdir()
            state = temp / "xhost-state"
            log = temp / "xhost-log"
            xauth = fake_bin / "xauth"
            xauth.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            xauth.chmod(0o755)
            xhost = fake_bin / "xhost"
            xhost.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
state=${FAKE_XHOST_STATE:?}
log=${FAKE_XHOST_LOG:?}
if [ $# -eq 0 ]; then
  printf '%s\n' 'access control enabled'
  cat "$state" 2>/dev/null || true
  exit 0
fi
printf '%s\n' "$1" >> "$log"
case "$1" in
  +SI:localuser:*) printf '%s\n' "${1#+}" >> "$state" ;;
  -SI:localuser:*) grep -Fvx "${1#-}" "$state" > "$state.next" || true; mv "$state.next" "$state" ;;
esac
""",
                encoding="utf-8",
            )
            xhost.chmod(0o755)
            script = ROOT / "scripts/x11_access.sh"
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                "DISPLAY": ":99",
                "XAUTHORITY": "",
                "SORIDORMI_X11_LOCAL_USER": "tester",
                "FAKE_XHOST_STATE": str(state),
                "FAKE_XHOST_LOG": str(log),
            }

            state.write_text("", encoding="utf-8")
            command = "\n".join(
                [
                    "set -euo pipefail",
                    f"source {script}",
                    "soridormi_x11_acquire 1",
                    '[ "$SORIDORMI_X11_XHOST_RULE_ADDED" = 1 ]',
                    "soridormi_x11_cleanup 0",
                ]
            )
            subprocess.run(["bash", "-c", command], check=True, env=env)
            actions = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                actions,
                ["+SI:localuser:tester", "-SI:localuser:tester"],
            )

            state.write_text("  SI:localuser:tester  \n", encoding="utf-8")
            log.write_text("", encoding="utf-8")
            reuse_command = "\n".join(
                [
                    "set -euo pipefail",
                    f"source {script}",
                    "soridormi_x11_acquire 1",
                    '[ "$SORIDORMI_X11_XHOST_RULE_ADDED" = 0 ]',
                    "soridormi_x11_cleanup 0",
                ]
            )
            subprocess.run(["bash", "-c", reuse_command], check=True, env=env)
            self.assertEqual(log.read_text(encoding="utf-8"), "")

    def test_headless_acquire_is_a_noop(self) -> None:
        script = ROOT / "scripts/x11_access.sh"
        command = "\n".join(
            [
                "set -euo pipefail",
                f"source {script}",
                "soridormi_x11_acquire 0",
                '[ "${#SORIDORMI_X11_DOCKER_ARGS[@]}" -eq 0 ]',
                '[ "$SORIDORMI_X11_ACQUIRED" = 0 ]',
            ]
        )
        subprocess.run(["bash", "-c", command], check=True)


if __name__ == "__main__":
    unittest.main()
