from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _occupied_port() -> tuple[subprocess.Popen[str], int]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time,sys; s=socket.socket(); s.bind(('127.0.0.1',int(sys.argv[1]))); "
            "s.listen(); print('ready',flush=True); time.sleep(30)",
            str(port),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    return process, port


def _run_wrapper(tmp_path: Path, *, port: int, owner_pid: int, match: bool, runtime_busy: bool) -> subprocess.CompletedProcess[str]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/run_scenario.sh", scripts / "run_scenario.sh")
    (scripts / "x11_access.sh").write_text(
        "SORIDORMI_X11_DOCKER_ARGS=()\n"
        "soridormi_x11_acquire() { :; }\n"
        "soridormi_x11_cleanup() { :; }\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("", encoding="utf-8")
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "log=Path(os.environ['FAKE_DOCKER_LOG'])\n"
        "with log.open('a') as out: out.write(' '.join(args[:2])+'\\n')\n"
        "if args[0]=='ps':\n"
        "    if os.environ['FAKE_MATCH']=='1': print('old-scenario')\n"
        "elif args[0]=='inspect':\n"
        "    target=args[-1]\n"
        "    fmt=args[2]\n"
        "    if target=='soridormi-runtime-mcp':\n"
        "        if 'State.Running' in fmt: print('true' if os.environ['FAKE_RUNTIME_BUSY']=='1' else 'false')\n"
        "        else: print('SIM_PORT='+os.environ['SIM_PORT'])\n"
        "    elif 'json' in fmt: print('[\"python -m scenario_runner\"]')\n"
        "    else: print('SIM_PORT='+os.environ['SIM_PORT'])\n"
        "elif args[0]=='exec':\n"
        "    sys.exit(1 if os.environ['FAKE_RUNTIME_BUSY']=='1' else 0)\n"
        "elif args[0]=='stop':\n"
        "    os.kill(int(os.environ['FAKE_OWNER_PID']), signal.SIGTERM)\n"
        "elif args[0]=='compose':\n"
        "    pass\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        SIM_PORT=str(port),
        FAKE_OWNER_PID=str(owner_pid),
        FAKE_MATCH="1" if match else "0",
        FAKE_RUNTIME_BUSY="1" if runtime_busy else "0",
        FAKE_DOCKER_LOG=str(tmp_path / "docker.log"),
    )
    return subprocess.run(
        ["bash", str(scripts / "run_scenario.sh"), "--no-viewer", "--scenario", str(scenario)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_run_scenario_replaces_only_matching_idle_scenario(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(tmp_path, port=port, owner_pid=owner.pid, match=True, runtime_busy=False)
        assert result.returncode == 0, result.stderr
        assert "Replacing prior scenario container" in result.stdout
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert calls.index("stop --timeout") < calls.index("compose -f")
        assert calls.index("rm old-scenario") < calls.index("compose -f")
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_refuses_unknown_port_owner(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(tmp_path, port=port, owner_pid=owner.pid, match=False, runtime_busy=False)
        assert result.returncode != 0
        assert "expected one scenario container" in result.stderr
        assert "stop --timeout" not in (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_refuses_active_runtime(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(tmp_path, port=port, owner_pid=owner.pid, match=True, runtime_busy=True)
        assert result.returncode != 0
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert "exec -i" in calls
        assert "stop --timeout" not in calls
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)
