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


def _run_wrapper(
    tmp_path: Path,
    *,
    port: int,
    owner_pid: int,
    owner_kind: str,
    runtime_running: bool = False,
    runtime_busy: bool = False,
    foreign_runtime: bool = False,
    build: bool = False,
    build_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
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
        "import os, signal, socket, sys, time\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "log=Path(os.environ['FAKE_DOCKER_LOG'])\n"
        "with log.open('a') as out: out.write(' '.join(args)+'\\n')\n"
        "if args[0]=='ps':\n"
        "    if os.environ['FAKE_OWNER_KIND']!='none': print('old-'+os.environ['FAKE_OWNER_KIND'])\n"
        "elif args[0]=='inspect':\n"
        "    target=args[-1]\n"
        "    fmt=args[2]\n"
        "    if target=='soridormi-runtime-mcp':\n"
        "        if 'State.Running' in fmt: print('true' if os.environ['FAKE_RUNTIME_RUNNING']=='1' else 'false')\n"
        "        elif 'project.working_dir' in fmt: print('/another/checkout' if os.environ['FAKE_FOREIGN_RUNTIME']=='1' else os.getcwd())\n"
        "        else: print('SIM_PORT='+os.environ['SIM_PORT'])\n"
        "    elif 'json' in fmt:\n"
        "        kind=os.environ['FAKE_OWNER_KIND']\n"
        "        module={'scenario':'scenario_runner','standard':'soridormi_sim.mujoco_server'}.get(kind,'unrelated')\n"
        "        print('[\"python -m '+module+'\"]')\n"
        "    else: print('SIM_PORT='+os.environ['SIM_PORT'])\n"
        "elif args[0]=='exec':\n"
        "    sys.exit(1 if os.environ['FAKE_RUNTIME_BUSY']=='1' else 0)\n"
        "elif args[0]=='stop':\n"
        "    os.kill(int(os.environ['FAKE_OWNER_PID']), signal.SIGTERM)\n"
        "elif args[0]=='compose':\n"
        "    if 'build' in args and os.environ['FAKE_BUILD_FAILURE']=='1': sys.exit(1)\n"
        "    if 'run' in args:\n"
        "        with socket.socket() as listener:\n"
        "            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "            listener.bind(('127.0.0.1', int(os.environ['SIM_PORT'])))\n"
        "            listener.listen()\n"
        "            time.sleep(1)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        SIM_PORT=str(port),
        FAKE_OWNER_PID=str(owner_pid),
        FAKE_OWNER_KIND=owner_kind,
        FAKE_RUNTIME_RUNNING="1" if runtime_running else "0",
        FAKE_RUNTIME_BUSY="1" if runtime_busy else "0",
        FAKE_FOREIGN_RUNTIME="1" if foreign_runtime else "0",
        FAKE_BUILD_FAILURE="1" if build_failure else "0",
        FAKE_DOCKER_LOG=str(tmp_path / "docker.log"),
    )
    return subprocess.run(
        ["bash", str(scripts / "run_scenario.sh"), "--no-viewer", "--scenario", str(scenario), *(["--build"] if build else [])],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_run_scenario_replaces_matching_idle_scenario(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(tmp_path, port=port, owner_pid=owner.pid, owner_kind="scenario")
        assert result.returncode == 0, result.stderr
        assert "Replacing prior simulator container" in result.stdout
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert calls.index("stop --timeout") < calls.index("compose -f")
        assert calls.index("rm old-scenario") < calls.index("compose -f")
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_replaces_standard_sim_and_reconnects_idle_runtime(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(
            tmp_path, port=port, owner_pid=owner.pid,
            owner_kind="standard", runtime_running=True,
        )
        assert result.returncode == 0, result.stderr
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert calls.index("exec -i soridormi-runtime-mcp") < calls.index("compose -f compose.sim.yaml --profile mcp-runtime stop mcp-runtime")
        assert calls.index("compose -f compose.sim.yaml --profile mcp-runtime stop mcp-runtime") < calls.index("stop --timeout")
        assert "rm old-standard" in calls
        assert "up -d --no-build --pull never mcp-runtime" in calls
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_refuses_unknown_port_owner(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(tmp_path, port=port, owner_pid=owner.pid, owner_kind="unknown")
        assert result.returncode != 0
        assert "expected one Soridormi simulator container" in result.stderr
        assert "stop --timeout" not in (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_refuses_active_runtime(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(
            tmp_path, port=port, owner_pid=owner.pid,
            owner_kind="standard", runtime_running=True, runtime_busy=True,
        )
        assert result.returncode != 0
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert "exec -i" in calls
        assert "stop --timeout" not in calls
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_refuses_foreign_runtime_on_same_port(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(
            tmp_path, port=port, owner_pid=owner.pid,
            owner_kind="standard", runtime_running=True, foreign_runtime=True,
        )
        assert result.returncode != 0
        assert "belongs to another checkout" in result.stderr
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert "stop --timeout" not in calls
        assert "compose -f compose.sim.yaml --profile mcp-runtime stop mcp-runtime" not in calls
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_builds_before_replacing_sim(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(
            tmp_path, port=port, owner_pid=owner.pid,
            owner_kind="standard", build=True,
        )
        assert result.returncode == 0, result.stderr
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert calls.index("compose -f compose.sim.yaml --profile mcp-runtime build runtime sim mcp-runtime") < calls.index("stop --timeout")
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_failed_build_preserves_running_sim(tmp_path: Path) -> None:
    owner, port = _occupied_port()
    try:
        result = _run_wrapper(
            tmp_path, port=port, owner_pid=owner.pid,
            owner_kind="standard", build=True, build_failure=True,
        )
        assert result.returncode != 0
        calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
        assert "build runtime sim mcp-runtime" in calls
        assert "stop --timeout" not in calls
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_run_scenario_declares_runtime_mcp_start_after_simulator_readiness() -> None:
    script = (ROOT / "scripts/run_scenario.sh").read_text(encoding="utf-8")
    readiness = 'if ! tcp_port_active "$sim_port"; then'
    mcp_start = 'up -d --no-build --pull never mcp-runtime'
    assert readiness in script
    assert mcp_start in script
    assert script.index(readiness, script.index("scenario_pid=$!")) < script.index(mcp_start)


def test_run_scenario_validate_exits_before_runtime_mcp_start() -> None:
    script = (ROOT / "scripts/run_scenario.sh").read_text(encoding="utf-8")
    validate_branch = 'if [ "$VALIDATE" = "1" ]; then\n  exec docker compose'
    mcp_start = 'up -d --no-build --pull never mcp-runtime'
    assert validate_branch in script
    assert script.index(validate_branch) < script.index(mcp_start)
