# Soridormi

**Soridormi** is a sim-to-real humanoid robot development stack based on Open Duck Mini v2.

The project goal is to replace the original Raspberry Pi-style onboard runtime with an NVIDIA Jetson-class runtime, while keeping the same high-level robot API in simulation and on real hardware.

Project name note: the intended bronze dragon name is **Soridormi**.

## Whole-system role

Soridormi is the platform execution runtime paired with Chromie. Open Duck Mini
v2 body control, locomotion, safety, simulation, training/evaluation, and future
hardware remain its verified foundation. The approved target also places
platform-facing vocal, media, audio/sensor, and device execution behind
Soridormi.

Chromie is the robot **brain**, maintained in
`https://github.com/TimeTreker/chromie.git` on `main`. Chromie owns conversation,
memory, Goal meaning, response authorship, vocal mode, confirmation,
authorization, and user-level interaction orchestration.

The target boundary is:

```text
Chromie brain and Interaction Orchestrator
  -> immutable platform-neutral execution envelope
Soridormi Execution Runtime
  -> capability resources, preparation, timing, execution, cancellation, recovery, evidence
Soridormi Platform Provider
  -> MuJoCo OR physical robot OR desktop platform, devices, drivers, and safety
```

Chromie must not send raw joint actions, low-level 14D policy actions, audio-
device indexes, SDK objects, or platform-specific commands. Soridormi may reject
an unsupported or unsafe request, but it may not reinterpret Goal meaning,
rewrite authored content, change vocal mode, or widen confirmation.

Chromie uses one Cognitive Core with Social-Attention, Speaking, and Activity
semantic lanes. Singing and humming are Speaking outcomes; playing existing
music is Activity. Under the approved target, Soridormi executes body, vocal,
and media capabilities and may coordinate compatible members while retaining
provider-local resource and safety authority.

This is a target architecture, not a current capability claim. Today Soridormi
primarily executes body work, while Chromie still owns TTS/playback and much of
cross-provider coordination. See `docs/STATUS.md` before describing a migrated
capability as implemented.

## Design goals

```text
soridormi-runtime
  Platform-independent execution runtime.
  Runs stable capabilities, policy inference, provider-local resources,
  preparation, scheduling, cancellation, recovery, and normalized evidence.
  No MuJoCo, robot-SDK, or sound-device dependency.

soridormi-platform
  Exactly one active simulator, physical-robot, or desktop-platform provider.
  The current PC implementation is soridormi-sim / MuJoCo.
  Owns devices, drivers, sensors, controllers, calibration, and hardware safety.

soridormi-api
  Shared public execution envelope and private Platform Contract packages.
```

The repository's `soridormi_runtime` source package also contains optional MCP,
evaluation, packaging, and training support. The production runtime process and
runtime dependency set must still remain free of MuJoCo, desktop viewer, and
training-only imports.

In simulation:

```text
soridormi-runtime  <---- same robot API ---->  soridormi-sim / MuJoCo
```

On real robot:

```text
soridormi-runtime  <---- same robot API ---->  motors / IMU / encoders / power
```

The execution runtime should not need to know whether it talks to MuJoCo, a
physical robot, or a desktop audio/sensor platform.

For fresh-machine simulator bootstrap, use
[Soridormi Deployment](docs/SORIDORMI_DEPLOYMENT.md) and
`./scripts/deploy_soridormi.sh`.

## Target hardware

Primary robot deployment targets:

- NVIDIA Jetson Orin NX
- NVIDIA Jetson AGX Orin
- the matching JetPack/L4T/CUDA container base

Constrained compatibility target:

- NVIDIA Jetson Orin Nano-class hardware with an explicitly reduced runtime
  profile

Exploratory target:

- NVIDIA Jetson AGX Thor

Thor compatibility is useful, but it is not a prerequisite or the current
primary robot-compute target.

Development host:

- Ubuntu desktop PC
- NVIDIA GPU
- Docker + Docker Compose
- MuJoCo simulation

## Repository structure

```text
soridormi/
├── compose.sim.yaml             # PC simulation: runtime-dev + simulator
├── compose.orin.yaml            # Jetson AGX Orin runtime deployment template
├── compose.thor.yaml            # Jetson AGX Thor runtime deployment template
├── configs/
│   └── robots/open_duck_mini_v2.yaml # robot-specific MuJoCo/API mapping
├── docker/
│   ├── runtime/Dockerfile       # Runtime image, parameterized by BASE_IMAGE
│   └── simulator/Dockerfile     # PC MuJoCo/CUDA simulation image
├── scripts/
│   ├── add_submodules.sh
│   ├── setup_env.sh
│   ├── build_sim.sh
│   ├── enter_runtime_dev.sh
│   ├── enter_sim.sh
│   ├── run_sim_server.sh
│   └── run_runtime_loop.sh
├── src/
│   ├── soridormi_api/
│   ├── soridormi_runtime/
│   └── soridormi_sim/
├── tests/
└── workspace/
    └── upstream Open Duck repos as git submodules
```


### Docker container user

Both Docker images create and use the same non-root user:

```text
chromie
```

The username is controlled by `CONTAINER_USER=chromie` in `.env`. The Dockerfiles also handle base images where UID/GID 1000 already exists, so builds should not fail with `GID '1000' already exists`.

## MCP boundary for Chromie

Soridormi exposes its safe robot capability boundary from a dedicated
`soridormi-mcp` container. Chromie runs separately and connects over MCP
Streamable HTTP:

```bash
./scripts/run_mcp_server.sh
```

The default MCP service is intentionally dry-run only. For runtime-backed
simulation, start the simulator and then run:

```bash
./scripts/run_runtime_mcp_server.sh
```

The runtime adapter executes bounded plans through the existing runtime
robot/controller interfaces and supports preemptive stop, cancellation, and
emergency stop. It rejects hardware modes until `HardwareRobot` is implemented.
See [Soridormi MCP server](docs/SORIDORMI_MCP_SERVER.md) for the deployment
contract and Chromie endpoint configuration. The concurrency contracts are
[Chromie cognitive concurrency](docs/CHROMIE_COGNITIVE_CONCURRENCY_MODEL.md)
and [Soridormi body concurrency](docs/SORIDORMI_BODY_CONCURRENCY.md).

## Quick start on PC

### 1. Clone this repo

```bash
git clone https://github.com/TimeTreker/soridormi.git
cd soridormi
```

If you started from the zip file, unzip it, then initialize git:

```bash
cd soridormi
git init
```

### 2. Add upstream submodules

```bash
./scripts/add_submodules.sh
```

This adds:

```text
workspace/Open_Duck_Mini
workspace/Open_Duck_Mini_Runtime
workspace/Open_Duck_Playground
```

### 3. Create local `.env`

```bash
./scripts/setup_env.sh
```

`scripts/setup_env.sh` is the single source of truth for Docker image
references. It writes both the Soridormi output-image names and their base
images to `.env`; all Compose files consume those variables without carrying
their own fallback tags.

To customize an image, edit the generated `.env`, or pass an override when
regenerating it, for example:

```bash
SORIDORMI_SIM_IMAGE=soridormi-sim:local ./scripts/setup_env.sh
```

### 4. Build PC simulation stack

```bash
./scripts/build_sim.sh
```

This builds two logical images:

```text
soridormi-runtime:cuda13.1-cudnn-dev  GPU-capable runtime-dev image for PC simulation
soridormi-sim:cuda13.1-cudnn          heavy MuJoCo/CUDA/cuDNN simulation image
```

The real robot runtime uses the same runtime Dockerfile but a Jetson-specific base image.

By default, the PC-side runtime-dev and simulator images use:

```text
nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
```

The runtime-dev container is GPU-enabled so ONNX Runtime can prefer `CUDAExecutionProvider` during PC simulation. If `onnxruntime-gpu` is not compatible with your local driver/CUDA stack, set `RUNTIME_DEV_EXTRA=runtime` and `SORIDORMI_ONNXRUNTIME_GPU=0` in `.env`, then rebuild with `./scripts/build_sim.sh`.

Python dependencies are installed during `docker compose build`. Source code is still bind-mounted from `./src`, so normal code edits do not require rebuilding the image. Rebuild only when `pyproject.toml`, a Dockerfile, system dependencies, or base images change.

### Robot config

The MuJoCo backend is config-driven. Robot/model-specific details live in:

```text
configs/robots/open_duck_mini_v2.yaml
```

The default `.env` still allows the safe fake backend for low-level API development:

```text
SORIDORMI_ROBOT_CONFIG=/app/configs/robots/open_duck_mini_v2.yaml
SORIDORMI_SIM_BACKEND=fake
```

For locomotion validation, start the real MuJoCo backend explicitly. The default functional-test command is headless/no-viewer:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

To watch MuJoCo visually, enable the passive viewer explicitly:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

For longer walking runs where the duck may leave the initial frame, enable the viewer follow camera:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

To switch robot models later, add another YAML file under `configs/robots/` and change only `SORIDORMI_ROBOT_CONFIG`. The backend code should not hardcode robot-specific actuator names or model paths. Viewer settings also live in the robot config under `viewer:`.

### 5. Enter simulator container

```bash
./scripts/enter_sim.sh
```

Inside the container:

```bash
whoami
sim
python - <<'PY'
import soridormi_api, soridormi_sim
print("Soridormi simulator imports OK")
PY
```

The simulator Python environment is now built into the Docker image at `/opt/venvs/sim`, so `bootstrap_simulator` is no longer required for normal daily use.

### 6. Start simulator API server

You can start it directly from the host:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Or from inside the simulator container:

```bash
sim
python -m soridormi_sim.mujoco_server
```

The host wrapper defaults to the MuJoCo backend for locomotion validation. Use `--backend fake` only for low-level API development that does not need physics.

### 7. Start runtime container in another terminal

You can start the runtime loop directly from the host:

```bash
./scripts/run_runtime_loop.sh
```

Or enter the runtime-dev container:

```bash
./scripts/enter_runtime_dev.sh
```

Inside the runtime-dev container:

```bash
runtime
python -m soridormi_runtime.main
```

The runtime Python environment is built into the Docker image at `/opt/venvs/runtime`. The runtime will connect to the simulator using the same API shape that the real hardware backend should implement.

## Common commands

Build simulation images:

```bash
./scripts/build_sim.sh
```

Enter runtime-dev container:

```bash
./scripts/enter_runtime_dev.sh
```

Enter simulator container:

```bash
./scripts/enter_sim.sh
```

Run simulator server from host through Compose:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Run runtime loop from host through Compose:

```bash
./scripts/run_runtime_loop.sh
```

### Simulator ownership for live tools

Soridormi live MuJoCo commands use two patterns:

- External-sim evaluation/runtime tools require a separately running simulator
  server. Start it with `./scripts/run_sim_server.sh --backend mujoco --profile
  open_duck_forward --viewer --follow-camera`, then run tools such as
  `evaluate_scenario_rollout.sh`, `evaluate_scenario_suite.sh`,
  `run_skill_in_sim.sh`, or scripted social skill commands in another terminal.
- Random teacher dataset collection owns its collection lifecycle. Do not start
  a second `run_sim_server.sh` for `collect_random_teacher_dataset.sh`; pass
  `--viewer` to the collector itself when visual inspection is needed.

For the scenario-aware context data pipeline, see
`docs/SORIDORMI_CONTEXT_DATA_PIPELINE.md`.

For the curated documentation map, see `docs/README.md`. Keep durable project
contracts and runbooks in `docs/`; generated reports should go under
`artifacts/` and stay out of git.

For verified current capability and blockers, see `docs/STATUS.md`. For the
durable target and gated capability sequence, see
`docs/SORIDORMI_TARGET_AND_ROADMAP.md` and
`docs/SORIDORMI_EXECUTION_ROADMAP.md`.

## Runtime backend selection

The runtime uses `SORIDORMI_BACKEND`:

```bash
SORIDORMI_BACKEND=sim
```

or later:

```bash
SORIDORMI_BACKEND=hardware
```

Your controller should keep calling the same interface:

```python
state = robot.read_state()
robot.send_motor_command(command)
```

## Jetson notes

For Jetson AGX Orin:

```bash
docker compose -f compose.orin.yaml build runtime
```

For Jetson AGX Thor:

```bash
docker compose -f compose.thor.yaml build runtime
```

The exact Jetson base image depends on your installed JetPack/L4T version. Edit `.env` after `scripts/setup_env.sh` if needed:

```env
RUNTIME_DEV_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
ORIN_RUNTIME_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
THOR_RUNTIME_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
SIM_BASE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
SORIDORMI_ONNXRUNTIME_GPU=1
SORIDORMI_USE_CUDA_PROVIDER=1
```

On Jetson, build the runtime image on the Jetson itself unless you already have a cross-build setup.

For a real Jetson deployment, verify the base image against the JetPack/L4T release installed on the board. The CUDA 13.1 cuDNN image is a good PC development default, but Jetson images may require NVIDIA's Jetson/L4T container images instead of normal desktop CUDA images.

## Important warning

Do not try to make the simulator image identical to the robot image. The simulator PC is usually x86_64 with a desktop NVIDIA GPU, while the robot computer is ARM64 Jetson hardware. Instead, keep the **API identical** and keep the runtime code portable.

## Upstream repositories

This project uses upstream repos as submodules rather than copying their code:

- `apirrone/Open_Duck_Mini`, branch `v2`
- `apirrone/Open_Duck_Mini_Runtime`, branch `v2`
- `apirrone/Open_Duck_Playground`, default branch

## License

This repository skeleton is MIT licensed. Upstream repositories keep their own licenses.
