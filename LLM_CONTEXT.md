# LLM_CONTEXT.md

This file is a compact handoff for starting a new LLM session on Soridormi.

## One-paragraph project summary

Soridormi is a sim-to-real humanoid robot stack for Open Duck Mini v2. It separates runtime, simulator, and shared API so the same policy runtime can talk to MuJoCo now and real robot hardware later. The current Soridormi-side objective is command-conditioned free walking in MuJoCo first: stop, stand, walk forward, turn, curve, tolerate conservative command switches, and compare teacher/candidate policies before any hardware walking.

## Current repository assumptions

- GitHub repo: `https://github.com/TimeTreker/soridormi.git`
- Branch: `main`
- Main containers:
  - `soridormi-runtime`: policy/runtime/API client, no MuJoCo dependency.
  - `soridormi-sim`: MuJoCo/API server.
  - `soridormi-api`: shared types and API messages.
- Upstream repos are expected under `workspace/`:
  - `Open_Duck_Mini`
  - `Open_Duck_Mini_Runtime`
  - `Open_Duck_Playground`

## Latest known direction

Do not move directly to hardware walking, MCP execution, or Chromie orchestration as the next Soridormi milestone. The next Soridormi milestone is the M6 command-conditioned free-walk gate in MuJoCo.

Read these docs first:

```text
docs/SORIDORMI_FREE_WALK_PLAN.md
docs/M6_SIM_STATUS.md
docs/M6_SIM_TRAINING_LOOP.md
docs/PROJECT_STATUS_AFTER_M6.md
docs/PATCH_DELIVERY_AND_VALIDATION.md
```

## Current practical focus

Soridormi should prioritize:

```text
M6A: commanded free-walk evaluation in MuJoCo
M6B: command-distribution teacher data collection
M6C: neural BC over command-grid/random-command data
M6D: teacher-vs-candidate closed-loop comparison
M6E: residual policy improvement if BC/evaluation are reliable
M6F: sim acceptance gate
M7: hardware read-only / dry-run bridge only after sim acceptance
M8: Chromie/MCP/LLM orchestration later
```

The core walking target is bounded high-level command control:

```text
vx: forward/backward velocity command
vy: lateral velocity command
yaw: turn-rate command
stop / cancel / emergency stop
```

"Freely" does not mean raw torque control or unbounded motor control. It means walking within the trained command envelope while preserving joint limits, fall detection, runtime limits, and rollout acceptance gates.

## Architecture boundary

Soridormi owns robot capability and safety:

```text
simulation
policy runtime
training/evaluation
action mapping
RobotState / MotorCommand contract
walking safety limits
future hardware backend
```

Chromie can later own user-facing orchestration:

```text
LLM routing
ASR/TTS
user confirmation
global MCP registry
multi-agent DAG planning
```

Do not let Chromie/MCP work imply that Soridormi's walking capability improved unless there is MuJoCo rollout evidence.

## M6A free-walk evaluation entrypoint

The conservative M6A fixed-command evaluation suite is:

```text
configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

Validate it without MuJoCo:

```bash
PYTHONPATH=src python -m soridormi_runtime.free_walk_eval --suite configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

This host-side static validator must not require PyYAML to be installed. `free_walk_eval` should keep a small fallback parser for the conservative M6A suite so basic validation works before the full runtime container or editable Python environment is rebuilt.

Run the generated teacher-vs-candidate free-walk grid with:

```bash
./scripts/run_free_walk_eval.sh neural_bc_teacher_grid --dry-run --force
```

Remove `--dry-run` only after the MuJoCo sim server is running.

## Patch delivery rule

The user prefers plain `.patch` files and usually downloads them to:

```bash
~/Downloads
```

Every patch response must include both:

```text
1. Patch integrity check: git apply --check ~/Downloads/<patch>.patch
2. Functional validation: commands that prove the behavior/docs/interface after applying
```

For docs-only patches, functional validation still means checking expected files/phrases and Markdown fences. For code patches, run relevant tests, compile checks, and CLI smoke tests. For sim/training patches, give both local/unit validation and live MuJoCo validation commands. Be explicit about anything not run.

## New-session prompt

A good prompt to start the next session:

> Please read `LLM_CONTEXT.md`, `docs/SORIDORMI_FREE_WALK_PLAN.md`, and `docs/PATCH_DELIVERY_AND_VALIDATION.md` first. We are focused on Soridormi only. The goal is Open Duck Mini v2 command-conditioned free walking in MuJoCo before hardware or Chromie/MCP orchestration. Continue with the next M6 free-walk simulation/evaluation task. If you provide a patch, make it a plain git patch and include both `git apply --check ~/Downloads/<patch>.patch` and functional validation commands.

## Simulator validation command policy

When giving Soridormi functional validation commands that require the simulator, start the simulator explicitly with the MuJoCo backend. The default should be headless/no-viewer:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

Also provide the viewer-enabled variant when a visual test is useful:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

If the robot may walk out of the initial frame, include the follow-camera variant:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Do not rely on an implicit simulator backend in future instructions.

### M6B random teacher data collection

Soridormi now includes an M6B random teacher dataset collector for command-conditioned free walking:

```bash
./scripts/collect_random_teacher_dataset.sh \
  --profile open_duck_forward \
  --output data/teacher_random_walk/dataset.jsonl \
  --episodes 100 \
  --steps-per-episode 800 \
  --vx-range -0.03,0.15 \
  --vy-range -0.03,0.03 \
  --yaw-range -0.20,0.20 \
  --command-hold-steps 80,250 \
  --backend mujoco \
  --no-viewer
```

Negative range values are valid in either shell style, so both `--vx-range -0.03,0.15` and `--vx-range=-0.03,0.15` are supported.

Always provide a MuJoCo sim server command for live validation:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

The random collector changes `vx/vy/yaw` several times inside each episode and records command segment metadata. It is for walking/turning/stopping command transitions. Do not claim it teaches sit-down or stand-up unless a separate pose-transition teacher exists.

- If official Open Duck baseline walks but `./scripts/run_policy_rollout_smoke.sh open_duck_forward` only wiggles, treat it as a Soridormi runtime parity bug, not a training/data problem. `open_duck_forward` should export `SORIDORMI_SIM_PREROLL_STEPS=1` so sync-step MuJoCo pre-rolls one API step before first policy inference.
- For official-vs-Soridormi trace parity, force JSONL logging through the smoke wrapper with `--log-format jsonl --log-prefix parity_open_duck_forward --log-dir /data/logs`. Policy profiles may default to MCAP, so parity commands must not rely on `SORIDORMI_RUNTIME_LOG_FORMAT=jsonl` alone unless `run_policy_experiment.sh` preserves the override after profile resolution.
