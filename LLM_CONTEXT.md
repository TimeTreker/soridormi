# LLM_CONTEXT.md

This file is a compact handoff for starting a new LLM session on Soridormi.

## One-paragraph project summary

Soridormi is a sim-to-real humanoid robot stack for Open Duck Mini v2. It separates runtime, simulator, and shared API so the same policy runtime can talk to MuJoCo now and real robot hardware later. The immediate goal is to make the official Open Duck ONNX walking policy run successfully through Soridormi's engineering runtime. After that, the project will support model replacement, training, and transfer to Jetson/real hardware.

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
- PC dev currently uses CUDA 12.8/cuDNN because ONNX Runtime GPU needed CUDA 12 libraries in this environment.

## Latest known status

The project is in M4, trying to reproduce official Open Duck inference behavior inside Soridormi.

Already working:

- Docker sim/runtime stack.
- MuJoCo backend with Open Duck Mini v2 model.
- Official Open Duck baseline runner.
- MCAP logging and trace analysis.
- Policy profiles such as `open_duck_forward`.
- Model checker for ONNX input/output contract.
- Official-vs-Soridormi trace comparison.
- Official motor-target replay.
- Observation/action parity checker.

Important confirmed facts:

- Official Open Duck baseline using `BEST_WALK_ONNX_2.onnx` walks forward in the same Docker/MuJoCo environment.
- Replaying official motor targets through Soridormi exactly reproduces official trajectory over the compared window.
- Soridormi's ONNX wrapper reproduces official actions exactly when fed official observations.
- Soridormi command and phase now match official trace exactly.
- Soridormi policy run still moves much less forward than official; remaining mismatch is in closed-loop observation/history/timing, mainly IMU/contact/history divergence after the first step.

## Most recent metrics to remember

After M4.8/M4.9/M4.12, comparison still looked approximately like:

- `phase mean_mae = 0.000000`
- `command mean_mae = 0.000000`
- observation mean MAE around `0.127`
- action mean MAE around `0.134`
- worst observation segments:
  - `accelerometer_xyz`
  - `gyro_xyz`
  - `feet_contacts`
  - `last_action` / `last_last_action`
- official forward displacement over 100 compared policy steps: about `0.1708 m`
- Soridormi forward displacement over same compared window: about `0.0385 m`

Parity checker also showed:

- `official_obs_vs_official_action mean_mae = 0.0`
- `soridormi_obs_vs_soridormi_action mean_mae = 0.0`
- First step samples are nearly identical between official and Soridormi.

Therefore the next work should focus on first divergence / loop order, not model/provider/backend.

## Next recommended milestone: M4.13

Title: exact official loop-order parity and first-divergence analyzer.

Goals:

1. Add a first-divergence report for official-vs-Soridormi traces.
2. Identify the first step and exact observation segment that diverges.
3. Verify `last_action`, `motor_targets`, IMU, and contacts are sampled/updated in the same order as official.
4. Port the exact official update order into Soridormi runtime if mismatch is found.
5. Re-run official vs Soridormi comparison and target forward displacement.

Do not start M5/M6 training until Soridormi can reproduce the official walking policy well enough.

## New-session prompt

A good prompt to start the next session:

> Please read `CLAUDE.md`, `LLM_CONTEXT.md`, and `docs/LLM_HANDOFF_M4.md` first. We are debugging Soridormi M4. The official Open Duck baseline walks forward, official motor-target replay in Soridormi matches exactly, ONNX wrapper parity is exact, phase/command match, but Soridormi closed-loop still diverges after step 0 with IMU/contact/history differences and weak forward displacement. Continue with M4.13: exact official loop-order parity and first-divergence analyzer. Do not switch to open-loop or training yet.
