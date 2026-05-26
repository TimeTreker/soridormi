# CLAUDE.md

Project-local instructions for Claude Code or any coding assistant working on Soridormi.

## Project identity

Soridormi is a sim-to-real humanoid robot development stack based on Open Duck Mini v2. The engineering goal is to make one runtime/policy engine run in MuJoCo first, then support training/model replacement, then transfer the same runtime contracts to real robot hardware.

Do not treat this as a one-off demo script. The official Open Duck scripts are the reference behavior; Soridormi is the engineering platform that should reproduce that behavior through clean APIs, logging, profiles, and Docker workflows.

## Long-term roadmap

M4: Runnable ONNX policy system
- Starts with one command.
- Robot moves in MuJoCo.
- Logs state/action/debug info.
- Can compare runs.
- Can survive/reset/restart.
- Model path is configurable.

M5: Model replacement interface
- Any compatible ONNX model can be dropped in.
- Observation/action contract is documented.
- Policy metadata/config is externalized.
- Multiple policies can be selected by profile.
- Validation tools check model compatibility before runtime.

M6: Training pipeline
- Collect data.
- Define reward/tasks.
- Train control policy.
- Export ONNX.
- Run the same Soridormi runtime with the new model.

M7: Transfer to real robot
- Jetson runtime image.
- Hardware backend implementation.
- Motor driver interface.
- IMU/battery/state reader.
- Safety supervisor and emergency stop.
- Dry-run mode, low-power tests, tethered walking.

## Current active milestone

Current milestone: M4.x, specifically M4.13 next.

Current goal: make Soridormi's ONNX policy runtime reproduce the official Open Duck MuJoCo inference loop closely enough that `open_duck_forward` walks forward inside Soridormi, not only inside the official baseline runner.

## Critical evidence already established

1. Official Open Duck baseline works in the Docker/MuJoCo environment.
   - `./scripts/run_official_forward_baseline.sh`
   - Result observed: `BEST_WALK_ONNX_2.onnx` walks forward.
   - Example summary: about 1.82 m forward over about 19.4 s.

2. Soridormi target replay exactly reproduces official motion.
   - Official motor targets replayed through Soridormi MuJoCo backend produced zero error on motor targets, joint positions, joint velocities, contacts, and forward displacement over the compared window.
   - Conclusion: simulator backend, reset pose, MuJoCo model, stepping, and command application are correct when given official motor targets.

3. ONNX inference wrapper is correct.
   - Official observations run through Soridormi's ONNX wrapper reproduce official actions with zero error.
   - Soridormi observations run through Soridormi's ONNX wrapper reproduce Soridormi logged actions with zero error.
   - Conclusion: ONNX Runtime provider/session/input/output handling is not the bug.

4. Phase and command are already aligned.
   - `phase mean_mae = 0.0`
   - `command mean_mae = 0.0`
   - This was fixed by mounting/using official `polynomial_coefficients.pkl` reference data in the runtime container.

5. Remaining divergence is closed-loop observation/history/timing.
   - First step official and Soridormi observations/actions are nearly identical.
   - Later steps diverge.
   - Worst segments repeatedly: `accelerometer_xyz`, `gyro_xyz`, `feet_contacts`, plus action history segments caused by earlier action divergence.
   - M4.12 synchronous API did not improve metrics; do not assume it solved the issue.

## Do not regress these decisions

- Keep the ONNX policy path. Do not switch to open-loop gait unless explicitly requested.
- Do not train a new model yet. The official model already works in the official baseline.
- Do not rewrite the MuJoCo backend unless trace evidence proves it is necessary. Target replay proved backend correctness.
- Do not randomly tune command/action scale to hide the mismatch. Use official-vs-Soridormi trace comparison.
- Do not remove logging/comparison tools. They are the main debugging mechanism.

## Active next task: M4.13

Implement exact official loop-order parity.

The suspected mismatch is not static observation layout. It is loop order / history timing / sensor sample timing:

Official-style loop should be treated as the reference:

1. Observe current state after prior MuJoCo decimation.
2. Build observation using current IMU/joints/contacts, previous action history, previous motor targets, command, and phase.
3. Run ONNX inference.
4. Update action history according to official timing.
5. Compute motor targets from `default_actuator + action * action_scale` with motor speed limit.
6. Apply motor targets.
7. Step MuJoCo decimation times.
8. Read next state.

M4.13 should add a first-divergence report:

- first step where obs MAE exceeds a small threshold
- first segment to diverge
- official values and Soridormi values for that segment
- last_action / motor_targets history at that step
- sensor sample time and robot time

## Useful commands

Official reference:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/show_official_baseline_summary.sh
```

Soridormi policy run:

```bash
./scripts/run_official_compatible_policy_server.sh open_duck_forward
./scripts/run_policy_experiment.sh open_duck_forward
```

Trace comparison:

```bash
./scripts/compare_latest_official_soridormi_trace.sh
./scripts/check_latest_observation_action_parity.sh
```

Target replay isolation:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/replay_latest_official_targets.sh
./scripts/compare_official_replay_trace.sh
```

Tests:

```bash
pytest -q
```

## Engineering style

- Make small, trace-driven changes.
- Prefer complete files in patch zips when updating the user.
- Keep default behavior safe.
- If adding scripts, make them host-runnable wrappers that enter Docker internally.
- Avoid silent fallback for official-compatibility modes; fail fast if required files are missing.
- Preserve compatibility with future real robot backend: the runtime should speak `RobotState` and `MotorCommand`, while backend implementation changes.
