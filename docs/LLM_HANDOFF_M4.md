# Soridormi LLM Handoff: M4 Status

## Big picture

The user wants Soridormi to become a sim-to-real robot engineering platform:

1. Run successfully in simulator.
2. Train/replace models.
3. Transfer to real robot with the same runtime contracts.

Current work is still M4: runnable ONNX policy system. The official Open Duck policy works in the official baseline runner, but Soridormi's own runtime still does not reproduce full forward walking.

## Roadmap agreement

M4: Runnable ONNX policy system
- One-command start.
- Robot moves in MuJoCo.
- Logs all state/action/debug info.
- Compare runs.
- Survive/reset/restart.
- Model path configurable.

M5: Model replacement interface
- Compatible ONNX models can be dropped in.
- Observation/action contracts documented.
- Policy metadata/config externalized.
- Multiple policies selectable by profile.
- Validation tools check model compatibility.

M6: Training pipeline
- Data collection.
- Rewards/tasks.
- Train control policy.
- Export ONNX.
- Run same runtime with new model.

M7: Transfer to real robot
- Jetson image.
- Hardware backend.
- Motor/IMU/battery interfaces.
- Safety supervisor/emergency stop.
- Dry-run, low-power tests, tethered walking.

## What has been proven

### Official baseline works

Command:

```bash
./scripts/run_official_forward_baseline.sh
```

Observed result:

- Official baseline finished.
- `BEST_WALK_ONNX_2.onnx` walked forward.
- Example: about 1.82 m forward over 19.4 s.

The official baseline initially hit dependency/asset/cleanup issues:

- Missing `jax`: fixed with a JAX-free `get_assets()` stub.
- Duplicate `head.stl` asset names: fixed with asset dedupe.
- Segfault on exit after summary: fixed with fast process exit after summary.

### Soridormi backend is correct under official target replay

Command sequence:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/replay_latest_official_targets.sh
./scripts/compare_official_replay_trace.sh
```

Result over 100 compared steps:

- `motor_targets mean_mae = 0.0`
- `joint_positions mean_mae = 0.0`
- `joint_velocities mean_mae = 0.0`
- `contacts mean_mae = 0.0`
- official and Soridormi forward displacement matched exactly.

Conclusion: the MuJoCo backend/control stepping/reset/model are correct when fed official motor targets.

### ONNX wrapper is correct

Command:

```bash
./scripts/check_latest_observation_action_parity.sh
```

Result:

- Official observations through Soridormi ONNX wrapper reproduce official actions with zero error.
- Soridormi observations through Soridormi ONNX wrapper reproduce Soridormi logged actions with zero error.
- First step official and Soridormi samples are nearly identical.

Conclusion: model loading/provider/input/output handling is correct.

### Phase and command are correct

After mounting runtime access to official reference data:

- `phase mean_mae = 0.0`
- `command mean_mae = 0.0`

This fixed a previous major source of mismatch.

## Remaining problem

Soridormi closed-loop policy still diverges after the first step. Latest representative metrics:

- observation mean MAE around `0.127`
- action mean MAE around `0.134`
- motor target mean MAE around `0.033`
- contacts mean MAE around `0.18`
- official forward displacement over compared window about `0.1708 m`
- Soridormi forward displacement about `0.0385 m`

Worst observation segments:

1. `accelerometer_xyz`
2. `gyro_xyz`
3. `feet_contacts`
4. `last_action`
5. `last_last_action`

Since first step matches, these are likely closed-loop timing/history effects, not static layout errors.

## Likely next work: M4.13

Add exact official loop-order parity and first-divergence analyzer.

The analyzer should answer:

- At which step does official-vs-Soridormi first exceed threshold?
- Which observation segment diverges first?
- Is divergence caused by `last_action` history, `motor_targets`, IMU, or contacts?
- Are logs comparing pre-step vs post-step states by mistake?
- Does Soridormi update action history before or after the same operation as official?
- Does Soridormi log motor targets before or after speed limiting in the same way as official?

Suggested output:

```text
First divergence:
  step: 1
  segment: last_action or motor_targets or imu/contact
  metric: mean_abs_error / max_abs_error
  official: [...]
  soridormi: [...]
  interpretation: ...
```

## Do not do next

- Do not train a new model yet.
- Do not switch to open-loop gait.
- Do not tune action scale/command to hide mismatch.
- Do not rewrite backend; target replay already proved backend behavior.

## If the new assistant needs to browse

The repo is public: `https://github.com/TimeTreker/soridormi.git`, branch `main`.

The README states the project goal: sim-to-real humanoid stack based on Open Duck Mini v2 with separated runtime, simulator, and shared API.
