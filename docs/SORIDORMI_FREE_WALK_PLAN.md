# Soridormi free-walk plan

## Goal

The Soridormi-side goal is to make Open Duck Mini v2 walk freely in simulation first, then later on hardware, under bounded high-level velocity commands:

```text
vx: forward/backward velocity command
vy: lateral velocity command
yaw: turn-rate command
stop / cancel / emergency stop
```

"Freely" does **not** mean unbounded motor control or raw torque control. It means the robot should be able to stand, stop, walk forward, turn, curve, and handle command changes within the policy's trained command envelope while preserving joint limits, fall detection, runtime limits, and rollout acceptance gates.

The policy direction is context-conditioned control:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

For M6, `desired_command` is continuous `vx_mps`, `vy_mps`, and `yaw_radps`. For later terrain and obstacle skills, task/environment context should include fields such as `skill_id`, gait style, target clearance, terrain type, obstacle distance, and obstacle height. See `docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md`.

## Current main-branch progress

The current main branch already has the important simulation-learning pieces:

- official/Open Duck policy runtime profile and observation/action parity work;
- fixed `obs[101] -> action[14]` policy contract;
- action-to-target mapping instead of direct torque output;
- teacher-suite generation and command-grid comparison utilities;
- direct live teacher rollout collection from MuJoCo;
- grouped train/validation/test splitting to avoid rollout leakage;
- neural behavior-cloning training and ONNX export;
- rollout comparison and failure diagnosis tooling;
- residual policy wrapper, residual training scaffold, RL fine-tune environment, and walking reward;
- hardware backend is still a placeholder and should not be used for walking.

That means the next Soridormi milestone is **not** MCP, LLM routing, or hardware walking. The next milestone is proving command-conditioned walking in MuJoCo.

## Updated milestone order

```text
M6A: Commanded free-walk evaluation in MuJoCo
M6B: Command-distribution teacher data collection
M6B.1: Continuous-speed teacher data with smooth command ramps and coverage reports
M6C: Neural BC policy trained on command-grid/random-command data
M6D: Teacher-vs-candidate closed-loop comparison across the command suite
M6E: Residual policy improvement only after BC and evaluation are reliable
M6F: Sim acceptance gate for free-walk candidates
M7: Hardware read-only / dry-run bridge, then low-power bring-up
M8: Chromie/MCP/LLM orchestration after Soridormi exposes reliable robot capabilities
```

## Why evaluation comes before more training

Training without a commanded-walk evaluation gate can produce a model with good offline loss but poor closed-loop locomotion. The first practical deliverable should be an evaluation suite that answers:

- Can the teacher stand still under zero command?
- Can the teacher walk forward at several speeds?
- Can it turn in place left/right?
- Can it curve while walking?
- Can it tolerate short command switches?
- Which commands terminate early or fall?
- Does a neural BC candidate match the teacher closed-loop, not just offline?

Only after those questions are measurable should we invest in larger datasets, DAgger, residual RL, PPO/SAC, or recurrent actors.

## Minimum commanded-walk suite

Use the existing teacher suite as the baseline command coverage:

```text
stop
walk_slow_forward
walk_forward
fast_forward
turn_left_in_place
turn_right_in_place
curve_left
curve_right
lateral_left
lateral_right
```

The first acceptance suite should also add command switching:

```text
stand -> slow forward -> curve left -> curve right -> stop
stand -> turn left -> turn right -> stop
stand -> lateral left -> lateral right -> stop
```

Command-switching scenarios should start conservative and use command ramping. They are meant to find instability, not to demonstrate maximum speed.

## Metrics that matter

A free-walk report should include at least:

```text
survival_time_s
termination_reason
fall_or_reset_count
forward_distance_m
mean_forward_velocity_mps
velocity_tracking_error
lateral_drift_m
yaw_rate_tracking_error
height_error
upright_error
action_abs_max
action_l2_mean
action_rate_mean
residual_l2_mean, for residual policies
```

A candidate that only improves supervised MAE is not accepted unless it also survives and tracks commands in MuJoCo.

## Near-term implementation plan

1. Add a commanded-walk evaluation suite and report format.
2. Make teacher and candidate profiles run through the same command scenarios.
3. Summarize pass/fail per command, not only one aggregate score.
4. Add command-switching scenarios after fixed-command scenarios are stable.
5. Train neural BC on the command grid only after the teacher command suite is measured.
6. Use residual fine-tuning only after teacher-vs-BC closed-loop comparison is reproducible.


### Current blocker: official-vs-Soridormi runtime parity

If `run_official_forward_baseline.sh` walks but `run_policy_rollout_smoke.sh open_duck_forward` only wiggles, stop all random teacher data collection. The ONNX model is good, but Soridormi's engineering path is not yet parity-compatible with the official `MjInfer` loop. See `docs/official_sync_preroll_m6_debug.md` for the sync pre-roll compatibility hook and the required MuJoCo validation commands.

## M6A free-walk evaluation entrypoint

The first M6A implementation artifact is a conservative fixed-command evaluation suite:

```text
configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

It covers stand/stop, slow and medium forward walking, small backward motion, yaw-in-place, curved walking, and small lateral commands. Validate the suite without MuJoCo:

```bash
PYTHONPATH=src python -m soridormi_runtime.free_walk_eval --suite configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

The static validator is intentionally host-friendly: it should work even when PyYAML is not installed, because this command often runs before the full runtime container or editable Python environment is rebuilt. Keep the suite syntax conservative if new scenarios are added.

Run the teacher-vs-candidate free-walk comparison with the MuJoCo backend already running. The default functional test should be headless/no-viewer:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

For visual inspection, start the same MuJoCo backend with the passive viewer explicitly enabled:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

If the duck walks out of the initial frame, use the viewer follow camera:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Optional follow-camera tuning:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --camera-distance 1.6 --camera-azimuth 135 --camera-elevation -20
```

In another terminal:

```bash
./scripts/run_free_walk_eval.sh neural_bc_teacher_grid --force
```

Use `--dry-run` first when checking profiles and generated rollout commands:

```bash
./scripts/run_free_walk_eval.sh neural_bc_teacher_grid --dry-run --force
```

The wrapper delegates to the existing command-grid comparison path, so the output should include per-scenario teacher/candidate rollout comparisons plus a command-grid summary.


## M6B random teacher data collection

After the fixed-command M6A suite is measurable, collect teacher data from random piecewise velocity commands. This produces trajectories that include command transitions instead of one constant command per episode. The random collector owns its MuJoCo collection lifecycle, so do not start a separate `run_sim_server.sh` for this command. Use `--viewer` on the collector command when visual inspection is needed.

Collect a conservative random-command teacher dataset:

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
  --viewer \
  --json | python -m json.tool
```

Negative range values are valid in either shell style, so both `--vx-range -0.03,0.15` and `--vx-range=-0.03,0.15` are supported.

The output JSONL stores `scenario_id`, `rollout_id`, `command_segment_index`, `command_segment_id`, `command_segment_step_index`, and the active `policy_command` for each sample. Use grouped splits by `source_log` for smoke training and by `scenario_id` or held-out seeds when testing broader generalization. Continue only if the collection JSON reports `ok: true` and a positive `sample_count`.

This collector should be used for walking, turning, stopping, small lateral motion, and command transitions. Do not include sit-down or stand-up motions until Soridormi has a separate pose-transition teacher or scripted pose-transition controller and an explicit task/mode conditioning contract.

## Hardware rule

Do not start hardware walking from this milestone. Hardware work may only begin as read-only state, dry-run command validation, watchdog, emergency stop, and low-power single-joint tests. Walking hardware execution is blocked until commanded free-walk simulation acceptance exists.

## M6C foot-clearance and rough-ground evaluation

If the duck walks but its swing feet stay very close to the ground, do not change motor limits first. Measure clearance and rough-ground robustness before changing the policy. The M6C evaluation layer adds two tools:

```text
src/soridormi_runtime/foot_clearance_eval.py
src/soridormi_sim/rough_ground_scene.py
```

Analyze a normal flat-ground rollout by starting MuJoCo with the official walking profile:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Then run a foot-clearance rollout/report in another terminal:

```bash
./scripts/run_foot_clearance_eval.sh open_duck_forward --steps 1000
```

The report is written under:

```text
data/foot_clearance/open_duck_forward/foot_clearance_report.md
```

For small-stone testing, start the simulator with a generated rough-ground scene:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera \
  --rough-ground \
  --rough-stone-height 0.008 \
  --rough-stone-count 8
```

Then run the same foot-clearance evaluation. Treat rough-ground testing as an evaluation gate first, not a training shortcut. If the teacher itself trips or scuffs on stones, behavior cloning will copy that limitation. Use residual/RL only after the flat-ground and rough-ground reports show exactly where the teacher or BC candidate lacks clearance.

Important metrics:

```text
left/right min clearance
left/right swing clearance p05/p50/mean
low-clearance swing step ratio
warnings when median swing clearance is below target
warnings when low-clearance swing steps are frequent
```


## Continuous-speed teacher data rule

The walking policy target is continuous command-conditioned locomotion, not a small set of named speeds. Teacher data should therefore sample velocity ranges and transitions:

```text
vx  in a conservative forward/backward range
vy  in a conservative lateral range
yaw in a conservative turn-rate range
```

Random teacher collection should ramp command changes over a short number of control steps by default. This teaches the BC model smooth changes such as slow -> normal -> fast -> stop and straight -> curve -> straight, which better matches future skill/navigation commands than abrupt fixed-speed jumps. Fixed command grids remain useful for repeatable evaluation, but training data should include varied continuous speed coverage.
