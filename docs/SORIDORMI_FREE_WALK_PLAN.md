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

## M6A free-walk evaluation entrypoint

The first M6A implementation artifact is a conservative fixed-command evaluation suite:

```text
configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

It covers stand/stop, slow and medium forward walking, small backward motion, yaw-in-place, curved walking, and small lateral commands. Validate the suite without MuJoCo:

```bash
PYTHONPATH=src python -m soridormi_runtime.free_walk_eval --suite configs/teacher_suites/open_duck_free_walk_eval_v1.yaml
```

Run the teacher-vs-candidate free-walk comparison with MuJoCo already running:

```bash
./scripts/run_sim_server.sh
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

## Hardware rule

Do not start hardware walking from this milestone. Hardware work may only begin as read-only state, dry-run command validation, watchdog, emergency stop, and low-power single-joint tests. Walking hardware execution is blocked until commanded free-walk simulation acceptance exists.
