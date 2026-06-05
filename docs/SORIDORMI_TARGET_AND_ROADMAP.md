# Soridormi target and roadmap

## System target

Soridormi is the robot cerebellum: the body-control, locomotion, safety, and
sim-to-real runtime for Open Duck Mini v2.

Chromie is the robot brain, maintained separately at:

```text
https://github.com/TimeTreker/chromie.git
branch: main
```

The intended whole-robot stack is:

```text
human / environment
  -> Chromie brain
       conversation, memory, intent, high-level planning, skill choice
  -> structured skill/context request
  -> Soridormi cerebellum
       validation, body skills, locomotion policy, safety, MuJoCo/hardware backend
  -> robot body
```

Chromie should decide what the robot should do. Soridormi decides whether the
body can safely do it and how to execute it.

## Boundary

Chromie must not directly emit joint targets, motor commands, or low-level 14D
policy actions. It should call bounded structured skills or provide bounded
structured context:

```text
stand_idle()
stop()
walk_velocity(vx_mps, vy_mps, yaw_radps, duration_s)
turn_in_place(yaw_radps, duration_s)
curve_walk(vx_mps, yaw_radps, duration_s)
look_at_person(target_id or target bearing)
look_direction(yaw_rad, pitch_rad)
nod_yes()
shake_no()
```

Soridormi validates command ranges, skill availability, task context,
environment context, runtime state, and safety limits before running any body
controller.

Low-level policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Natural language belongs to Chromie. Physical execution belongs to Soridormi.

## Main milestones

### M4: official policy parity

Reproduce the official Open Duck walking policy through Soridormi's own runtime,
logging, profiles, and backend contracts. Preserve official baseline, replay,
comparison, and parity tooling.

### M5: model replacement interface

Make policy profiles, model validation, ONNX/provider checks, packaging,
install/restore, and contract export reliable enough that compatible models can
be swapped without rewriting runtime code.

### M6: training and evaluation backbone

Turn runtime logs into supervised datasets, prepare deterministic splits, train
linear/neural BC baselines, and evaluate candidate policies offline and in
MuJoCo.

### M7: skill/task interface

Define safe named body skills and map them to structured commands/context. Keep
unsupported hardware skills declared but unavailable. This is the explicit body
interface Chromie should call.

### M8: interaction and scenario layer

Add scripted head/social skills, scenario rollout evaluation, readiness reports,
and interaction-oriented body behaviors. Validate in MuJoCo before hardware.

### M9: context BC data pipeline

Collect scenario-aware teacher rows, export context BC datasets, validate the
context contract, gate scenario coverage, prepare grouped splits, and train
Stage 1 context BC:

```text
robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps) -> action_14d
```

### M10: runtime context policy plumbing

Extend runtime policy execution so it can provide the same context features used
for training. Add profile/model contracts for 104D+ context inputs, then allow
context-mode neural policies to export as runnable ONNX profiles.

M10 Stage 1 starts with the explicit policy input mode:

```text
context_stage1_command:
  robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
```

Profiles must declare this mode and a `[1, 104]` model input shape before a
context-trained model is allowed to run through the runtime path.

The next checkpoint is to train/export a real Stage 1 context neural candidate,
validate its profile/model contract, and compare it in MuJoCo against the
official teacher before any promotion.

Current Stage 1 context candidates:

```text
profile: context_stage1_flat_walk_v1_10ep
input: robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
model input shape: [1, 104]
model path: /data/training_runs/context_stage1_flat_walk_v1_10ep_neural_bc_m10/neural_behavior_clone.onnx

profile: context_stage1_flat_walk_v1_10ep_e80
input: robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
model input shape: [1, 104]
model path: /data/training_runs/context_stage1_flat_walk_v1_10ep_neural_bc_m10_e80/neural_behavior_clone.onnx
```

Initial candidate checkpoint evidence:

```text
profile: context_stage1_flat_walk_v1_10ep
model/profile contract: OK
offline evaluation on flat_walk_varied_speed_v1_10ep: test MAE ~= 0.00953
bounded MuJoCo smoke: 200 steps, 104D inputs, no resets, forward_x ~= 0.259 m
```

Scenario-suite evidence:

```text
context_stage1_flat_walk_v1_10ep:
  flat/start-stop/curve suite: FAIL, 0/3 scenarios accepted
  total_forward_distance_m ~= 0.0419
  mean_forward_speed_mps ~= 0.00251
  fallen_count: 0

open_duck_forward teacher baseline on the same suite:
  PASS, 3/3 scenarios accepted
  total_forward_distance_m ~= 0.728
  mean_forward_speed_mps ~= 0.0435
```

Conclusion: M10 runtime plumbing is useful, but this Stage 1 context candidate
is not promotable. It stays upright but mostly stands still at scenario nominal
commands.

E80 candidate checkpoint evidence:

```text
profile: context_stage1_flat_walk_v1_10ep_e80
model/profile contract: OK
offline evaluation on flat_walk_varied_speed_v1_10ep: test MAE ~= 0.00596
bounded MuJoCo velocity smoke:
  vx=0.125 -> forward_x ~= 0.226 m, mean speed ~= 0.0568 m/s
  vx=0.140 -> forward_x ~= 0.308 m, mean speed ~= 0.0774 m/s
  vx=0.150 -> forward_x ~= 0.348 m, mean speed ~= 0.0874 m/s
flat/start-stop/curve suite: FAIL, 2/3 scenarios accepted
  flat_walk_varied_speed_v1: PASS
  start_stop_velocity_ramp_v1: PASS
  curve_turn_walk_v1: FAIL
  total_forward_distance_m ~= 0.502
  mean_forward_speed_mps ~= 0.0307
  fallen_count: 0
```

Conclusion: E80 is a better experimental runtime profile and fixes the
low-speed command-response threshold, but it is still not promotable because
the curve/turning scenario gets stuck. The next M10/M11 work is broader
multi-scenario context-policy data, especially curve/yaw coverage, followed by
retraining and scenario-suite comparison against the official teacher.

### M11: broader locomotion generalization

Expand scenario coverage to start/stop transitions, turning, curves, lateral
motion, rough ground, slopes, obstacles, and held-out randomized suites. Add
task context, environment context, and bounded short history in staged policy
contracts.

### M12: residual/RL improvement

Use BC as initialization. Improve beyond the teacher only where evaluation shows
a specific weakness, such as stride length, foot clearance, rough-ground
progress, obstacle crossing, recovery, or velocity tracking.

### M13: hardware bridge

Move to hardware in strict phases: read-only state streaming, motor-command
dry-run, safety limits, watchdog, emergency stop, low-power single-joint test,
standing pose, then tethered low-speed walking after MuJoCo gates pass.

### M14: Chromie brain integration

Expose Soridormi skills and status as a structured API for Chromie. Chromie
chooses high-level actions such as talking, walking, looking, nodding, or
stopping; Soridormi validates and executes body actions. The first integration
target is MuJoCo-only.

Status feedback should include at least:

```text
idle
walking
turning
looking
executing_skill
failed
unsafe
stopped
```

Soridormi must return explicit refusal/failure reasons for unsafe, unsupported,
or unavailable skill requests.
