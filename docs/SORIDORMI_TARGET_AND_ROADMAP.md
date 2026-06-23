# Soridormi target and roadmap

This document defines the system target, ownership boundaries, milestone
direction, and current candidate evidence. For the gated execution sequence,
acceptance criteria, risks, and immediate work queue, see
`docs/SORIDORMI_EXECUTION_ROADMAP.md`. For the agreed Chromie/Soridormi
multi-agent split, see
`docs/CHROMIE_SORIDORMI_MULTI_AGENT_ARCHITECTURE.md`.

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

Destination language such as "walk forward to the house" is not a locomotion
command. It is a navigation goal. Soridormi must refuse it until a sensing and
navigation layer resolves the target, localizes the robot, plans a route,
checks local obstacles, and produces bounded local trajectory or velocity
segments. See `docs/SORIDORMI_NAVIGATION_GOAL_CONTRACT.md`.

Low-level policy direction:

```text
robot_state + desired_command + task_context + environment_context + short_history -> action_14d
```

Natural language belongs to Chromie. Physical execution belongs to Soridormi.
Both systems may use a DAG engine: Chromie for global human-facing
orchestration, Soridormi for embodied task execution and safety-critical local
planning.

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

profile: context_stage1_three_scenario_10ep_e80
input: robot_state.observation[101] + desired_command(vx_mps, vy_mps, yaw_radps)
model input shape: [1, 104]
model path: /data/training_runs/context_stage1_three_scenario_10ep_neural_bc_m10_e80/neural_behavior_clone.onnx
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

Three-scenario candidate checkpoint evidence:

```text
profile: context_stage1_three_scenario_10ep_e80
prepared dataset: /data/training_datasets/context_bc/prepared/context_stage1_three_scenario_10ep/prepared_manifest.json
raw scenario data:
  flat_walk_varied_speed_v1_10ep: 3000 samples
  start_stop_velocity_ramp_v1_10ep: 3000 samples
  curve_turn_walk_v1_10ep: 3000 samples
prepared splits: train 7200, val 900, test 900
model/profile contract: OK
offline evaluation:
  train MAE ~= 0.00720
  val MAE ~= 0.01101
  test MAE ~= 0.01244
flat/start-stop/curve suite: PASS, 3/3 scenarios accepted
  flat_walk_varied_speed_v1: forward_distance ~= 0.312 m, mean speed ~= 0.0627 m/s
  start_stop_velocity_ramp_v1: forward_distance ~= 0.267 m, mean speed ~= 0.0411 m/s
  curve_turn_walk_v1: forward_distance ~= 0.155 m, mean speed ~= 0.0282 m/s
  total_forward_distance_m ~= 0.733
  mean_forward_speed_mps ~= 0.0440
  fallen_count: 0
```

Conclusion: the three-scenario candidate remains the best restored context-BC
baseline, but it is blocked by the current G10 clearance gate. On 2026-06-22
the historical ONNX was restored locally with sha256
`2a7e41afe855702638aed56ec32e0f5e067a6b76fdcd76af4d43a101191730b7`; the
locally regenerated ONNX was preserved separately because it did not reproduce
the historical pass. The restored model reproduces the old movement behavior,
but all three scenarios remain below the `0.015 m` swing-clearance target.

The strongest retained residual reference has advanced beyond
`m10_command_state_mlp_cem4x14_s79`. The first restored warm-start candidate,
`clearance_gap_sequence_restored_s83`, was reproducible and profile-contract
valid, but failed `0/3` scenarios and did not beat `s79` because the wrapper
changed residual scale from the retained checkpoint's `0.1` to `0.05`. The
wrapper now preserves `0.1` by default.

The balanced retained residual reference is
`clearance_lowratio_gatepush_s111`. It is still blocked from promotion, but it
improves total distance to about `1.928 m`, gets all three p50 swing-clearance
metrics above `0.015 m`, and reduces max low-clearance ratio to about `0.496`.
The quantile-tail probe `clearance_lowratio_quantile_s119` is also blocked, but
it improves total distance to about `1.938 m` and reduces the worst-case
low-clearance ratio to about `0.473`; it is not a clean replacement because
flat low-clearance ratio regressed from about `0.388` to about `0.408`.
The remaining blocker is the low-clearance-ratio gate in all three scenarios.
The next M10 work is clearance refinement against both `s111` and `s119`,
followed by quantitative clearance readiness and a direct human follow-camera
visual pass before any broader promotion.

### M11A: task-agent contract foundation

Expose rich embodied goals to Chromie without pretending that full autonomy is
implemented. This milestone adds a no-motion task-level MCP contract above the
existing named-skill surface:

```text
soridormi.task.get_capabilities
soridormi.task.preview
soridormi.task.submit
soridormi.task.status
soridormi.task.events
soridormi.task.cancel
```

Soridormi keeps its own task readiness table in
`configs/task_capabilities/open_duck_mini_v2_task_capabilities.json`, returns
structured `plan_steps`, `blocked_subsystems`, `recommended_next_actions`, and
a Soridormi-owned `task_graph`, and refuses missing navigation, perception,
manipulation, unsafe physical tasks, and stop-through-task requests by default.

Validation:

```bash
./scripts/validate_task_agent_contract.sh
```

Passing this gate means Chromie can inspect and monitor Soridormi's embodied
interpretation safely. It does not mean Soridormi can physically navigate,
fetch objects, or execute task-level plans yet.

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

Current integration direction: Chromie now treats Soridormi as a named-skill
provider through MCP. The intended body path is `soridormi.skill.list`,
`soridormi.skill.create_plan`, safety monitoring, `soridormi.skill.execute_plan`,
and post-action `soridormi.robot.get_status` safe-idle confirmation. The older
bounded `soridormi.motion.*` tools remain useful for low-level velocity-plan
tests and stop/cancel controls, but Chromie-facing body behavior should be
expressed as named skills.

Soridormi remains authoritative for availability, command bounds, cancellation,
safe hold, emergency stop, MuJoCo execution, and future hardware execution.
Chromie remains authoritative for conversation, confirmation, speech, task
choice, and interaction-scoped scheduling.

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
