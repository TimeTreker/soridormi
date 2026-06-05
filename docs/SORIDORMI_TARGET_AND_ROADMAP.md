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
for training. Add profile/model contracts for 104D+ context inputs. Only after
this should context-mode neural policies be exported and packaged as runnable
ONNX profiles.

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
