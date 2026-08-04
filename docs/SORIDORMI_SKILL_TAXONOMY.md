# Soridormi Skill Taxonomy and Behavior Platform

Status: structured body-skill interface bootstrap proposal.

Soridormi is the robot cerebellum/body runtime. Chromie is the robot brain in
`TimeTreker/chromie.git` on `main`. structured body-skill interface starts the body-skill interface between
them: Chromie or another high-level planner should ask Soridormi for safe named
behaviors instead of directly manipulating velocity commands, joint targets, or
motor commands.

## Design decision: define the full skill universe first

For interactive robots, the skill vocabulary is a product/API boundary. It should not be limited to the 6 to 8 behaviors that are executable today.

Soridormi should therefore define the full desired skill universe now, but land implementations one by one:

```text
Skill declared in manifest
  ↓
status says whether it is executable, planned, future, or unsupported
  ↓
implementation phase says when it should land
  ↓
controller/evaluation fills in over time
```

This means a future skill such as `run`, `step_over_obstacle`, or `wave_hand` can exist in the manifest today without being executable. Higher-level systems can understand the desired body vocabulary while the local Soridormi validator rejects unavailable or unsupported skills.

## Why skills instead of one walking controller?

An interactive robot needs more than forward walking. It should be able to stand, stop, walk, turn, look toward a person, nod, shake its head, bow, sit, stand up, recover, and eventually step over obstacles or run.

A skill is not just an animation name. It is a robot-body capability with:

- a stable identifier;
- a category;
- parameter ranges;
- preconditions;
- required actuator groups;
- safety constraints;
- execution type;
- current availability;
- implementation phase;
- validation/evaluation requirements.

## Important Open Duck Mini v2 hardware boundary

The current `open_duck_mini_v2` actuator map has legs plus head/neck joints:

```text
legs:
  left/right hip yaw, hip roll, hip pitch, knee, ankle

head_neck:
  neck_pitch, head_pitch, head_yaw, head_roll
```

It does **not** currently expose arm or hand actuators in Soridormi's 14-action policy contract. Therefore skills such as `wave_hand`, `point_direction`, or `high_five` are useful to declare conceptually, but they must be marked `unsupported_current_robot` until matching hardware and a controller exist.

For the current robot, first social skills should use head/neck/body-safe gestures:

```text
look_direction
look_at_person
nod_yes
shake_no
small bow
attention gesture
```

Speech/TTS belongs outside Soridormi, in Chromie. Soridormi owns the body
action and may reject a requested skill if it is unsupported, unsafe,
unavailable, or outside validated ranges.

## Skill capability structure

### skill_manifest: full skill manifest

Add the machine-readable skill universe:

```text
configs/skills/open_duck_mini_v2_skills.json
```

The manifest is both documentation and a future runtime contract. It describes which behaviors exist, which are currently executable in simulation, which are planned, which are future work, and which are unsupported on the current hardware.

### skill_listing_validation: skill listing and validation

Add tools so Soridormi can report:

- what skills exist;
- which are available now;
- which are planned;
- which are future;
- which require missing hardware;
- which require policy, keyframe, residual/RL, perception, or external speech.

### locomotion_wrappers: locomotion wrappers

Wrap existing command-conditioned locomotion as skills:

```text
stand_idle
stop
walk_velocity
walk_forward
walk_backward
turn_in_place
turn_left
turn_right
curve_walk
curve_left
curve_right
sidestep
sidestep_left
sidestep_right
```

These should lower to the existing velocity-command policy path and continue to use MuJoCo acceptance gates.

For human-facing language, Chromie should prefer semantic wrappers such as
`walk_forward(speed=slow|normal|medium|quick|fast_limited)` instead of asking
the user or LLM to provide exact `vx_mps` values. `walk_velocity` remains the
bounded engineering/debug primitive beneath those wrappers.

### head_social: head/neck social skills

Implement safe scripted/keyframe social behaviors in MuJoCo first:

```text
look_direction
look_at_person
track_person later
nod_yes
shake_no
bow
express_attention
greeting as a composition
```

These should be interruptible, bounded by joint limits, and default to simulation first.

### posture_transitions: posture transitions

Implement and evaluate:

```text
sit_down
stand_up
crouch
recover_stand
balance_recover
```

These should start as scripted pose/keyframe teachers or separate teacher controllers, not as random walking data.

### obstacle_and_terrain: obstacle and terrain skills

Evaluate and eventually train:

```text
trajectory_follow
step_over_obstacle
rough_ground_walk
```

The target is not merely surviving rough terrain. A pass requires following the requested trajectory, maintaining progress, and clearing obstacles.

### fast_locomotion: fast locomotion

Declare `run` as future. Do not expose it as executable until fast walking is proven in MuJoCo with velocity tracking, fall-rate, and thermal/actuator-margin acceptance gates.

### future_hardware_extensions: future hardware extensions

Declare arm/hand gestures but reject them on current Open Duck Mini v2:

```text
wave_hand
point_direction
high_five
```

These require hardware and controller support that the current actuator map does not provide.

## Skill categories

The first manifest uses these categories:

```text
locomotion
posture
social
hardware_extension
```

The `hardware_extension` category is intentionally included so the desired interaction vocabulary is visible without pretending current hardware can execute it.

## Status vocabulary

The manifest distinguishes declaration from executability:

```text
available_sim                 executable in MuJoCo simulation
available_sim_experimental    executable in MuJoCo but still weak/experimental
planned                       planned controller, not executable yet
planned_wrapper               planned human-facing wrapper around an existing lower-level skill
planned_external_target       needs target information from perception/Chromie/etc.
future                        future capability issue
future_pose_teacher           needs pose/keyframe/teacher controller
future_residual_rl            needs residual/RL or terrain-aware training
future_evaluation             needs evaluation infrastructure first
future_perception             needs perception target tracking
future_composite              composition of other skills and external capabilities
unsupported_current_robot     declared, but current robot lacks required actuators
```

Only `available_sim` and carefully selected `available_sim_experimental` skills should be callable by early runtime/demo tools.

## Execution types

A skill can be implemented by different controller types:

```text
policy_velocity              existing command-conditioned walking policy
skill_wrapper                human-facing wrapper that lowers to another policy command
scripted_keyframe            deterministic joint/head/neck trajectory
scripted_tracking            scripted tracking loop using external target direction
pose_transition              keyframe or controller for posture changes
trajectory_policy            future trajectory-following interface
residual_policy              teacher plus learned residual correction
future_policy                future learned controller
future_hardware_extension    requires hardware not present today
composite                    composition of multiple Soridormi and external skills
```


## skill capability export skill listing and validation CLI

skill capability export turns the JSON skill universe into an inspectable runtime contract. It still does not execute skills. It only answers what Soridormi declares, what is currently available in simulation, and what remains future or unsupported.

Use the host-side wrapper:

```bash
./scripts/list_skills.sh
./scripts/list_skills.sh --validate-only
./scripts/list_skills.sh --available
./scripts/list_skills.sh --category social --include-unsupported
./scripts/list_skills.sh --skill walk_velocity --json
./scripts/list_skills.sh --llm-context --language zh
```

The same implementation is available as a Python module:

```bash
PYTHONPATH=src python -m soridormi_runtime.skill_manifest --validate-only
PYTHONPATH=src python -m soridormi_runtime.skill_manifest --available --json
```

Validation checks include:

- unique skill IDs;
- declared status and execution vocabulary;
- known implementation phases;
- safe defaults with `hardware_enabled=false`;
- parameter min/default/max sanity;
- available skills must not require unsupported actuator groups.

The `--llm-context` output is intentionally read-only context for Chromie/MCP planners. It must not be interpreted as permission to execute planned or unsupported skills.

## First executable subset

The first runtime implementation should target 6 to 8 safe simulation skills, not the full manifest.

Recommended first executable subset:

```text
stand_idle
stop
walk_velocity
turn_in_place
curve_walk
sidestep
look_direction
nod_yes or shake_no
```

Everything else may remain declared but unavailable until its controller and validation exist.

## Skill parameters and policy context

The skill manifest is the human/agent-facing vocabulary, but executable locomotion skills should lower to structured policy context rather than hard-coded gait names. For example, `walk_velocity` should carry continuous velocity parameters, while future terrain/obstacle skills may add clearance, stride, terrain, and obstacle fields:

```text
skill_id + desired_command + task_context + environment_context -> policy input context
```

Semantic aliases such as `slow`, `normal`, `fast`, `careful`, or `high_clearance` may be useful at the UI or planner layer, but they must resolve to bounded numeric/context fields before reaching the low-level policy. Soridormi should reject unsupported context combinations the same way it rejects unsupported skills.

See `docs/SORIDORMI_POLICY_CONTEXT_CONTRACT.md` for the policy input contract.

## Relationship to BC, residual, and RL

Behavior cloning can copy an existing teacher skill. It cannot invent a better skill than the teacher. If the teacher drags its feet, BC will likely drag its feet too.

Residual/RL is the right tool when Soridormi needs to improve beyond a teacher, such as obstacle-aware leg lift, rough-ground progression, faster walking, or trajectory-aware stepping.

Recommended order:

1. define full skill universe;
2. land listing/validation;
3. implement safe locomotion wrappers;
4. implement head/neck social gestures;
5. implement pose-transition teachers;
6. evaluate trajectory/progress/clearance;
7. use residual/RL to improve skills that need adaptation.

## Out of scope for skill taxonomy

skill taxonomy does not implement skill execution. It only defines the taxonomy and manifest so future code and future LLM/MCP sessions share the same vocabulary.


## skill execution contract dry-run skill execution registry

skill execution contract turns the manifest into a safe execution-facing registry without moving the robot.
The registry resolves available simulation skills into high-level velocity command plans
and rejects future or unsupported skills.  Use:

```bash
./scripts/run_skill_dry_run.sh --list
./scripts/run_skill_dry_run.sh walk_velocity --args '{"vx_mps":0.12,"duration_s":2.0}'
```

This is intentionally dry-run only.  MuJoCo execution, MCP exposure, and hardware
execution must be added in later capability issues after controller-specific validation.
