# Soridormi Policy Context Contract

Status: design contract for upcoming BC/RL data and policy work.

Soridormi policies should evolve from a pure walking-policy wrapper into a context-conditioned robot-body controller. The long-term policy contract is:

```text
policy(
  robot_state,
  desired_state_or_command,
  task_context,
  environment_context,
  short_history
) -> action_14d
```

For Open Duck Mini v2, `action_14d` remains the 14-dimensional action/target vector consumed by the Soridormi action mapper and Robot API. The contract does not mean a language model should directly output joint actions. Natural language should be resolved by a planner or skill router into bounded structured context first.

## Input groups

### Robot state

Robot state is mandatory proprioception and runtime state. It answers: "what is my body doing now?"

Examples:

```text
joint_positions
joint_velocities
base_orientation / projected_gravity
angular_velocity
foot/contact signals when available
phase state
previous action
runtime profile and action mapping version
```

The current Open Duck policy path already provides a fixed observation vector for the official walking policy. Future models may extend or replace that observation vector, but the observation schema must be versioned and validated before training.

### Desired state or command

Desired state/command is mandatory for interactive locomotion. It answers: "what should my body do next?"

For current free walking, the primary command is continuous velocity:

```text
vx_mps
vy_mps
yaw_radps
```

Future desired-state fields may include:

```text
desired_heading_rad
desired_body_height_m
stride_scale
foot_clearance_target_m
target_pose / target_relative_pose
stop_or_hold flag
```

Do not train only on named speed buckets such as `slow`, `normal`, and `fast`. Those names may exist at the skill/UI layer, but the low-level policy should be trained and evaluated across continuous command ranges.

### Task context

Task context is compact structured intent. It answers: "which behavior mode am I executing?"

Examples:

```text
skill_id: walk_velocity | turn_in_place | curve_walk | step_over_obstacle | recover_stand
locomotion_mode: normal | cautious | obstacle_crossing | recovery
gait_style: normal | careful | long_stride | high_clearance
priority: stable | fast | clearance | energy_saving
```

Task context must be bounded and enumerable. Do not pass raw natural language to the low-level policy.

### Environment context

Environment context describes the local world relevant to the physical action. It answers: "what am I moving through?"

Start with low-dimensional, testable context:

```text
terrain_type: flat | rough | slope | obstacle
friction_estimate
slope_estimate
obstacle_distance_m
obstacle_height_m
path_curvature
local_target_direction
```

Later versions can add perception embeddings or height maps, but only behind a versioned context schema with data-quality checks.

### Short history

Some locomotion decisions depend on recent motion and command transitions. Short history may include previous commands, previous actions, contact history, or a small recurrent state. The history interface must be explicit so BC, closed-loop evaluation, and RL use the same contract.

## Data row requirement

Every BC/RL training sample should preserve the context needed to debug failures. A future context-conditioned dataset row should include at least:

```json
{
  "schema": "soridormi.policy_supervision.context_v1",
  "scenario_id": "flat_walk_varied_speed_v1",
  "rollout_id": "...",
  "timestep": 123,
  "robot_state": {
    "observation": [101]
  },
  "desired_command": {
    "vx_mps": 0.12,
    "vy_mps": 0.0,
    "yaw_radps": 0.05
  },
  "task_context": {
    "skill_id": "walk_velocity",
    "gait_style": "normal",
    "priority": "stable"
  },
  "environment_context": {
    "terrain_type": "flat",
    "obstacle_height_m": 0.0,
    "friction_estimate": 1.0
  },
  "teacher_action": [14],
  "flags": {
    "fallen": false,
    "stuck": false,
    "low_clearance": false
  }
}
```

The current `soridormi.policy_supervision.v1` JSONL format can continue to work for near-term walking BC, but collectors should add context fields as they become available instead of hiding them in file names or scripts.

## Scenario variation rule

Model quality depends on data variation. A policy trained only on one clean scene and one speed will overfit that situation. Training data should vary:

```text
speed and yaw commands
command ramps and stop/start transitions
initial pose perturbations
terrain type and friction
rough-ground/stones/slopes when supported
obstacle distance/height when training obstacle skills
skill/task context labels
```

Evaluation must include both fixed checkpoints and randomized stress tests:

```text
training: broad continuous/random scenario distribution
evaluation: fixed reproducible suites + held-out randomized scenarios
```

## BC then RL

Behavior cloning should come first when a reliable teacher exists. BC copies the teacher distribution and gives Soridormi a stable policy initialization. BC should not be expected to invent better foot clearance, larger stride, or obstacle strategy than the teacher.

Residual/RL should come after BC and evaluation when Soridormi needs to improve beyond the teacher:

```text
larger stride
higher foot clearance
rough-ground progress
obstacle crossing
recovery after disturbance
better velocity tracking under terrain variation
```

RL reward and termination conditions must reference the same context fields used by BC and evaluation, such as `skill_id`, `terrain_type`, `obstacle_height_m`, and target velocity.

## Staged adoption

Use a staged policy input roadmap:

```text
Stage 1: robot_state + continuous velocity command -> action_14d
Stage 2: add skill_id, gait_style, stride, clearance context
Stage 3: add terrain context such as flat/rough/slope/friction
Stage 4: add obstacle context such as distance and height
Stage 5: add bounded perception features or height-map embeddings
```

Each stage needs its own manifest/schema update, collector support, fixed eval suite, randomized eval suite, and acceptance gate before hardware exposure.

## Safety boundary

The context-conditioned policy is still a low-level physical controller. It must remain below Soridormi's safety layers:

```text
joint limits
action clipping
fall detection
runtime watchdog
emergency stop
sim acceptance gates before hardware
```

Chromie or any future planner may choose skill/context values, but Soridormi must validate ranges and reject unavailable, unsupported, or unsafe requests before the policy runs.
