# Soridormi body concurrency

## Purpose

Soridormi executes compatible physical abilities concurrently while retaining
one final motor-command and physical-safety authority.

Speech and singing are not Soridormi body members. They remain a peer Chromie
Speaking Execution Lane linked by `coordination_id`.

## Ability classes

### Subtle expression

Subtle expressions do not own the primary locomotion objective. They are split
by control coupling:

- `independent_output`: eye or display expression with no motor writes;
- `body_command_overlay`: bounded joint targets composed into the final body
  motor command;
- `standalone_body_motion`: small-looking gestures that still require exclusive
  body control.

The visual `blink_eyes` skill is independent. `look_direction` and
`look_at_person` are bounded head overlays. Nodding, shaking, bowing,
`neutral_head`, and `express_attention` remain standalone body motions until
separately qualified during locomotion.

### Locomotion / whole-body

Locomotion skills own the single primary body objective. An activity may have
zero or one primary locomotion member.

## Resource model

Every executable skill declares:

```text
ability_class
control_coupling
write_resources
safety_preemption
optional locomotion envelope
```

Current resources are:

```text
body.primary_motion  primary locomotion or standalone body control
body.head_pose       bounded head/neck command overlay
visual.eyes          independent visual eye output
```

Only one member may write a resource. Resource compatibility, not the skill
name, determines whether members can coexist.

## Current supported concurrency

Supported in the runtime-backed simulation adapter:

- locomotion plus `blink_eyes`;
- locomotion plus bounded `look_direction`;
- locomotion plus bounded `look_at_person`;
- locomotion plus one bounded gaze overlay plus eye blinking;
- standalone head/body gesture plus eye blinking;
- visual-only body activities.

Rejected by plan validation:

- two primary locomotion members;
- two writers for the same eye or head resource;
- bowing, nodding, shaking, or other standalone body motion during locomotion;
- a head overlay outside its declared locomotion yaw/pitch envelope;
- speech or singing as a Soridormi body member;
- raw motor, torque, joint, pose, or policy-action authority from Chromie.

## MCP surface

```text
soridormi.activity.get_capabilities
soridormi.activity.create_plan
soridormi.activity.execute_plan
soridormi.activity.status
soridormi.activity.cancel
```

The authoritative planner selects exact body skills first. The Activity
Execution Lane then creates one opaque Soridormi body-activity plan.

Example:

```json
{
  "coordination_id": "interaction-123",
  "members": [
    {
      "member_id": "walk",
      "skill_id": "walk_velocity",
      "parameters": {
        "vx_mps": 0.12,
        "vy_mps": 0.0,
        "yaw_radps": 0.0,
        "duration_s": 4.0
      }
    },
    {
      "member_id": "gaze",
      "skill_id": "look_at_person",
      "parameters": {
        "target_ref": "person_1",
        "target_yaw_rad": 0.08,
        "target_pitch_rad": -0.04,
        "duration_s": 4.0
      }
    },
    {
      "member_id": "blink",
      "skill_id": "blink_eyes",
      "parameters": {"count": 4},
      "optional": true
    }
  ]
}
```

Chromie may run a song under the same `coordination_id`, but the song remains a
peer speaking-lane execution and is not sent to this API.

## Command composition

The current runtime has no production WBC backend. It therefore implements the
same authority principle through a bounded body-command composer:

```text
locomotion policy MotorCommand
+ validated head-pose overlay
= one final MotorCommand sent to the robot API
```

Independent visual expressions use the separate visual-expression API and
never write motor commands.

A future WBC implementation may replace the composer internals, but it must
preserve the same resource, safety, cancellation, and evidence contracts.

## Safety

Safety surrounds all body lanes; it is not an ordinary schedulable peer.
Soridormi may reject a plan, stop physical members, restore a neutral visual
output, enter safe hold, or emergency-stop without waiting for Chromie.

Current implementation provides:

- static resource and concurrency-envelope validation before execution;
- one primary motor-command authority;
- activity-specific cancellation;
- global motion stop and emergency stop;
- per-member status and aggregate reconciliation;
- neutral eye restoration after completion, cancellation, or failure.

Not yet claimed:

- live balance-margin-based overlay reduction or suspension;
- autonomous person tracking or target perception;
- hardware execution;
- concurrent whole-body gestures that have not passed simulator qualification.
