# Soridormi look-target provider boundary

Soridormi resolves gaze targets itself. A caller such as Chromie names whom to
look at with `target_ref` and never supplies yaw or pitch:

```text
Chromie: look_at_person(target_ref="requester")
  -> Soridormi resolves the person from its own scene observation
  -> bounded look_at_person head trajectory toward the observed bearing
  -> MuJoCo actuator commands
```

`"requester"` is the agreed referent for the person who made the current
request; any other `target_ref` must match exactly one observed marker by its
identifying words. In simulation the scene markers are read through the
simulator (not camera perception) and carry no elevation, so the resolved
pitch is level. The skill, activity, and task MCP paths reject caller-supplied
`target_yaw_rad`/`target_pitch_rad`.

Temporary mocked requester position: Soridormi cannot perceive people yet, so in
simulation a `"requester"` gaze that finds no scene actor resolves to a declared
mocked position at `SORIDORMI_MOCK_REQUESTER_BEARING_RAD` (robot-frame yaw,
default `0.0`, straight ahead). The result keeps `status: resolved` and adds
`source: mocked_requester_position` and `object_ref: mocked_requester`, so it is
never mistaken for perception. A scene actor tagged `"requester"` still wins.
Set the variable to `off` to report `not_observed` instead. This stand-in is
replaced when a person-perception skill exists; it never applies outside sim.

When Soridormi cannot resolve the target, the skill ends without motion as a
declared non-completion with `completed: false`, a summary naming what was not
found, and a typed `target_resolution`:

| `status` | Meaning |
| --- | --- |
| `resolved` | One observed target within the head range; the skill executes. |
| `not_observed` | No observed marker matches the target. |
| `ambiguous` | More than one observed marker matches; Soridormi does not pick one. |
| `outside_head_range` | The target is observed beyond the head yaw range (or the stricter walking-overlay envelope); turning the body is a separate skill Chromie selects. |
| `perception_unavailable` | The mode has no perception to locate the target (outside simulation today). |

Soridormi never substitutes a straight-ahead (0/0) look and reports it as
looking at the person.

## Operator target tools

The provider below converts already-structured target hints into bounded
`look_at_person` yaw/pitch for operator tools, scripted acceptance, and future
perception handoff files. It is not the path a Chromie request takes.

social eye behavior changes the default gaze behavior: after acquiring a person target,
`look_at_person` now keeps looking at that target instead of immediately
returning to neutral. Use `--end-mode return_neutral` only for tests or explicit
"glance then reset" behavior.

## Supported target sources

Provide exactly one source.

### Manual yaw/pitch

```bash
./scripts/run_look_at_person_target.sh \
  --target-yaw-rad 0.30 \
  --target-pitch-rad -0.06 \
  --duration-s 4.0 \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool
```

### Image-point stub

This is not a detector. It assumes an upstream system already selected the
person center in normalized image coordinates. The point `(0.5, 0.5)` is image
center. Image y increases downward, so targets above center produce positive
pitch.

```bash
./scripts/run_look_at_person_target.sh \
  --image-x-norm 0.75 \
  --image-y-norm 0.45 \
  --duration-s 4.0 \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool
```

The default field of view is 60 degrees horizontal and 45 degrees vertical. You
can override it:

```bash
./scripts/run_look_at_person_target.sh \
  --image-x-norm 0.75 \
  --image-y-norm 0.45 \
  --horizontal-fov-rad 1.0471975512 \
  --vertical-fov-rad 0.7853981634 \
  --duration-s 4.0 \
  --backend mujoco \
  --dry-run
```

### JSON fixture

Use this for deterministic tests and future perception handoff files.

```json
{
  "target_ref": "person",
  "image_x_norm": 0.75,
  "image_y_norm": 0.45,
  "confidence": 0.8
}
```

Then run:

```bash
./scripts/run_look_at_person_target.sh \
  --target-json /tmp/person_target.json \
  --duration-s 4.0 \
  --backend mujoco \
  --resolve-only \
  --json | python -m json.tool
```

## Bounds

The operator provider clamps to the `look_at_person` manifest intent:

- yaw: `[-0.55, 0.55]` rad
- pitch: `[-0.25, 0.20]` rad
- confidence: `[0.0, 1.0]`

Clamping keeps operator inputs from commanding unsafe head offsets. The
Chromie-facing runtime path does not clamp a resolved target into range; it
reports `outside_head_range` instead.

## Live MuJoCo validation

Start MuJoCo first:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Second terminal:

```bash
./scripts/run_look_at_person_target.sh \
  --image-x-norm 0.75 \
  --image-y-norm 0.45 \
  --duration-s 4.0 \
  --backend mujoco
```

To explicitly return to neutral after the gaze hold:

```bash
./scripts/run_look_at_person_target.sh \
  --image-x-norm 0.75 \
  --image-y-norm 0.45 \
  --duration-s 4.0 \
  --end-mode return_neutral \
  --backend mujoco
```

The wrapper runs inside the runtime Docker container, matching the other
Soridormi sim tools. Hardware remains disabled.
