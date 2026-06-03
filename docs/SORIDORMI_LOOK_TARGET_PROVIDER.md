# Soridormi look-target provider boundary

M8K made `look_at_person` executable in MuJoCo, but it intentionally did not run
camera perception. M8L adds the structured target-provider boundary that future
perception, Chromie, or a planner can call before the scripted head executor.

The contract is:

```text
camera/person detector/planner/Chromie
  -> structured target yaw/pitch
  -> look_at_person scripted head trajectory
  -> MuJoCo actuator commands
```

Soridormi still does **not** know where the person is by itself. This patch only
converts already-structured target hints into bounded `look_at_person` arguments.

M8M changes the default gaze behavior: after acquiring a person target,
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

The provider clamps to the `look_at_person` manifest intent:

- yaw: `[-0.55, 0.55]` rad
- pitch: `[-0.25, 0.20]` rad
- confidence: `[0.0, 1.0]`

Clamping keeps upstream modules from commanding unsafe head offsets.

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
