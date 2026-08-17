# Soridormi Visual Body Overlay

Soridormi adds simulated social eyes and simple arms as a generated MuJoCo
overlay instead of editing the official Open Duck Mini XML or URDF. The overlay
is visual-only: it keeps the actuator list unchanged and does not alter policy
observation or action dimensions.

The eyes are forward-face-mounted, non-colliding open-eye and closed-eye geoms
under the robot's `head_assembly` link. The default open eyes are thin
round `0.02m` radius face discs at the hand-tuned anchor
`x=0.01`, `y=+/-0.04`, `z=-0.06`, then pitched `90deg`
(`quat="0.707107 0 0.707107 0"`) so they sit on the forward face plane instead
of looking like balls mounted on top of the head.

The arms use body-overlapping shoulder mounts plus shoulder, upper-arm, elbow,
forearm, cuff, wrist, palm, and finger geoms attached to `trunk_assembly`. Every
display pose now preserves an `0.082m` upper arm and `0.070m` forearm, giving a
consistent forearm-to-upper-arm ratio of about `0.85` instead of stretching the
segments to arbitrary pose coordinates. The mounts bridge each shoulder into
the tapered torso instead of leaving an air gap. Short orange cylindrical
housings and graphite axle pins give the shoulders and elbows the compact hinge
language used by Open Duck rather than a ball-joint appearance. They add no
joints, actuators,
inertials, sensors, or contacts. Every arm geom explicitly declares `contype=0`
and `conaffinity=0`. A short white cuff now ends before a narrow orange wrist;
the wrist connects to a flattened, lengthened palm instead of being hidden in a
large spherical hand. The four fingers follow the human MCP/PIP/DIP layout with
three tapered phalange segments, while the opposing thumb uses a separate
CMC/MCP/IP chain. Only the root knuckles retain a muted warm accent; smaller
PIP/DIP knuckles use a quiet warm gray. Each fixed hand pose applies a modest,
anatomically ordered finger curl. These knuckles remain non-contact visual
geoms, not MuJoCo kinematic `joint` elements or a claim of physical finger
actuation. Raised and forward gestures use explicit wrist directions instead
of simply continuing every hand along the forearm.

The same generated cosmetic profile adds rounded off-white shin shells plus
compact orange ankle accents to the existing moving lower-leg bodies. The
official thigh panels remain unobscured. The shells follow the knee and ankle
hierarchy but do not replace its joints, collision geometry, inertials, or
controls. This keeps the friendly silhouette entirely visual while preserving
the official locomotion model.

Nine fixed display poses are generated: `rest`, `reach`, `hold`, `place`,
`wave_up`, `wave_out`, `celebrate`, `welcome_open`, and `welcome_close`. Pose
changes only switch the alpha channel of those geom sets; they do not write
MuJoCo `qpos`, `qvel`, controls, or policy state.
The fixed coordinates keep the visible arm geometry outside the official leg
geometry in the home pose. A compiled-model regression requires at least
`0.015m` separation there; bounded leg-motion sampling is retained as visual
inspection evidence rather than a physical collision or hardware claim.

The eyes and cosmetic limbs are enabled by default when starting the Soridormi
MuJoCo server:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Disable both when validating a strict official visual model:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --no-social-eyes --no-visual-arms
```

`SORIDORMI_MUJOCO_VISUAL_ARMS` is owned by the MuJoCo simulator launcher. Its
default is `1`; supported values are `1` and `0`, mirrored by `--visual-arms`
and `--no-visual-arms`. It changes only generated display geometry and is not a
provider capability or hardware-availability switch. It can be removed when a
future canonical robot appearance asset replaces generated overlays without
changing the official locomotion baseline.

The open eyes move with the existing head/neck social skills, so Chromie or
another planner should continue to request structured Soridormi skills such as
`look_at_person`, `express_attention`, `nod_yes`, and `shake_no`. There are no
independent eye motors in this first version.

The simulation-only resource acquisition/delivery mock uses the fixed arm poses
as fail-soft decoration: acquisition displays `reach` then `hold`; handover
displays `place` then `rest`. The arm display never establishes acquisition or
delivery. Only the existing validated `resource_outcome` evidence can complete
that provider capability, and it remains marked `mocked_simulation=true`.

Three display-only social skills use the same overlay through the independent
`visual.arms` body-activity resource:

- `wave_hand` alternates `wave_up` and `wave_out` for one selected side;
- `celebrate` briefly shows both arms raised;
- `hug_gesture` shows an open-and-close arm expression without claiming person
  contact or a physical hug.

These skills are safe to compose with locomotion only because they are visual
outputs with no dynamics. `point_direction` and `high_five` remain rejected:
the current robot has neither target-aligned arm control nor contact-safe arm
hardware.

Blinking is exposed as the high-level `blink_eyes` ability. It uses a
simulator-only visual-expression API to toggle the generated open-eye geoms off
and closed-eye geoms on briefly. Keep testing it through the normal Chromie and
Soridormi service path instead of adding one script per ability:

```bash
cd /home/chromie/github/soridormi
./scripts/start_soridormi_mujoco.sh --profile open_duck_forward --viewer --follow-camera --keep-running

cd /home/chromie/github/chromie
./scripts/start_chromie.sh --mcp-url http://127.0.0.1:8000/mcp --keep-services
./scripts/run_voice_mujoco_text_case.sh "please blink your eyes" \
  --no-speaker \
  --expect-skill soridormi.blink_eyes
```

The combined Chromie voice/MuJoCo launcher may still be useful as a demo helper,
but the normal architecture keeps Chromie and Soridormi started separately.

For a dry-run of the low-level Soridormi plan without connecting to MuJoCo, use
the Python module directly:

```bash
python -m soridormi_runtime.visual_expression_skill blink_eyes \
  --args '{"count":2}' \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool
```

Visual arm gestures have an equivalent dry-run entry point:

```bash
python -m soridormi_runtime.visual_arm_gesture_skill wave_hand \
  --args '{"side":"right","count":2,"duration_s":2.4}' \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool
```

For a direct generator smoke test:

```bash
python -m soridormi_sim.social_eye_scene \
  --base /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml \
  --output /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/soridormi_social_eyes_scene.xml \
  --visual-arms \
  --json
```
