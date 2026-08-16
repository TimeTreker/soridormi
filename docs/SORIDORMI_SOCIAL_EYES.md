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
forearm, and hand geoms under `trunk_assembly`. The mounts bridge each shoulder
into the tapered torso instead of leaving a visible air gap. Chunkier white
limb shells, graphite joints, and gold five-digit hands reuse the official
robot's visual language. Each hand has four unequal rounded fingers aligned with
the forearm plus a shorter opposing thumb, with every digit embedded into the
palm for a continuous silhouette. They add no joints, actuators, inertials,
sensors, or contacts. Every arm and finger geom explicitly declares `contype=0`
and `conaffinity=0`. Four fixed display poses are generated: `rest`, `reach`,
`hold`, and `place`. Pose changes only switch the alpha channel of those geom
sets; they do not write MuJoCo `qpos`, `qvel`, controls, or policy state, and
finger visibility is not grasp evidence.
The fixed coordinates keep the visible arm geometry outside the official leg
geometry in the home pose. A compiled-model regression requires at least
`0.015m` separation there; bounded leg-motion sampling is retained as visual
inspection evidence rather than a physical collision or hardware claim.

The eyes and visual arms are enabled by default when starting the Soridormi
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

For a direct generator smoke test:

```bash
python -m soridormi_sim.social_eye_scene \
  --base /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml \
  --output /workspaces/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/soridormi_social_eyes_scene.xml \
  --visual-arms \
  --json
```
