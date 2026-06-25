# Soridormi Social Eyes

Soridormi adds simulated social eyes as a MuJoCo overlay instead of editing the
official Open Duck Mini XML submodule. The overlay is visual-only: it adds
forward-face-mounted, non-colliding open-eye and closed-eye geoms under the
robot's `head_assembly` link, keeps the actuator list unchanged, and does not
alter policy observation or action dimensions. The default open eyes are thin
round `0.02m` radius face discs at the hand-tuned anchor
`x=0.01`, `y=+/-0.04`, `z=-0.06`, then pitched `90deg`
(`quat="0.707107 0 0.707107 0"`) so they sit on the forward face plane instead
of looking like balls mounted on top of the head.

The eyes are enabled by default when starting the Soridormi MuJoCo server:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Disable them only when validating a strict official visual model:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera --no-social-eyes
```

The open eyes move with the existing head/neck social skills, so Chromie or
another planner should continue to request structured Soridormi skills such as
`look_at_person`, `express_attention`, `nod_yes`, and `shake_no`. There are no
independent eye motors in this first version.

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
  --json
```
