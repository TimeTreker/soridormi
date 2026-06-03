# Soridormi M8G scripted social acceptance gates

M8G adds an acceptance layer for the head/neck-only scripted social skills that
were promoted in M8E/M8F. The goal is to keep social behaviors inspectable before
more skills are promoted from `planned` to `available_sim_experimental`.

The acceptance command has two modes:

- **dry-run trajectory mode**: validates the planned head-pose trajectory without
  connecting to MuJoCo;
- **live MuJoCo mode**: streams the same skills to an already-running MuJoCo sim
  and reports observed joint ranges plus base-height fall telemetry.

No hardware backend is exposed.

## Default acceptance cases

The default gate checks the scripted social skills that are currently promoted for MuJoCo validation:

| Skill | Gate |
| --- | --- |
| `look_direction` | Produces a bounded visible yaw target. |
| `nod_yes` | Moves only `head_pitch`, reaches down/up ranges, and keeps yaw/roll/neck pitch neutral. |
| `shake_no` | Moves only `head_yaw`, reaches left/right ranges, and keeps pitch/roll/neck pitch neutral. |
| `bow` | Moves only `neck_pitch`/`head_pitch`, reaches a shallow down pose, holds briefly, and returns neutral. |

For `nod_yes`, `shake_no`, and `bow`, non-moving axes must remain at zero in the
commanded trajectory. This directly protects the behavior expectation that
`shake_no` starts straight, turns left/right, and returns straight without adding
a head-down motion.

## Dry-run validation

```bash
./scripts/evaluate_scripted_social_skills.sh --json | python -m json.tool
```

Limit the gate to one skill:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --skill shake_no \
  --json | python -m json.tool
```

## Live MuJoCo validation

Start the simulator explicitly:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then evaluate the scripted social gates in a second terminal:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --execute \
  --backend mujoco \
  --require-observed
```

The wrapper runs inside `compose.sim.yaml` service `runtime`, matching the other
Soridormi sim/runtime scripts. Host Python does not need `pyzmq`.

## Base-height fall telemetry

`scripted_head_skill` now records live base-height telemetry when MuJoCo provides
`RobotState.base_position_xyz`:

- `observed_min_base_height_m`
- `observed_max_base_height_m`
- `final_base_height_m`
- `fall_height_m`
- `fallen`

The default fall threshold is `0.14m`, matching the existing RL environment
fallback. You can override it for debugging:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --execute \
  --backend mujoco \
  --fall-height-m 0.12
```

The single-skill runner also prints the same live telemetry:

```bash
./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"medium","duration_s":2.0}' \
  --backend mujoco \
  --control-hz 50
```

## Validation

```bash
PYTHONPATH=src pytest -q \
  tests/test_scripted_social_acceptance_m8g.py \
  tests/test_scripted_social_skill_m8e.py \
  tests/test_skill_execution_m7c.py \
  tests/test_skill_sim_execution_m7d.py \
  tests/test_skill_manifest_m7.py \
  tests/test_skill_manifest_cli_m7.py \
  tests/test_scenario_curriculum_m8b.py \
  tests/test_scenario_manifest_m8.py

python -m compileall -q src tests
bash -n scripts/run_scripted_social_skill_in_sim.sh
bash -n scripts/evaluate_scripted_social_skills.sh
```


## M8H neutral-home gate

The acceptance suite now includes `neutral_head` because the social manifest uses it as the fallback for scripted head skills. In dry-run mode the gate verifies that the planned command remains at the neutral straight-ahead pose. In live MuJoCo mode it also reports base-height stability like the other social skills.

Run the complete scripted social gate set after changing any head trajectory code:

```bash
./scripts/evaluate_scripted_social_skills.sh --json | python -m json.tool
```

For live validation, start MuJoCo first with the `open_duck_forward` profile and then run:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --execute \
  --backend mujoco \
  --require-observed
```

## M8I bow acceptance gate

The acceptance suite now includes `bow`. Dry-run acceptance verifies that `bow` commands a visible negative `head_pitch` range, allows only bounded `neck_pitch`/`head_pitch` motion, and keeps `head_yaw` and `head_roll` neutral. Live acceptance reuses the same base-height fall telemetry used for `nod_yes` and `shake_no`.

Run only the bow gate:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --skill bow \
  --json | python -m json.tool
```

Live MuJoCo validation:

```bash
./scripts/evaluate_scripted_social_skills.sh \
  --skill bow \
  --execute \
  --backend mujoco \
  --require-observed
```
